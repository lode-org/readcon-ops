//! Exactly rounded sums for order- and rank-invariant reductions.
//!
//! The method follows the Ozaki scheme's core step: each finite `f64` is an
//! integer mantissa times a power of two, so it can be added into a wide
//! fixed-point integer with no rounding. Products of two or three `f64` are
//! formed as exact integer products of the mantissas (at most 159 bits),
//! split into `u128` terms. The accumulator rounds once, to nearest with
//! ties to even, when the result is read.
//!
//! Integer addition is associative and commutative, so the result does not
//! depend on the order of the terms, on how they are split across threads
//! or MPI ranks, or on the order in which partial accumulators are merged.
//! A partial accumulator serialises to fixed-size bytes for that merge.
//!
//! Non-finite inputs follow IEEE rules for the final value: any NaN, or
//! both infinities, gives NaN; otherwise an infinity gives that infinity.
//! An exact zero is returned as `+0.0`.

/// Fixed-point position of limb bit 0: below the smallest three-way
/// product of subnormals (2^-3222).
const LSB_EXP: i32 = -3264;
/// 32-bit limbs held in `i64` so that carries can be deferred.
pub const N_LIMBS: usize = 208;
/// Limb additions between carry propagations (each adds < 2^32 per limb).
const NORM_EVERY: u32 = 1 << 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactAcc {
    limbs: Vec<i64>,
    pending: u32,
    nan: bool,
    pos_inf: bool,
    neg_inf: bool,
}

impl Default for ExactAcc {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn decompose(x: f64) -> (u64, i32, bool) {
    let b = x.to_bits();
    let neg = b >> 63 == 1;
    let e = ((b >> 52) & 0x7ff) as i32;
    let f = b & ((1u64 << 52) - 1);
    if e == 0 {
        (f, -1074, neg)
    } else {
        (f | (1u64 << 52), e - 1075, neg)
    }
}

impl ExactAcc {
    pub fn new() -> Self {
        ExactAcc {
            limbs: vec![0; N_LIMBS],
            pending: 0,
            nan: false,
            pos_inf: false,
            neg_inf: false,
        }
    }

    fn nonfinite(&mut self, v: f64) {
        if v.is_nan() {
            self.nan = true;
        } else if v > 0.0 {
            self.pos_inf = true;
        } else {
            self.neg_inf = true;
        }
    }

    /// Add `mag * 2^lsb_exp` with sign `neg`.
    #[inline]
    fn add_int(&mut self, mag: u128, lsb_exp: i32, neg: bool) {
        if mag == 0 {
            return;
        }
        let s = (lsb_exp - LSB_EXP) as usize;
        let (idx, off) = (s / 32, s % 32);
        for k in 0..4 {
            let c = ((mag >> (32 * k)) & 0xffff_ffff) as u64;
            if c == 0 {
                continue;
            }
            let v = c << off;
            let (lo, hi) = ((v & 0xffff_ffff) as i64, (v >> 32) as i64);
            if neg {
                self.limbs[idx + k] -= lo;
                self.limbs[idx + k + 1] -= hi;
            } else {
                self.limbs[idx + k] += lo;
                self.limbs[idx + k + 1] += hi;
            }
        }
        self.pending += 1;
        if self.pending >= NORM_EVERY {
            self.normalize();
        }
    }

    /// Add `x` exactly.
    #[inline]
    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return self.nonfinite(x);
        }
        let (m, e, n) = decompose(x);
        self.add_int(m as u128, e, n);
    }

    /// Add the exact product `a * b`.
    #[inline]
    pub fn add_prod(&mut self, a: f64, b: f64) {
        if !(a.is_finite() && b.is_finite()) {
            return self.nonfinite(a * b);
        }
        let ((ma, ea, na), (mb, eb, nb)) = (decompose(a), decompose(b));
        self.add_int(ma as u128 * mb as u128, ea + eb, na ^ nb);
    }

    /// Add the exact product `a * b * c`.
    #[inline]
    pub fn add_prod3(&mut self, a: f64, b: f64, c: f64) {
        if !(a.is_finite() && b.is_finite() && c.is_finite()) {
            return self.nonfinite(a * b * c);
        }
        let ((ma, ea, na), (mb, eb, nb), (mc, ec, nc)) = (decompose(a), decompose(b), decompose(c));
        let p = ma as u128 * mb as u128;
        let (pl, ph) = (p & u64::MAX as u128, p >> 64);
        let (e, n) = (ea + eb + ec, na ^ nb ^ nc);
        self.add_int(pl * mc as u128, e, n);
        self.add_int(ph * mc as u128, e + 64, n);
    }

    /// Propagate carries: limbs below the top end in `[0, 2^32)`.
    pub fn normalize(&mut self) {
        for i in 0..N_LIMBS - 1 {
            let c = self.limbs[i] >> 32;
            self.limbs[i] -= c << 32;
            self.limbs[i + 1] += c;
        }
        self.pending = 0;
    }

    /// Add another partial sum. Order of merges does not matter.
    pub fn merge(&mut self, other: &ExactAcc) {
        self.normalize();
        let mut o = other.clone();
        o.normalize();
        for (a, b) in self.limbs.iter_mut().zip(&o.limbs) {
            *a += *b;
        }
        self.nan |= o.nan;
        self.pos_inf |= o.pos_inf;
        self.neg_inf |= o.neg_inf;
        self.normalize();
    }

    /// Fixed-size little-endian encoding (flags byte plus limbs).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut a = self.clone();
        a.normalize();
        let mut out = Vec::with_capacity(1 + 8 * N_LIMBS);
        out.push(u8::from(a.nan) | u8::from(a.pos_inf) << 1 | u8::from(a.neg_inf) << 2);
        for l in &a.limbs {
            out.extend_from_slice(&l.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != 1 + 8 * N_LIMBS {
            return None;
        }
        let limbs = b[1..]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some(ExactAcc {
            limbs,
            pending: 0,
            nan: b[0] & 1 != 0,
            pos_inf: b[0] & 2 != 0,
            neg_inf: b[0] & 4 != 0,
        })
    }

    /// The sum rounded once to the nearest `f64`, ties to even.
    pub fn round(&self) -> f64 {
        if self.nan || (self.pos_inf && self.neg_inf) {
            return f64::NAN;
        }
        if self.pos_inf {
            return f64::INFINITY;
        }
        if self.neg_inf {
            return f64::NEG_INFINITY;
        }
        let mut a = self.clone();
        a.normalize();
        let neg = a.limbs[N_LIMBS - 1] < 0;
        if neg {
            for l in a.limbs.iter_mut() {
                *l = -*l;
            }
            a.normalize();
        }
        let Some(h) = (0..N_LIMBS).rev().find(|&i| a.limbs[i] != 0) else {
            return 0.0;
        };
        let top = a.limbs[h] as u64;
        let b = h * 32 + 63 - top.leading_zeros() as usize;
        let bit = |i: usize| (a.limbs[i / 32] as u64 >> (i % 32)) & 1;
        let min_sh = (-1074 - LSB_EXP) as usize;
        let mut sh = if b >= 52 {
            (b - 52).max(min_sh)
        } else {
            min_sh
        };
        let mut m: u64 = 0;
        for i in (sh..=b.max(sh)).rev() {
            m = (m << 1) | bit(i);
        }
        let round_bit = bit(sh - 1);
        let sticky = {
            let k = sh - 1;
            let below_limb = (0..k / 32).any(|i| a.limbs[i] != 0);
            let mask = (1u64 << (k % 32)) - 1;
            below_limb || (a.limbs[k / 32] as u64 & mask) != 0
        };
        if round_bit == 1 && (sticky || m & 1 == 1) {
            m += 1;
            if m == 1u64 << 53 {
                m >>= 1;
                sh += 1;
            }
        }
        let q = sh as i32 + LSB_EXP;
        let mag = if m < (1u64 << 52) {
            f64::from_bits(m)
        } else {
            let biased = q + 1075;
            if biased >= 2047 {
                f64::INFINITY
            } else {
                f64::from_bits((biased as u64) << 52 | (m - (1u64 << 52)))
            }
        };
        if neg {
            -mag
        } else {
            mag
        }
    }
}

/// Exactly rounded sum of `xs`.
pub fn sum(xs: &[f64]) -> f64 {
    let mut a = ExactAcc::new();
    xs.iter().for_each(|&x| a.add(x));
    a.round()
}

/// Exactly rounded dot product of `x` and `y`.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut a = ExactAcc::new();
    x.iter().zip(y).for_each(|(&p, &q)| a.add_prod(p, q));
    a.round()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_values_round_trip() {
        for x in [
            0.0,
            1.0,
            -1.0,
            0.1,
            1e300,
            -1e-300,
            5e-324,
            f64::MAX,
            f64::MIN_POSITIVE,
        ] {
            assert_eq!(sum(&[x]).to_bits(), x.to_bits(), "{x}");
        }
    }

    #[test]
    fn cancellation_is_exact() {
        assert_eq!(sum(&[1e100, 1.0, -1e100]), 1.0);
        // The doubles nearest 0.1, 0.2 and 0.3 differ by exactly 2^-55.
        assert_eq!(sum(&[0.1, 0.2, -0.3]), 2f64.powi(-55));
    }

    #[test]
    fn ties_go_to_even() {
        // 1 + 2^-53 is a tie between 1 and 1 + 2^-52; even is 1.
        assert_eq!(sum(&[1.0, 2f64.powi(-53)]), 1.0);
        // 1 + 3 * 2^-53 rounds up to 1 + 2^-51.
        assert_eq!(
            sum(&[1.0, 2f64.powi(-53), 2f64.powi(-52)]),
            1.0 + 2f64.powi(-51)
        );
        // A sticky bit below the tie breaks it upward.
        assert_eq!(
            sum(&[1.0, 2f64.powi(-53), 2f64.powi(-100)]),
            1.0 + 2f64.powi(-52)
        );
    }

    #[test]
    fn overflow_and_nonfinite() {
        assert_eq!(sum(&[f64::MAX, f64::MAX]), f64::INFINITY);
        assert_eq!(sum(&[f64::MAX, f64::MAX, -f64::MAX]), f64::MAX);
        assert!(sum(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert_eq!(sum(&[1.0, f64::NEG_INFINITY]), f64::NEG_INFINITY);
    }

    #[test]
    fn products_are_exact() {
        let x = 1.0 + 2f64.powi(-30);
        // x * x = 1 + 2^-29 + 2^-60; subtracting 1 + 2^-29 leaves 2^-60.
        let mut a = ExactAcc::new();
        a.add_prod(x, x);
        a.add(-(1.0 + 2f64.powi(-29)));
        assert_eq!(a.round(), 2f64.powi(-60));
        let mut b = ExactAcc::new();
        b.add_prod3(x, x, x);
        b.add(-(1.0 + 3.0 * 2f64.powi(-30)));
        b.add(-3.0 * 2f64.powi(-60));
        assert_eq!(b.round(), 2f64.powi(-90));
    }

    #[test]
    fn order_and_merge_invariant() {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64 - 0.5) * 10f64.powi((s % 40) as i32 - 20)
        };
        let xs: Vec<f64> = (0..10_000).map(|_| next()).collect();
        let fwd = sum(&xs);
        let rev: Vec<f64> = xs.iter().rev().copied().collect();
        assert_eq!(fwd.to_bits(), sum(&rev).to_bits());
        let mut parts: Vec<ExactAcc> = xs
            .chunks(777)
            .map(|c| {
                let mut a = ExactAcc::new();
                c.iter().for_each(|&x| a.add(x));
                a
            })
            .collect();
        parts.reverse();
        let mut m = ExactAcc::new();
        for p in &parts {
            m.merge(&ExactAcc::from_bytes(&p.to_bytes()).unwrap());
        }
        assert_eq!(m.round().to_bits(), fwd.to_bits());
    }
}
