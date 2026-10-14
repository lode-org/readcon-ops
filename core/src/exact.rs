//! Exactly rounded sums for order- and rank-invariant reductions.
//!
//! The method follows the Ozaki scheme's core step: each finite `f64` is an
//! integer mantissa times a power of two, so it can be added into a wide
//! fixed-point integer with no rounding. Products of two or three `f64` are
//! formed as exact integer products of the mantissas (at most 159 bits),
//! split into `u128` terms. The accumulator rounds once, to nearest with
//! ties to even, when the result is read.
//!
//! [`sum`] and [`dot`] try a compensated sum of error-free product splits
//! first, and keep it only when a running error bound sits strictly inside
//! the correctly rounded bin. Longer products ([`dot_prod`], [`dot_prod3`])
//! stay on [`ExactAcc`]: once the bound is checked, the compensated loop is
//! not faster than the integer product. Both roundings are the exact sum, so
//! the bits do not depend on order.
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
    limbs: [i64; N_LIMBS],
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
            limbs: [0; N_LIMBS],
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
        let mut limbs = [0; N_LIMBS];
        for (i, c) in b[1..].chunks_exact(8).enumerate() {
            limbs[i] = i64::from_le_bytes(c.try_into().unwrap());
        }
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

const U: f64 = f64::from_bits(0x3ca0_0000_0000_0000); // 2^-53

#[inline]
fn next_up(x: f64) -> f64 {
    let bits = x.to_bits();
    if x.is_nan() || bits == f64::INFINITY.to_bits() {
        return x;
    }
    let abs = bits & 0x7fff_ffff_ffff_ffff;
    let next = if abs == 0 {
        1
    } else if bits == abs {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

#[inline]
fn next_down(x: f64) -> f64 {
    let bits = x.to_bits();
    if x.is_nan() || bits == f64::NEG_INFINITY.to_bits() {
        return x;
    }
    let abs = bits & 0x7fff_ffff_ffff_ffff;
    let next = if abs == 0 {
        0x8000_0000_0000_0001
    } else if bits == abs {
        bits - 1
    } else {
        bits + 1
    };
    f64::from_bits(next)
}

/// Compensated sum plus a strict overestimate of its absolute error.
struct Run {
    sum: f64,
    comp: f64,
    sigma: f64,
    /// Non-zero once a subnormal, non-finite, or broken error bound appears.
    flags: u64,
}

/// Subnormal or non-finite. Zero is allowed (it is exact).
#[inline(always)]
fn bad_class(x: f64) -> u64 {
    let b = x.to_bits();
    let exp = b & 0x7ff0_0000_0000_0000;
    let frac = b & 0x000f_ffff_ffff_ffff;
    u64::from((exp == 0 && frac != 0) || exp == 0x7ff0_0000_0000_0000)
}

impl Run {
    fn new() -> Self {
        Run {
            sum: 0.0,
            comp: 0.0,
            sigma: 0.0,
            flags: 0,
        }
    }

    /// Add one term. FastTwoSum is exact while every value stays normal;
    /// `flags` records a subnormal or an increment that underflowed.
    #[inline(always)]
    fn push(&mut self, x: f64) {
        let (a, b) = if self.sum.abs() >= x.abs() {
            (self.sum, x)
        } else {
            (x, self.sum)
        };
        let t = a + b;
        let delta = (a - t) + b;
        let c_new = self.comp + delta;
        // |fl(c+delta)-(c+delta)| < 2 U (|c|+|delta|). Adding a relative
        // 16 U afterwards covers that product and the sigma add rounding down.
        let scale = self.comp.abs() + delta.abs();
        let incr = (2.0 * U) * scale;
        self.sigma += incr;
        self.sigma += self.sigma * (16.0 * U);
        self.flags |= bad_class(x)
            | bad_class(t)
            | bad_class(delta)
            | bad_class(c_new)
            | u64::from(scale != 0.0 && incr == 0.0)
            | u64::from(!self.sigma.is_finite());
        self.comp = c_new;
        self.sum = t;
    }

    fn finish(self) -> Option<f64> {
        if self.flags != 0 || !self.sigma.is_finite() {
            return None;
        }
        let (a, b) = if self.sum.abs() >= self.comp.abs() {
            (self.sum, self.comp)
        } else {
            (self.comp, self.sum)
        };
        if (a != 0.0 && !a.is_normal()) || (b != 0.0 && !b.is_normal()) {
            return None;
        }
        let result = a + b;
        if !(result == 0.0 || result.is_normal()) {
            return None;
        }
        let delta = (a - result) + b;
        if !(delta == 0.0 || delta.is_normal()) {
            return None;
        }
        if result == 0.0 {
            return if self.sigma == 0.0 && delta == 0.0 {
                Some(0.0)
            } else {
                None
            };
        }
        let up = next_up(result);
        let down = next_down(result);
        if !up.is_finite() || !down.is_finite() {
            return None;
        }
        let gap = ((up - result) * 0.5).min((result - down) * 0.5);
        let raw = delta.abs() + self.sigma;
        let err = if raw == 0.0 { 0.0 } else { next_up(raw) };
        if err < gap {
            Some(result)
        } else {
            None
        }
    }
}

#[inline(always)]
fn split(run: &mut Run, a: f64, b: f64) -> (f64, f64) {
    run.flags |= bad_class(a) | bad_class(b);
    let hi = a * b;
    let lo = a.mul_add(b, -hi);
    run.flags |= bad_class(hi) | bad_class(lo);
    (hi, lo)
}

#[inline(always)]
fn try_values(n: usize, mut f: impl FnMut(usize) -> f64) -> Option<f64> {
    let mut run = Run::new();
    for i in 0..n {
        run.push(f(i));
    }
    run.finish()
}

#[inline(always)]
fn try_prod(n: usize, mut f: impl FnMut(usize) -> (f64, f64)) -> Option<f64> {
    let mut run = Run::new();
    for i in 0..n {
        let (a, b) = f(i);
        let (hi, lo) = split(&mut run, a, b);
        run.push(hi);
        if lo != 0.0 {
            run.push(lo);
        }
    }
    run.finish()
}

#[inline(always)]
fn try_prod3(n: usize, mut f: impl FnMut(usize) -> (f64, f64, f64)) -> Option<f64> {
    let mut run = Run::new();
    for i in 0..n {
        let (a, b, c) = f(i);
        let hi = a * b;
        let lo = a.mul_add(b, -hi);
        let h1 = hi * c;
        let l1 = hi.mul_add(c, -h1);
        run.flags |= bad_class(a) | bad_class(b) | bad_class(c) | bad_class(hi) | bad_class(lo);
        run.push(h1);
        if l1 != 0.0 {
            run.push(l1);
        }
        if lo != 0.0 {
            let h2 = lo * c;
            let l2 = lo.mul_add(c, -h2);
            run.flags |= bad_class(h2) | bad_class(l2);
            run.push(h2);
            if l2 != 0.0 {
                run.push(l2);
            }
        }
    }
    run.finish()
}

/// Exactly rounded sum of `xs`.
pub fn sum(xs: &[f64]) -> f64 {
    if let Some(v) = try_values(xs.len(), |i| xs[i]) {
        return v;
    }
    let mut a = ExactAcc::new();
    xs.iter().for_each(|&x| a.add(x));
    a.round()
}

/// Exactly rounded dot product of `x` and `y`.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    if x.len() == y.len() {
        if let Some(v) = try_prod(x.len(), |i| (x[i], y[i])) {
            return v;
        }
    }
    let mut a = ExactAcc::new();
    x.iter().zip(y).for_each(|(&p, &q)| a.add_prod(p, q));
    a.round()
}

/// Sum of `f(i).0 * f(i).1`, or `None` when the compensated bound is not strict.
pub fn sum_prod(n: usize, f: impl FnMut(usize) -> (f64, f64)) -> Option<f64> {
    try_prod(n, f)
}

/// Sum of `f(i).0 * f(i).1 * f(i).2`, or `None` when the bound is not strict.
pub fn sum_prod3(n: usize, f: impl FnMut(usize) -> (f64, f64, f64)) -> Option<f64> {
    try_prod3(n, f)
}

/// Sum of `f(i).0 * f(i).1` rounded once.
///
/// The compensated pre-pass is only a win for short sums. A Kabsch dot of a
/// few hundred atoms is faster as a straight integer product, and a rejected
/// bound would pay both, so this always uses [`ExactAcc`].
#[inline]
pub fn dot_prod(n: usize, mut f: impl FnMut(usize) -> (f64, f64)) -> f64 {
    let mut a = ExactAcc::new();
    for i in 0..n {
        let (p, q) = f(i);
        a.add_prod(p, q);
    }
    a.round()
}

/// Sum of `f(i).0 * f(i).1 * f(i).2` rounded once. See [`dot_prod`].
#[inline]
pub fn dot_prod3(n: usize, mut f: impl FnMut(usize) -> (f64, f64, f64)) -> f64 {
    let mut a = ExactAcc::new();
    for i in 0..n {
        let (p, q, r) = f(i);
        a.add_prod3(p, q, r);
    }
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

    #[test]
    fn compensated_matches_exact_acc() {
        let mut s = 0x0fed_cba9_8765_4321u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64 - 0.5) * 8.0
        };
        for n in [1usize, 3, 17, 500, 1_500] {
            let x: Vec<f64> = (0..n).map(|_| next()).collect();
            let y: Vec<f64> = (0..n).map(|_| next()).collect();
            let z: Vec<f64> = (0..n).map(|_| next().abs() + 0.1).collect();
            let mut slow = ExactAcc::new();
            x.iter().zip(&y).for_each(|(&p, &q)| slow.add_prod(p, q));
            assert_eq!(dot(&x, &y).to_bits(), slow.round().to_bits(), "dot {n}");
            assert!(sum_prod(n, |i| (x[i], y[i])).is_some(), "certified dot {n}");
            let mut slow = ExactAcc::new();
            for i in 0..n {
                slow.add_prod3(z[i], x[i], y[i]);
            }
            assert_eq!(
                dot_prod3(n, |i| (z[i], x[i], y[i])).to_bits(),
                slow.round().to_bits(),
                "dot3 {n}"
            );
            assert!(
                sum_prod3(n, |i| (z[i], x[i], y[i])).is_some(),
                "certified dot3 {n}"
            );
            let xs: Vec<f64> = (0..n).map(|_| next()).collect();
            let mut slow = ExactAcc::new();
            xs.iter().for_each(|&v| slow.add(v));
            assert_eq!(sum(&xs).to_bits(), slow.round().to_bits(), "sum {n}");
        }
        // Wide cancellation still matches, whether or not the bound certifies.
        let m: Vec<f64> = (0..500).map(|i| 1.0 + i as f64 * 0.11).collect();
        let tot = sum(&m);
        let w: Vec<f64> = m.iter().map(|x| x / tot).collect();
        let a: Vec<f64> = (0..500).map(|i| (i as f64) * 0.017 - 3.0).collect();
        let b: Vec<f64> = (0..500).map(|i| (i as f64) * -0.013 + 1.5).collect();
        assert!(
            sum_prod3(500, |i| (w[i], a[i], b[i])).is_some(),
            "covariance-shaped triple product did not certify"
        );
        let xs = [1e100, 1.0, -1e100, 1e-20, -1e-20];
        let mut slow = ExactAcc::new();
        xs.iter().for_each(|&v| slow.add(v));
        assert_eq!(sum(&xs).to_bits(), slow.round().to_bits());
    }
}
