//! Compiled kernels for readcon-ops.
//!
//! Coordinates are `&[[f64; 3]]` rows (the numpy `(n, 3)` C layout, used
//! without copying). Every reduction runs in a fixed order: per-element
//! work is split across rayon threads, but each sum is computed by one
//! thread in index order with four fixed lanes, so the bits do not depend
//! on the thread count. Rust does not contract `a * b + c` into an FMA
//! unless asked, so the same expression rounds the same way on every CPU.
//!
//! With the `exact` feature, each reduction rounds once. A compensated
//! pre-pass is kept only when its error bound proves the correctly rounded
//! value; otherwise the sum uses [`exact::ExactAcc`]. Either way the bits do
//! not depend on input order, thread count or rank count.

#[cfg(feature = "exact")]
pub mod exact;
#[cfg(feature = "python")]
mod py;

use nalgebra::Matrix3;
use rayon::prelude::*;

pub type Row = [f64; 3];

/// Same thresholds as `readcon_ops.match`.
pub const AXIS_RANK_TOL: f64 = 1e-8;
pub const RANK0_TOL: f64 = 1e-12;

/// Reduction mode. `Exact` needs the `exact` feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sum {
    Float,
    #[cfg(feature = "exact")]
    Exact,
}

/// Four-lane fixed-order sum of `f(i)` for `i in 0..n`.
#[inline]
fn lanes_sum(n: usize, f: impl Fn(usize) -> f64) -> f64 {
    let mut acc = [0.0f64; 4];
    let full = n / 4 * 4;
    let mut i = 0;
    while i < full {
        acc[0] += f(i);
        acc[1] += f(i + 1);
        acc[2] += f(i + 2);
        acc[3] += f(i + 3);
        i += 4;
    }
    for (k, j) in (full..n).enumerate() {
        acc[k] += f(j);
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Weighted centroid `sum_i w_i r_i` (weights already normalised).
pub fn centroid(r: &[Row], w: &[f64], mode: Sum) -> Row {
    match mode {
        Sum::Float => {
            let c = |d: usize| lanes_sum(r.len(), |i| w[i] * r[i][d]);
            [c(0), c(1), c(2)]
        }
        #[cfg(feature = "exact")]
        Sum::Exact => {
            let c = |d: usize| exact::dot_prod(r.len(), |i| (w[i], r[i][d]));
            [c(0), c(1), c(2)]
        }
    }
}

/// Unweighted RMSD `sqrt(mean_i |a_i - b_i|^2)`; 0 for no atoms.
pub fn rmsd(a: &[Row], b: &[Row], mode: Sum) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let d = |i: usize, k: usize| a[i][k] - b[i][k];
    let s = match mode {
        Sum::Float => lanes_sum(n, |i| {
            let (x, y, z) = (d(i, 0), d(i, 1), d(i, 2));
            x * x + y * y + z * z
        }),
        #[cfg(feature = "exact")]
        Sum::Exact => exact::dot_prod(n * 3, |t| {
            let d = a[t / 3][t % 3] - b[t / 3][t % 3];
            (d, d)
        }),
    };
    (s / n as f64).sqrt()
}

/// Weighted covariance `H[j][k] = sum_i w_i a_i[j] b_i[k]`.
pub fn covariance(a: &[Row], b: &[Row], w: &[f64], mode: Sum) -> Matrix3<f64> {
    let mut h = Matrix3::zeros();
    for j in 0..3 {
        for k in 0..3 {
            h[(j, k)] = match mode {
                Sum::Float => lanes_sum(a.len(), |i| w[i] * a[i][j] * b[i][k]),
                #[cfg(feature = "exact")]
                Sum::Exact => exact::dot_prod3(a.len(), |i| (w[i], a[i][j], b[i][k])),
            };
        }
    }
    h
}

/// Kinetic energy `sum_i 0.5 m_i |v_i|^2`.
pub fn kinetic_energy(m: &[f64], v: &[Row], mode: Sum) -> f64 {
    match mode {
        Sum::Float => lanes_sum(m.len(), |i| {
            0.5 * m[i] * (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2])
        }),
        #[cfg(feature = "exact")]
        Sum::Exact => exact::dot_prod3(m.len() * 3, |t| {
            let (i, k) = (t / 3, t % 3);
            (0.5 * m[i], v[i][k], v[i][k])
        }),
    }
}

/// Mean of `x` (for example a band average); NaN for an empty slice.
pub fn mean(x: &[f64], mode: Sum) -> f64 {
    let s = match mode {
        Sum::Float => lanes_sum(x.len(), |i| x[i]),
        #[cfg(feature = "exact")]
        Sum::Exact => exact::sum(x),
    };
    s / x.len() as f64
}

/// Result of one Kabsch superposition.
#[derive(Clone, Debug)]
pub struct Aligned {
    /// Row-form rotation (`moved = centered @ rotation + centroid_ref`).
    /// `None` when the covariance is rank 0 or rank 1: the caller then uses
    /// the Python path, which handles the collinear and point cases.
    pub rotation: Option<[[f64; 3]; 3]>,
    pub moved: Vec<Row>,
    pub rmsd: f64,
}

/// Weighted Kabsch superposition of `mobile` onto `reference` with the
/// determinant-sign guard, as in `readcon_ops.match._superimpose`.
pub fn kabsch(reference: &[Row], mobile: &[Row], w: &[f64], mode: Sum) -> Aligned {
    let cr = centroid(reference, w, mode);
    let cm = centroid(mobile, w, mode);
    let sub = |r: &[Row], c: Row| -> Vec<Row> {
        r.iter()
            .map(|p| [p[0] - c[0], p[1] - c[1], p[2] - c[2]])
            .collect()
    };
    let (rr, mm) = (sub(reference, cr), sub(mobile, cm));
    let translate_only = || {
        let moved: Vec<Row> = mm
            .iter()
            .map(|p| [p[0] + cr[0], p[1] + cr[1], p[2] + cr[2]])
            .collect();
        let r = rmsd(&moved, reference, mode);
        Aligned {
            rotation: None,
            moved,
            rmsd: r,
        }
    };
    if reference.len() < 2 {
        return translate_only();
    }
    let h = covariance(&rr, &mm, w, mode);
    let svd = h.svd(true, true);
    let (u, mut vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    // nalgebra does not sort singular values; order them like LAPACK.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
    let s: Vec<f64> = idx.iter().map(|&i| svd.singular_values[i]).collect();
    if s[0] <= RANK0_TOL || s[1] <= AXIS_RANK_TOL * s[0] {
        return Aligned {
            rotation: None,
            ..translate_only()
        };
    }
    let u = Matrix3::from_columns(&[u.column(idx[0]), u.column(idx[1]), u.column(idx[2])]);
    vt = Matrix3::from_rows(&[vt.row(idx[0]), vt.row(idx[1]), vt.row(idx[2])]);
    if (u * vt).determinant() < 0.0 {
        for k in 0..3 {
            vt[(2, k)] = -vt[(2, k)];
        }
    }
    let row = (u * vt).transpose();
    let rot = [
        [row[(0, 0)], row[(0, 1)], row[(0, 2)]],
        [row[(1, 0)], row[(1, 1)], row[(1, 2)]],
        [row[(2, 0)], row[(2, 1)], row[(2, 2)]],
    ];
    let moved: Vec<Row> = mm
        .iter()
        .map(|p| {
            let mut o = [0.0; 3];
            for (k, ok) in o.iter_mut().enumerate() {
                *ok = (p[0] * rot[0][k] + p[1] * rot[1][k]) + p[2] * rot[2][k] + cr[k];
            }
            o
        })
        .collect();
    let r = rmsd(&moved, reference, mode);
    Aligned {
        rotation: Some(rot),
        moved,
        rmsd: r,
    }
}

/// Align every frame onto `reference` in parallel; results in frame order.
pub fn kabsch_batch(reference: &[Row], frames: &[&[Row]], w: &[f64], mode: Sum) -> Vec<Aligned> {
    frames
        .par_iter()
        .map(|m| kabsch(reference, m, w, mode))
        .collect()
}

/// Displacements wrapped along the periodic axes of `cell` (rows are
/// lattice vectors): fractional coordinates are rounded to the nearest
/// image. This is the minimum image for orthorhombic cells; for a skewed
/// cell with all three axes periodic `readcon_ops` keeps using minimage.
pub fn min_image(dr: &[Row], cell: &[[f64; 3]; 3], pbc: [bool; 3]) -> Option<Vec<Row>> {
    let l = Matrix3::from_row_slice(&cell.concat());
    let inv = l.try_inverse()?;
    Some(
        dr.par_iter()
            .map(|p| {
                let mut f = [0.0; 3];
                for (k, fk) in f.iter_mut().enumerate() {
                    *fk = (p[0] * inv[(0, k)] + p[1] * inv[(1, k)]) + p[2] * inv[(2, k)];
                    if pbc[k] {
                        *fk -= fk.round_ties_even();
                    }
                }
                let mut o = [0.0; 3];
                for (k, ok) in o.iter_mut().enumerate() {
                    *ok = (f[0] * l[(0, k)] + f[1] * l[(1, k)]) + f[2] * l[(2, k)];
                }
                o
            })
            .collect(),
    )
}

/// Pairwise atom distances, `n x n` row-major.
pub fn pair_distances(r: &[Row]) -> Vec<f64> {
    let n = r.len();
    let mut out = vec![0.0; n * n];
    out.par_chunks_mut(n.max(1))
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                let (x, y, z) = (r[i][0] - r[j][0], r[i][1] - r[j][1], r[i][2] - r[j][2]);
                row[j] = (x * x + y * y + z * z).sqrt();
            }
        });
    out
}

/// Packed upper triangle of [`rmsd_matrix`] for rows `[i0, i1)`.
///
/// Row `i` contributes `rmsd(i, j)` for `j = i+1 .. m`, in that order.
/// Rows run in parallel; each entry is one serial pair, so the bits do not
/// depend on the thread count. The last row has no pairs.
pub fn rmsd_upper(
    frames: &[&[Row]],
    w: &[f64],
    align: bool,
    mode: Sum,
    i0: usize,
    i1: usize,
) -> Vec<f64> {
    let m = frames.len();
    let i0 = i0.min(m);
    let i1 = i1.min(m).max(i0);
    let mut rows: Vec<Vec<f64>> = vec![Vec::new(); i1 - i0];
    rows.par_iter_mut().enumerate().for_each(|(k, row)| {
        let i = i0 + k;
        row.extend(((i + 1)..m).map(|j| pair_rmsd(frames[i], frames[j], w, align, mode)));
    });
    rows.into_iter().flatten().collect()
}

/// RMSD between every pair of frames, `m x m` row-major and symmetric.
/// With `align`, each pair is Kabsch-aligned first (frame `j` onto `i`).
/// Entry `(i, j)` for `i < j` is computed once and mirrored.
pub fn rmsd_matrix(frames: &[&[Row]], w: &[f64], align: bool, mode: Sum) -> Vec<f64> {
    let m = frames.len();
    let packed = rmsd_upper(frames, w, align, mode, 0, m);
    let mut out = vec![0.0; m * m];
    let mut k = 0;
    for i in 0..m {
        for j in (i + 1)..m {
            let v = packed[k];
            k += 1;
            out[i * m + j] = v;
            out[j * m + i] = v;
        }
    }
    out
}

/// RMSD of one pair, aligned or not.
pub fn pair_rmsd(a: &[Row], b: &[Row], w: &[f64], align: bool, mode: Sum) -> f64 {
    if align {
        kabsch(a, b, w, mode).rmsd
    } else {
        rmsd(a, b, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(n: usize, seed: u64) -> Vec<Row> {
        let mut s = seed;
        let mut r = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
        };
        (0..n).map(|_| [r(), r(), r()]).collect()
    }

    #[test]
    fn kabsch_recovers_rotation() {
        let a = cloud(50, 1);
        let (c, s) = (0.3f64.cos(), 0.3f64.sin());
        let b: Vec<Row> = a
            .iter()
            .map(|p| {
                [
                    c * p[0] - s * p[1] + 1.0,
                    s * p[0] + c * p[1] - 2.0,
                    p[2] + 0.5,
                ]
            })
            .collect();
        let w = vec![1.0 / 50.0; 50];
        let al = kabsch(&a, &b, &w, Sum::Float);
        assert!(al.rmsd < 1e-12, "{}", al.rmsd);
        assert!(al.rotation.is_some());
    }

    #[test]
    fn thread_count_does_not_change_bits() {
        let r = cloud(64, 2);
        let frames: Vec<Vec<Row>> = (0..9).map(|k| cloud(64, 10 + k)).collect();
        let refs: Vec<&[Row]> = frames.iter().map(|f| f.as_slice()).collect();
        let w = vec![1.0 / 64.0; 64];
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    let m = rmsd_matrix(&refs, &w, true, Sum::Float);
                    let b: Vec<u64> = kabsch_batch(&r, &refs, &w, Sum::Float)
                        .iter()
                        .map(|a| a.rmsd.to_bits())
                        .collect();
                    (m.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b)
                })
        };
        assert_eq!(run(1), run(3));
    }

    #[cfg(feature = "exact")]
    #[test]
    fn exact_reductions_ignore_atom_order() {
        let a = cloud(97, 3);
        let b = cloud(97, 4);
        let m: Vec<f64> = (0..97).map(|i| 1.0 + i as f64 * 0.37).collect();
        let tot: f64 = exact::sum(&m);
        let w: Vec<f64> = m.iter().map(|x| x / tot).collect();
        let perm: Vec<usize> = (0..97).map(|i| (i * 31) % 97).collect();
        let p = |v: &[Row]| perm.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let pw: Vec<f64> = perm.iter().map(|&i| w[i]).collect();
        let pm: Vec<f64> = perm.iter().map(|&i| m[i]).collect();
        let e = Sum::Exact;
        assert_eq!(centroid(&a, &w, e), centroid(&p(&a), &pw, e));
        assert_eq!(rmsd(&a, &b, e).to_bits(), rmsd(&p(&a), &p(&b), e).to_bits());
        assert_eq!(
            covariance(&a, &b, &w, e),
            covariance(&p(&a), &p(&b), &pw, e)
        );
        assert_eq!(
            kinetic_energy(&m, &a, e).to_bits(),
            kinetic_energy(&pm, &p(&a), e).to_bits()
        );
        let k1 = kabsch(&a, &b, &w, e);
        let k2 = kabsch(&p(&a), &p(&b), &pw, e);
        assert_eq!(k1.rotation, k2.rotation);
        assert_eq!(k1.rmsd.to_bits(), k2.rmsd.to_bits());
    }

    #[cfg(feature = "exact")]
    #[test]
    fn exact_fast_path_matches_integer_accumulator() {
        let a = cloud(240, 11);
        let b = cloud(240, 12);
        let m: Vec<f64> = (0..240).map(|i| 1.0 + (i as f64) * 0.37).collect();
        let tot = exact::sum(&m);
        let w: Vec<f64> = m.iter().map(|x| x / tot).collect();
        let e = Sum::Exact;
        for k in 0..3 {
            let mut acc = exact::ExactAcc::new();
            a.iter().zip(&w).for_each(|(p, &q)| acc.add_prod(q, p[k]));
            assert_eq!(centroid(&a, &w, e)[k].to_bits(), acc.round().to_bits());
        }
        let mut acc = exact::ExactAcc::new();
        for i in 0..a.len() {
            for k in 0..3 {
                let d = a[i][k] - b[i][k];
                acc.add_prod(d, d);
            }
        }
        let want = (acc.round() / a.len() as f64).sqrt();
        assert_eq!(rmsd(&a, &b, e).to_bits(), want.to_bits());
        for j in 0..3 {
            for k in 0..3 {
                let mut acc = exact::ExactAcc::new();
                for i in 0..a.len() {
                    acc.add_prod3(w[i], a[i][j], b[i][k]);
                }
                assert_eq!(
                    covariance(&a, &b, &w, e)[(j, k)].to_bits(),
                    acc.round().to_bits()
                );
            }
        }
        let mut acc = exact::ExactAcc::new();
        for i in 0..m.len() {
            for k in 0..3 {
                acc.add_prod3(0.5 * m[i], a[i][k], a[i][k]);
            }
        }
        assert_eq!(kinetic_energy(&m, &a, e).to_bits(), acc.round().to_bits());
    }
}
