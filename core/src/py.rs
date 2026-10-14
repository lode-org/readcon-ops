//! PyO3 bindings. Arrays must be C-contiguous float64; they are read in
//! place. `exact=True` raises unless the module was built with `exact`.

use numpy::{
    IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{Row, Sum};

fn rows<'a>(a: &'a PyReadonlyArray2<'_, f64>) -> PyResult<&'a [Row]> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
    if a.shape()[1] != 3 {
        return Err(PyValueError::new_err("expected shape (n, 3)"));
    }
    // SAFETY: a contiguous (n, 3) float64 block has the layout of [[f64; 3]; n].
    Ok(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const Row, s.len() / 3) })
}

fn frames<'a>(a: &'a PyReadonlyArray3<'_, f64>) -> PyResult<Vec<&'a [Row]>> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
    let sh = a.shape();
    if sh[2] != 3 {
        return Err(PyValueError::new_err("expected shape (m, n, 3)"));
    }
    // SAFETY: as in `rows`, one (n, 3) block per frame.
    let all = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const Row, s.len() / 3) };
    Ok(all.chunks(sh[1].max(1)).take(sh[0]).collect())
}

fn mode(exact: bool) -> PyResult<Sum> {
    if !exact {
        return Ok(Sum::Float);
    }
    #[cfg(feature = "exact")]
    return Ok(Sum::Exact);
    #[cfg(not(feature = "exact"))]
    Err(PyValueError::new_err(
        "readcon_ops_core was built without the exact feature",
    ))
}

fn check_len(n: usize, k: usize, what: &str) -> PyResult<()> {
    if n != k {
        return Err(PyValueError::new_err(format!(
            "{what} has {k} entries, expected {n}"
        )));
    }
    Ok(())
}

/// Align each of `frames` (m, n, 3) onto `reference` (n, 3).
/// Returns (rmsd (m,), rotation (m, 3, 3), aligned (m,), moved (m, n, 3)).
/// Where `aligned` is False the rotation is NaN and the caller should use
/// the Python path (rank-0 or collinear covariance).
#[pyfunction]
#[pyo3(signature = (reference, frames, weights, exact=false))]
fn kabsch_batch<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray2<'py, f64>,
    frames: PyReadonlyArray3<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    exact: bool,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    let r = rows(&reference)?;
    let f = self::frames(&frames)?;
    let w = weights.as_slice()?;
    check_len(r.len(), w.len(), "weights")?;
    for x in &f {
        check_len(r.len(), x.len(), "frame")?;
    }
    let out = crate::kabsch_batch(r, &f, w, mode(exact)?);
    let (m, n) = (f.len(), r.len());
    let rm: Vec<f64> = out.iter().map(|a| a.rmsd).collect();
    let ok: Vec<bool> = out.iter().map(|a| a.rotation.is_some()).collect();
    let rot: Vec<f64> = out
        .iter()
        .flat_map(|a| {
            a.rotation
                .map(|x| x.concat())
                .unwrap_or([f64::NAN; 9].to_vec())
        })
        .collect();
    let mv: Vec<f64> = out
        .iter()
        .flat_map(|a| a.moved.iter().flatten().copied().collect::<Vec<_>>())
        .collect();
    (
        rm.into_pyarray(py),
        PyArray1::from_vec(py, rot).reshape([m, 3, 3])?,
        ok.into_pyarray(py),
        PyArray1::from_vec(py, mv).reshape([m, n, 3])?,
    )
        .into_pyobject(py)
}

/// Symmetric (m, m) RMSD matrix of `frames`, aligned pairwise if `align`.
#[pyfunction]
#[pyo3(signature = (frames, weights, align=true, exact=false))]
fn rmsd_matrix<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray3<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    align: bool,
    exact: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let f = self::frames(&frames)?;
    let w = weights.as_slice()?;
    let m = f.len();
    let v = crate::rmsd_matrix(&f, w, align, mode(exact)?);
    Ok(PyArray1::from_vec(py, v).reshape([m, m])?.into_any())
}

/// RMSD of the pairs `(i, j)` listed in `pairs` (k, 2), in that order.
#[pyfunction]
#[pyo3(signature = (frames, weights, pairs, align=true, exact=false))]
fn pair_rmsds<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray3<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    pairs: PyReadonlyArray2<'py, i64>,
    align: bool,
    exact: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use rayon::prelude::*;
    let f = self::frames(&frames)?;
    let w = weights.as_slice()?;
    let p = pairs.as_slice()?;
    let md = mode(exact)?;
    let v: Vec<f64> = p
        .par_chunks(2)
        .map(|ij| crate::pair_rmsd(f[ij[0] as usize], f[ij[1] as usize], w, align, md))
        .collect();
    Ok(v.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (a, b, exact=false))]
fn rmsd(a: PyReadonlyArray2<'_, f64>, b: PyReadonlyArray2<'_, f64>, exact: bool) -> PyResult<f64> {
    let (x, y) = (rows(&a)?, rows(&b)?);
    check_len(x.len(), y.len(), "b")?;
    Ok(crate::rmsd(x, y, mode(exact)?))
}

#[pyfunction]
#[pyo3(signature = (r, weights, exact=false))]
fn centroid(
    r: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    exact: bool,
) -> PyResult<(f64, f64, f64)> {
    let x = rows(&r)?;
    let w = weights.as_slice()?;
    check_len(x.len(), w.len(), "weights")?;
    let c = crate::centroid(x, w, mode(exact)?);
    Ok((c[0], c[1], c[2]))
}

#[pyfunction]
#[pyo3(signature = (a, b, weights, exact=false))]
fn covariance<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    exact: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (x, y, w) = (rows(&a)?, rows(&b)?, weights.as_slice()?);
    check_len(x.len(), y.len(), "b")?;
    check_len(x.len(), w.len(), "weights")?;
    let h = crate::covariance(x, y, w, mode(exact)?);
    let v: Vec<f64> = (0..3)
        .flat_map(|j| (0..3).map(move |k| h[(j, k)]))
        .collect();
    Ok(PyArray1::from_vec(py, v).reshape([3, 3])?.into_any())
}

#[pyfunction]
#[pyo3(signature = (masses, velocities, exact=false))]
fn kinetic_energy(
    masses: PyReadonlyArray1<'_, f64>,
    velocities: PyReadonlyArray2<'_, f64>,
    exact: bool,
) -> PyResult<f64> {
    let (m, v) = (masses.as_slice()?, rows(&velocities)?);
    check_len(v.len(), m.len(), "masses")?;
    Ok(crate::kinetic_energy(m, v, mode(exact)?))
}

#[pyfunction]
#[pyo3(signature = (x, exact=false))]
fn mean(x: PyReadonlyArray1<'_, f64>, exact: bool) -> PyResult<f64> {
    Ok(crate::mean(x.as_slice()?, mode(exact)?))
}

/// Wrap displacements (n, 3) along periodic axes of `cell` (3, 3 rows).
#[pyfunction]
fn min_image<'py>(
    py: Python<'py>,
    dr: PyReadonlyArray2<'py, f64>,
    cell: PyReadonlyArray2<'py, f64>,
    pbc: (bool, bool, bool),
) -> PyResult<Bound<'py, PyAny>> {
    let d = rows(&dr)?;
    let c = rows(&cell)?;
    if c.len() != 3 {
        return Err(PyValueError::new_err("cell must be (3, 3)"));
    }
    let out = crate::min_image(d, &[c[0], c[1], c[2]], [pbc.0, pbc.1, pbc.2])
        .ok_or_else(|| PyValueError::new_err("cell is singular"))?;
    let n = out.len();
    Ok(PyArray1::from_vec(py, out.concat())
        .reshape([n, 3])?
        .into_any())
}

#[pyfunction]
fn pair_distances<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let x = rows(&r)?;
    let n = x.len();
    Ok(PyArray1::from_vec(py, crate::pair_distances(x))
        .reshape([n, n])?
        .into_any())
}

/// Exact partial sum of `x` (and of `x * y` when `y` is given) as bytes.
/// Merge partials from any split, in any order, with `exact_round`.
#[cfg(feature = "exact")]
#[pyfunction]
#[pyo3(signature = (x, y=None))]
fn exact_partial<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut a = crate::exact::ExactAcc::new();
    let xs = x.as_slice()?;
    match y {
        None => xs.iter().for_each(|&v| a.add(v)),
        Some(y) => {
            let ys = y.as_slice()?;
            check_len(xs.len(), ys.len(), "y")?;
            xs.iter().zip(ys).for_each(|(&p, &q)| a.add_prod(p, q));
        }
    }
    Ok(PyBytes::new(py, &a.to_bytes()))
}

/// Exact partial sum of `x * y * z` as bytes (for example `0.5 m v v`).
#[cfg(feature = "exact")]
#[pyfunction]
fn exact_partial3<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (xs, ys, zs) = (x.as_slice()?, y.as_slice()?, z.as_slice()?);
    check_len(xs.len(), ys.len(), "y")?;
    check_len(xs.len(), zs.len(), "z")?;
    let mut a = crate::exact::ExactAcc::new();
    for i in 0..xs.len() {
        a.add_prod3(xs[i], ys[i], zs[i]);
    }
    Ok(PyBytes::new(py, &a.to_bytes()))
}

/// Merge exact partials and round once.
#[cfg(feature = "exact")]
#[pyfunction]
fn exact_round(parts: Vec<Vec<u8>>) -> PyResult<f64> {
    let mut a = crate::exact::ExactAcc::new();
    for p in parts {
        let b = crate::exact::ExactAcc::from_bytes(&p)
            .ok_or_else(|| PyValueError::new_err("bad exact partial"))?;
        a.merge(&b);
    }
    Ok(a.round())
}

#[pymodule]
fn readcon_ops_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("HAS_EXACT", cfg!(feature = "exact"))?;
    m.add_function(wrap_pyfunction!(kabsch_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rmsd_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(pair_rmsds, m)?)?;
    m.add_function(wrap_pyfunction!(rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(centroid, m)?)?;
    m.add_function(wrap_pyfunction!(covariance, m)?)?;
    m.add_function(wrap_pyfunction!(kinetic_energy, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(min_image, m)?)?;
    m.add_function(wrap_pyfunction!(pair_distances, m)?)?;
    #[cfg(feature = "exact")]
    {
        m.add_function(wrap_pyfunction!(exact_partial, m)?)?;
        m.add_function(wrap_pyfunction!(exact_partial3, m)?)?;
        m.add_function(wrap_pyfunction!(exact_round, m)?)?;
    }
    Ok(())
}
