//! Zero-copy, GIL-free Python endpoints returning NumPy edge arrays.
//!
//! These bypass the `VisibilityGraph` object (hash map + adjacency lists) and
//! are the fastest way to get edges into NumPy / PyTorch / PyG:
//!
//! ```python
//! import rustygraph as rg
//! e = rg.natural_visibility_edges(y)            # (E, 2) int64, rows (a, b) with a < b
//! e, off = rg.natural_visibility_batch(Y)       # Y: (B, n); edges of row k = e[off[k]:off[k+1]]
//! ```

use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::algorithms::fast::{self, Edge, Explicit, InputError, Uniform};
use crate::core::algorithms::vector::{self, Rows, VectorWeight};

fn value_err(e: InputError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Borrow a 1-D float64 array without copying, or convert anything array-like.
pub(crate) enum F64Vec<'py> {
    Borrowed(PyReadonlyArray1<'py, f64>),
    Owned(Vec<f64>),
}

impl F64Vec<'_> {
    pub(crate) fn as_slice(&self) -> &[f64] {
        match self {
            F64Vec::Borrowed(a) => a.as_slice().expect("checked contiguous"),
            F64Vec::Owned(v) => v,
        }
    }
}

pub(crate) fn to_f64<'py>(obj: &Bound<'py, PyAny>) -> PyResult<F64Vec<'py>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'py, f64>>() {
        if arr.is_contiguous() {
            return Ok(F64Vec::Borrowed(arr));
        }
    }
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "float64"))?;
    let arr = arr
        .extract::<PyReadonlyArray1<'py, f64>>()
        .map_err(|_| PyValueError::new_err("expected a 1-D array-like of numbers"))?;
    Ok(F64Vec::Owned(arr.as_slice()?.to_vec()))
}

fn edges_to_numpy<'py>(py: Python<'py>, edges: &[Edge]) -> PyResult<Bound<'py, PyAny>> {
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for &(a, b) in edges {
        flat.push(a as i64);
        flat.push(b as i64);
    }
    Ok(PyArray1::from_vec(py, flat)
        .reshape([edges.len(), 2])?
        .into_any())
}

/// natural_visibility_edges(y, x=None) -> ndarray[int64] of shape (E, 2)
///
/// Natural visibility graph edges `(a, b)`, `a < b`. `x` are optional strictly
/// increasing sample positions (defaults to 0, 1, 2, …). Large series are
/// processed in parallel; the GIL is released while computing.
///
/// `algorithm`: "auto" (default) measures the exact scan cost and picks the
/// cheaper engine; "scan" forces divide & conquer; "hull" forces the
/// output-sensitive O((n + E) log² n) engine.
#[pyfunction]
#[pyo3(signature = (y, x=None, algorithm="auto"))]
pub fn natural_visibility_edges<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    x: Option<&Bound<'py, PyAny>>,
    algorithm: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let algo: fast::Algorithm = algorithm.parse().map_err(PyValueError::new_err)?;
    let y = to_f64(y)?;
    let x = x.map(to_f64).transpose()?;
    let (ys, xs) = (y.as_slice(), x.as_ref().map(|v| v.as_slice()));
    let edges = py
        .allow_threads(|| fast::natural_with(ys, xs, algo))
        .map_err(value_err)?;
    edges_to_numpy(py, &edges)
}

/// natural_visibility_plan(y) -> dict(algorithm, scan_work, parallel, n)
///
/// Which engine `natural_visibility_edges(y)` would choose, and why (O(n)).
#[pyfunction]
pub fn natural_visibility_plan<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let y = to_f64(y)?;
    let ys = y.as_slice();
    fast::validate(ys, None).map_err(value_err)?;
    let plan = py.allow_threads(|| fast::plan_natural(ys, fast::Algorithm::Auto));
    let d = pyo3::types::PyDict::new(py);
    d.set_item("algorithm", format!("{:?}", plan.algorithm).to_lowercase())?;
    d.set_item("scan_work", plan.scan_work)?;
    d.set_item("parallel", plan.parallel)?;
    d.set_item("n", ys.len())?;
    Ok(d.into_any())
}

/// horizontal_visibility_edges(y) -> ndarray[int64] of shape (E, 2)
#[pyfunction]
pub fn horizontal_visibility_edges<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let y = to_f64(y)?;
    let ys = y.as_slice();
    let edges = py
        .allow_threads(|| fast::horizontal(ys))
        .map_err(value_err)?;
    edges_to_numpy(py, &edges)
}

type BatchFn = fn(&[&[f64]]) -> Result<Vec<Vec<Edge>>, InputError>;

fn batch<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
    kernel: BatchFn,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    // Fast path: contiguous (B, n) float64 matrix, zero copy.
    let owned: Vec<Vec<f64>>;
    let mat;
    let slices: Vec<&[f64]> = if let Ok(m) = rows.extract::<PyReadonlyArray2<'py, f64>>() {
        if m.is_c_contiguous() {
            mat = m;
            let n = mat.shape()[1];
            let data = mat.as_slice()?;
            if n == 0 {
                vec![&data[..0]; mat.shape()[0]]
            } else {
                data.chunks(n).collect()
            }
        } else {
            owned = m
                .as_array()
                .rows()
                .into_iter()
                .map(|r| r.to_vec())
                .collect();
            owned.iter().map(Vec::as_slice).collect()
        }
    } else {
        // Ragged: any iterable of 1-D array-likes.
        let mut v = Vec::new();
        for item in rows.try_iter()? {
            v.push(to_f64(&item?)?.as_slice().to_vec());
        }
        owned = v;
        owned.iter().map(Vec::as_slice).collect()
    };

    let per_row = py.allow_threads(|| kernel(&slices)).map_err(value_err)?;

    let total: usize = per_row.iter().map(Vec::len).sum();
    let mut flat = Vec::with_capacity(total * 2);
    let mut offsets = Vec::with_capacity(per_row.len() + 1);
    offsets.push(0i64);
    for e in &per_row {
        for &(a, b) in e {
            flat.push(a as i64);
            flat.push(b as i64);
        }
        offsets.push((flat.len() / 2) as i64);
    }
    let edges = PyArray1::from_vec(py, flat).reshape([total, 2])?.into_any();
    let offsets = PyArray1::from_vec(py, offsets).into_any();
    Ok((edges, offsets))
}

/// natural_visibility_batch(Y) -> (edges (E, 2) int64, offsets (B+1,) int64)
///
/// Natural visibility graphs of many independent series in one call, in
/// parallel. `Y` is a (B, n) array or an iterable of 1-D arrays (ragged
/// lengths allowed). Edges of series `k` are `edges[offsets[k]:offsets[k+1]]`
/// with node indices local to that series.
#[pyfunction]
pub fn natural_visibility_batch<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    batch(py, rows, fast::natural_batch)
}

/// horizontal_visibility_batch(Y) -> (edges (E, 2) int64, offsets (B+1,) int64)
#[pyfunction]
pub fn horizontal_visibility_batch<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    batch(py, rows, fast::horizontal_batch)
}

/// Contiguous row-major (n, d) float64 copy of a 1-D or 2-D array-like.
fn to_matrix<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(Vec<f64>, usize)> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "float64"))?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    let arr = match ndim {
        1 => arr.call_method1("reshape", (-1i64, 1i64))?,
        2 => arr,
        _ => {
            return Err(PyValueError::new_err(
                "expected a 1-D or 2-D array of shape (time, features)",
            ))
        }
    };
    let m = arr.extract::<PyReadonlyArray2<'py, f64>>()?;
    let d = m.shape()[1];
    Ok((m.as_slice()?.to_vec(), d))
}

fn parse_weight(weight: Option<&str>) -> PyResult<Option<VectorWeight>> {
    weight
        .map(|w| w.parse::<VectorWeight>().map_err(PyValueError::new_err))
        .transpose()
}

fn vector_common<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    t: Option<&Bound<'py, PyAny>>,
    weight: Option<&str>,
    natural: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (data, d) = to_matrix(x)?;
    let t = t.map(to_f64).transpose()?.map(|v| v.as_slice().to_vec());
    let weight = parse_weight(weight)?;
    let (edges, weights) = py
        .allow_threads(|| -> Result<(Vec<Edge>, Option<Vec<f64>>), InputError> {
            let rows = Rows::new(&data, d)?;
            if let Some(t) = &t {
                fast::validate(t, None).map_err(|_| InputError::NotIncreasing(0))?;
                if t.len() != rows.len() {
                    return Err(InputError::LengthMismatch);
                }
                if let Some(i) = t.windows(2).position(|w| w[1] <= w[0]) {
                    return Err(InputError::NotIncreasing(i + 1));
                }
            }
            let edges = match (&t, natural) {
                (Some(t), true) => vector::natural_vector_edges(&rows, &Explicit(t), true),
                (None, true) => vector::natural_vector_edges(&rows, &Uniform, true),
                (_, false) => vector::horizontal_vector_edges(&rows, true),
            };
            let weights = weight.map(|w| match &t {
                Some(t) => vector::vector_weights(&rows, &Explicit(t), &edges, w),
                None => vector::vector_weights(&rows, &Uniform, &edges, w),
            });
            Ok((edges, weights))
        })
        .map_err(value_err)?;
    let e = edges_to_numpy(py, &edges)?;
    match weights {
        None => Ok(e),
        Some(w) => Ok((e, PyArray1::from_vec(py, w)).into_pyobject(py)?.into_any()),
    }
}

/// natural_vector_visibility_edges(X, t=None, weight=None)
///
/// Natural vector visibility graph (Ren & Jin 2019; `vector-vis-graph`
/// semantics) of a multivariate series `X` of shape (time, features).
/// Returns an (E, 2) int64 array, or `(edges, weights)` when `weight` is one
/// of "cosine_similarity", "time_diff_cosine_similarity",
/// "euclidean_distance", "time_diff_euclidean_distance",
/// "normalized_euclidean_distance", "time_diff_normalized_euclidean_distance".
#[pyfunction]
#[pyo3(signature = (x, t=None, weight=None))]
pub fn natural_vector_visibility_edges<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    t: Option<&Bound<'py, PyAny>>,
    weight: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    vector_common(py, x, t, weight, true)
}

/// horizontal_vector_visibility_edges(X, weight=None)
#[pyfunction]
#[pyo3(signature = (x, weight=None))]
pub fn horizontal_vector_visibility_edges<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    weight: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    vector_common(py, x, None, weight, false)
}

type VectorBatchFn = fn(&[Rows<'_>]) -> Vec<Vec<Edge>>;

fn vector_batch<'py>(
    py: Python<'py>,
    windows: &Bound<'py, PyAny>,
    kernel: VectorBatchFn,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (windows, "float64"))?;
    let arr = arr.extract::<PyReadonlyArray3<'py, f64>>().map_err(|_| {
        PyValueError::new_err("expected a 3-D array of shape (windows, time, features)")
    })?;
    let (b, w, d) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let data = arr.as_slice()?.to_vec();
    let per = py
        .allow_threads(|| -> Result<Vec<Vec<Edge>>, InputError> {
            if w * d == 0 {
                return Ok(vec![Vec::new(); b]);
            }
            let rows: Vec<Rows<'_>> = data
                .chunks(w * d)
                .map(|c| Rows::new(c, d))
                .collect::<Result<_, _>>()?;
            Ok(kernel(&rows))
        })
        .map_err(value_err)?;
    let total: usize = per.iter().map(Vec::len).sum();
    let mut flat = Vec::with_capacity(total * 2);
    let mut offsets = Vec::with_capacity(b + 1);
    offsets.push(0i64);
    for e in &per {
        for &(a, c) in e {
            flat.push(a as i64);
            flat.push(c as i64);
        }
        offsets.push((flat.len() / 2) as i64);
    }
    Ok((
        PyArray1::from_vec(py, flat).reshape([total, 2])?.into_any(),
        PyArray1::from_vec(py, offsets).into_any(),
    ))
}

/// natural_vector_visibility_batch(W) -> (edges (E, 2) int64, offsets (B+1,) int64)
///
/// Natural VVGs of many windows `W` of shape (windows, time, features), in
/// parallel. Edges of window k are `edges[offsets[k]:offsets[k+1]]`.
#[pyfunction]
pub fn natural_vector_visibility_batch<'py>(
    py: Python<'py>,
    windows: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    vector_batch(py, windows, vector::natural_vector_batch)
}

/// horizontal_vector_visibility_batch(W) -> (edges, offsets)
#[pyfunction]
pub fn horizontal_vector_visibility_batch<'py>(
    py: Python<'py>,
    windows: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    vector_batch(py, windows, vector::horizontal_vector_batch)
}

/// Registers the fast endpoints on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(natural_visibility_edges, m)?)?;
    m.add_function(wrap_pyfunction!(natural_visibility_plan, m)?)?;
    m.add_function(wrap_pyfunction!(horizontal_visibility_edges, m)?)?;
    m.add_function(wrap_pyfunction!(natural_visibility_batch, m)?)?;
    m.add_function(wrap_pyfunction!(horizontal_visibility_batch, m)?)?;
    m.add_function(wrap_pyfunction!(natural_vector_visibility_edges, m)?)?;
    m.add_function(wrap_pyfunction!(horizontal_vector_visibility_edges, m)?)?;
    m.add_function(wrap_pyfunction!(natural_vector_visibility_batch, m)?)?;
    m.add_function(wrap_pyfunction!(horizontal_vector_visibility_batch, m)?)?;
    Ok(())
}
