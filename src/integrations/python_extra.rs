//! Python bindings: streaming visibility graphs and sequential motif profiles.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::algorithms::fast::InputError;
use crate::core::algorithms::motifs::{self, MotifKind, MOTIFS};
use crate::core::algorithms::stream::{HorizontalStream, NaturalStream, StreamMode, VectorStream};
use crate::integrations::python_fast::to_f64;

fn value_err(e: InputError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

enum Inner {
    Natural(NaturalStream),
    Horizontal(HorizontalStream),
    Vector(VectorStream),
}

/// VisibilityStream(kind="natural", window=None, dim=None, mode="auto")
///
/// Online visibility graph: `push(value, t=None)` appends one sample and returns
/// the sources of the new edges (source, new_index) as an int64 array.
///
/// kind: "natural" | "horizontal" | "vector_natural" | "vector_horizontal".
/// window: only connect to the last `window` samples (sliding window graph).
/// dim: vector dimension (required for vector kinds).
/// mode (natural only): "auto" (scan short histories, hull-index long ones),
/// "scan" or "indexed".
#[pyclass(name = "VisibilityStream")]
pub struct PyVisibilityStream {
    inner: Inner,
    kind: String,
}

#[pymethods]
impl PyVisibilityStream {
    #[new]
    #[pyo3(signature = (kind="natural", window=None, dim=None, mode="auto"))]
    fn new(kind: &str, window: Option<usize>, dim: Option<usize>, mode: &str) -> PyResult<Self> {
        if window == Some(0) {
            return Err(PyValueError::new_err("window must be >= 1"));
        }
        let mode = match mode {
            "auto" => StreamMode::Auto,
            "scan" => StreamMode::Scan,
            "indexed" => StreamMode::Indexed,
            m => {
                return Err(PyValueError::new_err(format!(
                    "unknown mode '{m}' (auto, scan, indexed)"
                )))
            }
        };
        let need_dim =
            || dim.ok_or_else(|| PyValueError::new_err("dim is required for vector kinds"));
        let inner = match kind {
            "natural" => Inner::Natural(NaturalStream::new(window, mode)),
            "horizontal" => Inner::Horizontal(HorizontalStream::new(window)),
            "vector_natural" => {
                Inner::Vector(VectorStream::new(need_dim()?, window, false).map_err(value_err)?)
            }
            "vector_horizontal" => {
                Inner::Vector(VectorStream::new(need_dim()?, window, true).map_err(value_err)?)
            }
            k => return Err(PyValueError::new_err(format!("unknown kind '{k}'"))),
        };
        Ok(Self {
            inner,
            kind: kind.to_string(),
        })
    }

    /// Appends one sample (scalar, or a length-`dim` vector) and returns the
    /// sources of its new edges.
    #[pyo3(signature = (value, t=None))]
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        t: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let src = match &mut self.inner {
            Inner::Natural(s) => s.push(value.extract::<f64>()?, t),
            Inner::Horizontal(s) => {
                if t.is_some() {
                    return Err(PyValueError::new_err(
                        "horizontal visibility does not use t",
                    ));
                }
                s.push(value.extract::<f64>()?)
            }
            Inner::Vector(s) => {
                let v = to_f64(value)?;
                s.push(v.as_slice(), t)
            }
        }
        .map_err(value_err)?;
        Ok(PyArray1::from_vec(
            py,
            src.into_iter().map(|s| s as i64).collect(),
        ))
    }

    /// Pushes many samples; returns all new edges as an (E, 2) int64 array.
    #[pyo3(signature = (values, t=None))]
    fn extend<'py>(
        &mut self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
        t: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let t = t.map(to_f64).transpose()?.map(|v| v.as_slice().to_vec());
        let mut flat: Vec<i64> = Vec::new();
        let base = self.__len__();
        match &mut self.inner {
            Inner::Vector(s) => {
                let np = py.import("numpy")?;
                let arr = np.call_method1("ascontiguousarray", (values, "float64"))?;
                let m = arr
                    .extract::<PyReadonlyArray2<'py, f64>>()
                    .map_err(|_| PyValueError::new_err("expected (n, dim) array"))?;
                let (n, d) = (m.shape()[0], m.shape()[1]);
                let data = m.as_slice()?.to_vec();
                check_t_len(&t, n)?;
                for i in 0..n {
                    let src = s
                        .push(&data[i * d..(i + 1) * d], t.as_ref().map(|t| t[i]))
                        .map_err(value_err)?;
                    for a in src {
                        flat.extend([a as i64, (base + i) as i64]);
                    }
                }
            }
            other => {
                let v = to_f64(values)?.as_slice().to_vec();
                check_t_len(&t, v.len())?;
                for (i, &y) in v.iter().enumerate() {
                    let src = match other {
                        Inner::Natural(s) => s.push(y, t.as_ref().map(|t| t[i])),
                        Inner::Horizontal(s) => s.push(y),
                        Inner::Vector(_) => unreachable!(),
                    }
                    .map_err(value_err)?;
                    for a in src {
                        flat.extend([a as i64, (base + i) as i64]);
                    }
                }
            }
        }
        let e = flat.len() / 2;
        Ok(PyArray1::from_vec(py, flat).reshape([e, 2])?.into_any())
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            Inner::Natural(s) => s.len(),
            Inner::Horizontal(s) => s.len(),
            Inner::Vector(s) => s.len(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VisibilityStream(kind='{}', n={})",
            self.kind,
            self.__len__()
        )
    }
}

fn check_t_len(t: &Option<Vec<f64>>, n: usize) -> PyResult<()> {
    match t {
        Some(t) if t.len() != n => Err(PyValueError::new_err("t must have one entry per sample")),
        _ => Ok(()),
    }
}

fn parse_motif_kind(kind: &str) -> PyResult<(MotifKind, bool)> {
    Ok(match kind {
        "natural" => (MotifKind::Natural, false),
        "horizontal" => (MotifKind::Horizontal, false),
        "vector_natural" => (MotifKind::Natural, true),
        "vector_horizontal" => (MotifKind::Horizontal, true),
        k => return Err(PyValueError::new_err(format!("unknown kind '{k}'"))),
    })
}

/// visibility_motifs(x, kind="natural") -> ndarray[uint64] of shape (8,) or (B, 8)
///
/// Sequential size-4 visibility-graph motif counts (Iacovacci & Lacasa 2016).
/// Code = [0,2] + 2·[1,3] + 4·[0,3] for the optional edges among 4
/// consecutive samples. Accepts one series (n,) / (n, d) for vector kinds, or a
/// batch (B, n) / (B, n, d), processed in parallel. O(n), no graph is built.
#[pyfunction]
#[pyo3(signature = (x, kind="natural"))]
pub fn visibility_motifs<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    kind: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (mk, vector) = parse_motif_kind(kind)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (x, "float64"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let data: Vec<f64> = arr
        .call_method0("ravel")?
        .extract::<numpy::PyReadonlyArray1<'py, f64>>()?
        .as_slice()?
        .to_vec();
    let (batch, n, d) = match (vector, shape.as_slice()) {
        (false, [n]) => (None, *n, 1),
        (false, [b, n]) => (Some(*b), *n, 1),
        (true, [n, d]) => (None, *n, *d),
        (true, [b, n, d]) => (Some(*b), *n, *d),
        _ => return Err(PyValueError::new_err(
            "expected (n,) or (B, n) for univariate kinds, (n, d) or (B, n, d) for vector kinds",
        )),
    };
    let counts: Vec<[u64; MOTIFS]> = py
        .allow_threads(|| -> Result<Vec<[u64; MOTIFS]>, InputError> {
            let per = n * d;
            let rows: Vec<&[f64]> = if per == 0 {
                vec![&data[..0]; batch.unwrap_or(1)]
            } else {
                data.chunks(per).collect()
            };
            if vector {
                motifs::vector_motif_counts_batch(&rows, d, mk)
            } else {
                motifs::motif_counts_batch(&rows, mk)
            }
        })
        .map_err(value_err)?;
    let rows: Vec<Vec<u64>> = counts.iter().map(|c| c.to_vec()).collect();
    let out = PyArray2::from_vec2(py, &rows)?;
    Ok(match batch {
        Some(_) => out.into_any(),
        None => out.get_item(0)?,
    })
}

/// Registers the streaming and motif endpoints.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVisibilityStream>()?;
    m.add_function(wrap_pyfunction!(visibility_motifs, m)?)?;
    Ok(())
}
