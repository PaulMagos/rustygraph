//! Node feature computation framework.
//!
//! This module provides an extensible system for computing features (properties)
//! for each node in a visibility graph. Features can be used for basis expansion
//! or data augmentation in machine learning pipelines.
//!
//! # Built-in Features
//!
//! The library includes several pre-defined features accessible through [`BuiltinFeature`]:
//!
//! - **Temporal derivatives**: DeltaForward, DeltaBackward, DeltaSymmetric
//! - **Local statistics**: LocalMean, LocalVariance
//! - **Geometric properties**: LocalSlope, Acceleration
//! - **Extrema detection**: IsLocalMax, IsLocalMin
//! - **Normalized values**: ZScore
//!
//! # Custom Features
//!
//! You can implement custom features in two ways:
//!
//! 1. **Simple functions**: Pass a closure to [`FeatureSet::add_function`]
//! 2. **Full trait implementation**: Implement the [`Feature`] trait for complex logic
//!
//! # Examples
//!
//! ## Using built-in features
//!
//! ```rust
//! use rustygraph::{FeatureSet, BuiltinFeature};
//!
//! let features = FeatureSet::new()
//!     .add_builtin(BuiltinFeature::DeltaForward)
//!     .add_builtin(BuiltinFeature::LocalSlope)
//!     .add_builtin(BuiltinFeature::LocalMean);
//! ```
//!
//! ## Custom function
//!
//! ```rust
//! use rustygraph::FeatureSet;
//!
//! let features = FeatureSet::new()
//!     .add_function("squared", |series, idx| {
//!         series[idx].map(|v| v * v)
//!     });
//! ```

pub mod missing_data;
pub mod builtin;

use missing_data::MissingDataHandler;

/// Trait for computing node features from time series data.
///
/// Implement this trait to create custom feature computations that can be
/// integrated into the visibility graph construction pipeline.
///
/// # Examples
///
/// ```rust
/// use rustygraph::features::{Feature, MissingDataHandler};
///
/// struct RangeFeature {
///     window: usize,
/// }
///
/// impl Feature<f64> for RangeFeature {
///     fn compute(
///         &self,
///         series: &[Option<f64>],
///         index: usize,
///         missing_handler: &dyn MissingDataHandler<f64>,
///     ) -> Option<f64> {
///         let start = index.saturating_sub(self.window / 2);
///         let end = (index + self.window / 2).min(series.len());
///
///         let valid: Vec<f64> = series[start..end]
///             .iter()
///             .filter_map(|&v| v)
///             .collect();
///
///         if valid.is_empty() {
///             return None;
///         }
///
///         let min = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
///         let max = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
///         Some(max - min)
///     }
///
///     fn name(&self) -> &str {
///         "range"
///     }
///
///     fn requires_neighbors(&self) -> bool {
///         true
///     }
///
///     fn window_size(&self) -> Option<usize> {
///         Some(self.window)
///     }
/// }
/// ```
pub trait Feature<T>: Send + Sync {
    /// Computes the feature value for a node.
    ///
    /// # Arguments
    ///
    /// - `series`: The time series data (with possible missing values)
    /// - `index`: Index of the node to compute feature for
    /// - `missing_handler`: Strategy for handling missing data
    ///
    /// # Returns
    ///
    /// Computed feature value, or `None` if computation is not possible
    fn compute(
        &self,
        series: &[Option<T>],
        index: usize,
        missing_handler: &dyn MissingDataHandler<T>,
    ) -> Option<T>;

    /// Returns the feature name for identification.
    ///
    /// This name is used as a key in the feature map returned by the graph.
    fn name(&self) -> &str;

    /// Indicates if the feature requires neighboring values.
    ///
    /// Return `true` if the feature needs access to points before/after the current index.
    fn requires_neighbors(&self) -> bool {
        false
    }

    /// Returns the local window size needed (if any).
    ///
    /// For features that compute statistics over a local window, return `Some(size)`.
    fn window_size(&self) -> Option<usize> {
        None
    }
}

/// Collection of features to compute for each node.
///
/// Use the builder pattern to construct a set of features that will be
/// computed for each node in the visibility graph.
///
/// # Examples
///
/// ```rust
/// use rustygraph::{FeatureSet, BuiltinFeature, MissingDataStrategy};
///
/// let features = FeatureSet::new()
///     .add_builtin(BuiltinFeature::DeltaForward)
///     .add_builtin(BuiltinFeature::LocalSlope)
///     .add_function("log", |series, idx| {
///         series[idx].map(|v| v.ln())
///     })
///     .with_missing_data_strategy(MissingDataStrategy::LinearInterpolation);
/// ```
pub struct FeatureSet<T> {
    features: Vec<Box<dyn Feature<T>>>,
    missing_strategy: missing_data::MissingDataStrategy,
}

impl<T> FeatureSet<T> {
    /// Creates an empty feature set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::FeatureSet;
    ///
    /// let features = FeatureSet::<f64>::new();
    /// ```
    pub fn new() -> Self {
        FeatureSet {
            features: Vec::new(),
            missing_strategy: missing_data::MissingDataStrategy::LinearInterpolation,
        }
    }

    /// Adds a built-in feature to the set.
    ///
    /// # Arguments
    ///
    /// - `feature`: A predefined feature type from [`BuiltinFeature`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{FeatureSet, BuiltinFeature};
    ///
    /// let features = FeatureSet::new()
    ///     .add_builtin(BuiltinFeature::DeltaForward)
    ///     .add_builtin(BuiltinFeature::LocalSlope);
    /// ```
    pub fn add_builtin(self, _feature: BuiltinFeature) -> Self {
        // Implementation will add the appropriate feature
        todo!("Built-in feature implementation")
    }

    /// Adds a custom feature implementation.
    ///
    /// Use this method when you have implemented the [`Feature`] trait.
    ///
    /// # Arguments
    ///
    /// - `feature`: A boxed trait object implementing `Feature`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::FeatureSet;
    /// // Assuming MyCustomFeature implements Feature<f64>
    /// // let features = FeatureSet::new()
    /// //     .add_custom(Box::new(MyCustomFeature));
    /// ```
    pub fn add_custom(mut self, feature: Box<dyn Feature<T>>) -> Self {
        self.features.push(feature);
        self
    }

    /// Adds a feature from a closure.
    ///
    /// This is the simplest way to add a custom feature without implementing
    /// the full [`Feature`] trait.
    ///
    /// # Arguments
    ///
    /// - `name`: Feature identifier (used as key in output)
    /// - `f`: Closure taking (series, index) and returning feature value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::FeatureSet;
    ///
    /// let features = FeatureSet::new()
    ///     .add_function("squared", |series, idx| {
    ///         series[idx].map(|v| v * v)
    ///     })
    ///     .add_function("reciprocal", |series, idx| {
    ///         series[idx].map(|v| 1.0 / v)
    ///     });
    /// ```
    pub fn add_function<F>(self, _name: &str, _f: F) -> Self
    where
        F: Fn(&[Option<T>], usize) -> Option<T> + Send + Sync + 'static,
    {
        // Implementation will wrap the function in a Feature implementation
        todo!("Function wrapper implementation")
    }

    /// Sets the global missing data handling strategy.
    ///
    /// This strategy will be used by all features when encountering missing values.
    ///
    /// # Arguments
    ///
    /// - `strategy`: The imputation strategy to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{FeatureSet, MissingDataStrategy};
    ///
    /// let features = FeatureSet::<f64>::new()
    ///     .with_missing_data_strategy(MissingDataStrategy::ForwardFill);
    /// ```
    pub fn with_missing_data_strategy(mut self, strategy: missing_data::MissingDataStrategy) -> Self {
        self.missing_strategy = strategy;
        self
    }

    /// Returns the number of features in the set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::{FeatureSet, BuiltinFeature};
    ///
    /// let features = FeatureSet::new()
    ///     .add_builtin(BuiltinFeature::DeltaForward)
    ///     .add_builtin(BuiltinFeature::LocalSlope);
    ///
    /// assert_eq!(features.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Returns true if no features have been added.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

impl<T> Default for FeatureSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-defined feature types for common computations.
///
/// These features cover typical time series analysis needs, including
/// derivatives, local statistics, and extrema detection.
///
/// # Examples
///
/// ```rust
/// use rustygraph::{FeatureSet, BuiltinFeature};
///
/// let features : FeatureSet<BuiltinFeature> = FeatureSet::new()
///     .add_builtin(BuiltinFeature::DeltaForward)
///     .add_builtin(BuiltinFeature::LocalMean)
///     .add_builtin(BuiltinFeature::IsLocalMax);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFeature {
    /// Forward difference: `y[i+1] - y[i]`
    ///
    /// Measures the change to the next time step. Returns `None` for the last point.
    DeltaForward,

    /// Backward difference: `y[i] - y[i-1]`
    ///
    /// Measures the change from the previous time step. Returns `None` for the first point.
    DeltaBackward,

    /// Symmetric difference: `(y[i+1] - y[i-1]) / 2`
    ///
    /// Central difference approximation. Returns `None` for first and last points.
    DeltaSymmetric,

    /// Local slope: `(y[i+1] - y[i-1]) / (t[i+1] - t[i-1])`
    ///
    /// Slope of the line through neighboring points. Accounts for non-uniform time spacing.
    LocalSlope,

    /// Second derivative approximation: `y[i+1] - 2*y[i] + y[i-1]`
    ///
    /// Measures acceleration or curvature. Returns `None` for first and last points.
    Acceleration,

    /// Mean over local window
    ///
    /// Average value in a neighborhood around the point.
    LocalMean,

    /// Variance over local window
    ///
    /// Statistical variance in a neighborhood around the point.
    LocalVariance,

    /// True (1.0) if local maximum, false (0.0) otherwise
    ///
    /// Detects peaks: `y[i] > y[i-1] && y[i] > y[i+1]`
    IsLocalMax,

    /// True (1.0) if local minimum, false (0.0) otherwise
    ///
    /// Detects valleys: `y[i] < y[i-1] && y[i] < y[i+1]`
    IsLocalMin,

    /// Z-score: `(y[i] - mean) / std`
    ///
    /// Normalized value relative to global mean and standard deviation.
    ZScore,
}

