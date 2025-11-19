//! Node feature computation framework.
//!
//! This module provides an extensible system for computing features (properties)
//! for each node in a visibility graph.
pub mod missing_data;
pub mod builtin;
pub use self::missing_data::MissingDataHandler;
pub use self::builtin::BuiltinFeature;
use std::collections::HashMap;

/// Type alias for feature computation functions.
pub type FeatureFn<T> = fn(&[Option<T>], usize) -> Option<T>;

/// A computed feature for a node.
pub trait Feature<T>: Send + Sync {
    /// Computes the feature value for a node at the given index.
    fn compute(&self, series: &[Option<T>], index: usize, handler: &dyn MissingDataHandler<T>) -> Option<T>;
    /// Returns the name of the feature.
    fn name(&self) -> &str;
    /// Returns whether this feature requires neighboring values.
    fn requires_neighbors(&self) -> bool {
        false
    }
    /// Returns the window size needed for this feature, if any.
    fn window_size(&self) -> Option<usize> {
        None
    }
}
/// A collection of features to compute for nodes.
pub struct FeatureSet<T> {
    pub(crate) features: Vec<Box<dyn Feature<T> + Send + Sync>>,
    pub(crate) functions: HashMap<String, FeatureFn<T>>,
    pub(crate) missing_strategy: missing_data::MissingDataStrategy,
}
impl<T> FeatureSet<T> {
    /// Creates a new empty feature set.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            functions: HashMap::new(),
            missing_strategy: missing_data::MissingDataStrategy::ZeroFill,
        }
    }
    /// Adds a built-in feature to the set.
    pub fn add_builtin(mut self, builtin: BuiltinFeature) -> Self
    where
        T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
           + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Into<f64> + Send + Sync,
    {
        self.features.push(Box::new(builtin));
        self
    }
    /// Adds a custom function as a feature.
    pub fn add_function(mut self, name: &str, f: FeatureFn<T>) -> Self {
        self.functions.insert(name.to_string(), f);
        self
    }
}
impl<T> Default for FeatureSet<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> FeatureSet<T> {
    /// Sets the missing data handling strategy.
    pub fn with_missing_data_strategy(mut self, strategy: missing_data::MissingDataStrategy) -> Self {
        self.missing_strategy = strategy;
        self
    }
    /// Returns the number of features in the set.
    pub fn len(&self) -> usize {
        self.features.len() + self.functions.len()
    }
    /// Returns whether the feature set is empty.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty() && self.functions.is_empty()
    }
}
