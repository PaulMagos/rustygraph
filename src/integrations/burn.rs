//! Burn integration for machine learning with time series data.
//!
//! This module provides Burn-compatible datasets and data loaders for
//! training ML models on windowed time series data.
//!
//! Requires the `burn-integration` feature flag.

#[cfg(feature = "burn-integration")]
use burn::data::dataloader::DataLoaderBuilder;
#[cfg(feature = "burn-integration")]
use burn::data::dataset::{Dataset, InMemDataset};
#[cfg(feature = "burn-integration")]
use burn::tensor::{Data, Tensor};
#[cfg(feature = "burn-integration")]
use burn::prelude::*;
use crate::WindowedTimeSeries;

/// Burn-compatible dataset for windowed time series data.
///
/// This dataset provides tensors suitable for training ML models on
/// time series windows with optional labels.
#[cfg(feature = "burn-integration")]
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset<B: Backend> {
    /// Input windows: shape [N, F, W] where N=windows, F=features, W=timesteps
    pub inputs: Tensor<B, 3>,
    /// Optional target labels: shape [N] for regression/classification
    pub targets: Option<Tensor<B, 1>>,
}

#[cfg(feature = "burn-integration")]
impl<B: Backend> TimeSeriesDataset<B> {
    /// Creates a dataset from windowed time series data.
    ///
    /// # Arguments
    ///
    /// - `windows`: Windowed time series data
    /// - `device`: Burn device for tensor allocation
    ///
    /// # Returns
    ///
    /// Burn-compatible dataset
    pub fn from_windows(windows: &WindowedTimeSeries<f32>, device: &B::Device) -> Self {
        let num_windows = windows.len();
        let num_features = windows.num_features;
        let window_size = windows.window_size;

        // Convert to flat vector for tensor creation
        let mut input_data = Vec::with_capacity(num_windows * num_features * window_size);

        for window in &windows.windows {
            for feature in window {
                input_data.extend_from_slice(feature);
            }
        }

        let inputs = Tensor::from_data(
            Data::new(input_data, [num_windows, num_features, window_size].into()),
            device,
        );

        let targets = windows.labels.as_ref().map(|labels| {
            Tensor::from_data(
                Data::new(labels.clone(), [num_windows].into()),
                device,
            )
        });

        Self { inputs, targets }
    }

    /// Returns the number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.inputs.shape().dims[0]
    }

    /// Returns true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "burn-integration")]
impl<B: Backend> Dataset<Tensor<B, 3>> for TimeSeriesDataset<B> {
    fn get(&self, index: usize) -> Tensor<B, 3> {
        self.inputs.clone().slice([index..index + 1, 0..self.inputs.shape().dims[1], 0..self.inputs.shape().dims[2]])
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Burn-compatible dataset for supervised learning with separate inputs and targets.
#[cfg(feature = "burn-integration")]
#[derive(Debug, Clone)]
pub struct SupervisedTimeSeriesDataset<B: Backend> {
    /// Input windows: shape [N, F, W]
    pub inputs: Tensor<B, 3>,
    /// Target labels: shape [N]
    pub targets: Tensor<B, 1>,
}

#[cfg(feature = "burn-integration")]
impl<B: Backend> SupervisedTimeSeriesDataset<B> {
    /// Creates a supervised dataset from windowed time series with labels.
    ///
    /// # Panics
    ///
    /// Panics if the windowed data doesn't have labels.
    pub fn from_windows(windows: &WindowedTimeSeries<f32>, device: &B::Device) -> Self {
        let dataset = TimeSeriesDataset::from_windows(windows, device);

        Self {
            inputs: dataset.inputs,
            targets: dataset.targets.expect("WindowedTimeSeries must have labels for supervised learning"),
        }
    }
}

#[cfg(feature = "burn-integration")]
impl<B: Backend> Dataset<(Tensor<B, 3>, Tensor<B, 1>)> for SupervisedTimeSeriesDataset<B> {
    fn get(&self, index: usize) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let input = self.inputs.clone().slice([index..index + 1, 0..self.inputs.shape().dims[1], 0..self.inputs.shape().dims[2]]);
        let target = self.targets.clone().slice([index..index + 1]);

        (input, target)
    }

    fn len(&self) -> usize {
        self.inputs.shape().dims[0]
    }
}

/// Builder for creating data loaders from time series data.
#[cfg(feature = "burn-integration")]
pub struct TimeSeriesDataLoaderBuilder<B: Backend> {
    device: B::Device,
}

#[cfg(feature = "burn-integration")]
impl<B: Backend> TimeSeriesDataLoaderBuilder<B> {
    /// Creates a new data loader builder.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Creates a data loader for unsupervised learning (inputs only).
    pub fn unsupervised_loader(
        self,
        windows: &WindowedTimeSeries<f32>,
        batch_size: usize,
    ) -> burn::data::dataloader::DataLoader<TimeSeriesDataset<B>, B> {
        let dataset = TimeSeriesDataset::from_windows(windows, &self.device);
        DataLoaderBuilder::new(dataset)
            .batch_size(batch_size)
            .build()
    }

    /// Creates a data loader for supervised learning (inputs + targets).
    ///
    /// # Panics
    ///
    /// Panics if the windowed data doesn't have labels.
    pub fn supervised_loader(
        self,
        windows: &WindowedTimeSeries<f32>,
        batch_size: usize,
    ) -> burn::data::dataloader::DataLoader<SupervisedTimeSeriesDataset<B>, B> {
        let dataset = SupervisedTimeSeriesDataset::from_windows(windows, &self.device);
        DataLoaderBuilder::new(dataset)
            .batch_size(batch_size)
            .build()
    }
}

#[cfg(test)]
#[cfg(feature = "burn-integration")]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type Backend = NdArray<f32>;

    #[test]
    fn test_dataset_creation() {
        let device = Default::default();

        // Create sample windowed data
        let windows = vec![
            vec![vec![1.0, 2.0, 3.0]], // 1 feature, 3 timesteps
            vec![vec![4.0, 5.0, 6.0]],
        ];
        let labels = Some(vec![0.0, 1.0]);

        let windowed = WindowedTimeSeries {
            windows,
            labels,
            series_indices: vec![0, 0],
            window_starts: vec![0, 1],
            window_size: 3,
            num_features: 1,
        };

        let dataset = TimeSeriesDataset::from_windows(&windowed, &device);
        assert_eq!(dataset.len(), 2);
        assert!(dataset.targets.is_some());
    }
}
