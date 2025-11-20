//! Machine Learning Dataloader Example.
//!
//! Demonstrates the new ML dataloader capabilities for time series data,
//! including windowing, data splitting, and integration with Burn and PyTorch.

use rustygraph::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustyGraph ML Dataloader Example ===\n");

    // 1. Create sample time series data
    println!("1. CREATING SAMPLE DATA");
    let sine_data = datasets::sine_wave(100, 2.0, 1.0);
    let series = TimeSeries::from_raw(sine_data)?;
    println!("✓ Created time series with {} points", series.len());

    // 2. Create windowed datasets
    println!("\n2. WINDOWING TIME SERIES");
    let windows = WindowedTimeSeries::from_series(&series, 10, 2)?;
    println!("✓ Created {} windows of size {} with stride 2", windows.len(), windows.window_size);

    // Add synthetic labels for supervised learning
    let labels: Vec<f64> = (0..windows.len()).map(|i| (i as f64 * 0.1).sin()).collect();
    let windows_with_labels = windows.with_labels(labels);
    println!("✓ Added synthetic labels for supervised learning");

    // 3. Data splitting strategies
    println!("\n3. DATA SPLITTING STRATEGIES");

    // Time-based split (preserves temporal order)
    let time_split = split_windowed_data(windows_with_labels.clone(), SplitStrategy::TimeBased {
        train_frac: 0.7,
        val_frac: 0.2,
    })?;
    println!("✓ Time-based split: {} train, {} val, {} test",
        time_split.train.len(),
        time_split.val.as_ref().map(|v| v.len()).unwrap_or(0),
        time_split.test.len());

    // Rolling window split (for time series forecasting)
    let rolling_split = split_windowed_data(windows_with_labels.clone(), SplitStrategy::RollingWindow {
        train_windows: 20,
        val_windows: 5,
    })?;
    println!("✓ Rolling window split: {} train, {} val, {} test",
        rolling_split.train.len(),
        rolling_split.val.as_ref().map(|v| v.len()).unwrap_or(0),
        rolling_split.test.len());

    // 4. Burn Integration (Rust ML)
    println!("\n4. BURN INTEGRATION");
    #[cfg(feature = "burn-integration")]
    {
        use burn::backend::NdArray;
        type Backend = NdArray<f32>;

        let device = Default::default();

        // Create Burn dataset
        let dataset = rustygraph::burn::TimeSeriesDataset::from_windows(&time_split.train, &device);
        println!("✓ Created Burn dataset with shape: {:?}", dataset.inputs.shape());

        // Create data loader
        let data_loader = rustygraph::burn::TimeSeriesDataLoaderBuilder::new(device)
            .supervised_loader(&time_split.train, 8);

        println!("✓ Created Burn data loader with batch size 8");

        // Iterate through one batch
        if let Some(batch_result) = data_loader.iter().next() {
            let (inputs, targets) = batch_result?;
            println!("  Sample batch - inputs: {:?}, targets: {:?}",
                inputs.shape(), targets.shape());
        }
    }
    #[cfg(not(feature = "burn-integration"))]
    println!("✗ Burn integration not available. Enable with: --features burn-integration");

    // 5. Python/PyTorch Integration
    println!("\n5. PYTHON/PYTORCH INTEGRATION");
    #[cfg(feature = "python-bindings")]
    {
        println!("✓ Python bindings available");
        println!("  Use in Python:");
        println!("  ```python");
        println!("  import rustygraph as rg");
        println!("  import torch");
        println!("  from torch.utils.data import DataLoader");
        println!("  ");
        println!("  # Create windows");
        println!("  series = rg.TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])");
        println!("  windows = rg.WindowedTimeSeries.from_series(series, 3, 1)");
        println!("  ");
        println!("  # Split data");
        println!("  split = rg.split_windowed_data(windows, rg.SplitStrategy.TimeBased(0.7, 0.2))");
        println!("  ");
        println!("  # Create PyTorch DataLoader");
        println!("  dataset = rg.TimeSeriesDataset(split.train)");
        println!("  loader = DataLoader(dataset, batch_size=4)");
        println!("  ```");
    }
    #[cfg(not(feature = "python-bindings"))]
    println!("✗ Python bindings not available. Enable with: --features python-bindings");

    // 6. Multiple time series handling
    println!("\n6. MULTIPLE TIME SERIES HANDLING");
    let series2 = TimeSeries::from_raw(datasets::random_walk(80, 42))?; // Use integer seed
    let multiple_series = vec![series.clone(), series2];
    let multi_windows = WindowedTimeSeries::from_multiple_series(&multiple_series, 8, 3)?;
    println!("✓ Created windows from {} time series: {} total windows", multiple_series.len(), multi_windows.len());

    // Series-based split
    let series_split = split_windowed_data(multi_windows, SplitStrategy::SeriesBased {
        train_frac: 0.5,
        val_frac: 0.3,
    })?;
    println!("✓ Series-based split: {} train, {} val, {} test",
        series_split.train.len(),
        series_split.val.as_ref().map(|v| v.len()).unwrap_or(0),
        series_split.test.len());

    // 7. Performance considerations
    println!("\n7. PERFORMANCE FEATURES");
    println!("✓ Memory-efficient windowing (no data copying)");
    println!("✓ Lazy evaluation for large datasets");
    println!("✓ SIMD-accelerated operations available");
    println!("✓ Parallel processing support");

    println!("\n=== ML Dataloader Setup Complete ===");
    println!("Ready for training with Burn or PyTorch!");

    Ok(())
}
