//! Example datasets for testing and benchmarking visibility graphs.
//!
//! This module provides various synthetic and real-world-inspired time series
//! for testing, benchmarking, and demonstrating the library.

use std::f64::consts::PI;

/// Generates a sine wave time series.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `frequency` - Frequency of the sine wave
/// * `amplitude` - Amplitude of the wave
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::sine_wave;
///
/// let data = sine_wave(100, 1.0, 1.0);
/// assert_eq!(data.len(), 100);
/// ```
pub fn sine_wave(n: usize, frequency: f64, amplitude: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amplitude * (2.0 * PI * frequency * i as f64 / n as f64).sin())
        .collect()
}

/// Generates a random walk time series.
///
/// # Arguments
///
/// * `n` - Number of steps
/// * `seed` - Seed for reproducibility
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::random_walk;
///
/// let data = random_walk(100, 42);
/// assert_eq!(data.len(), 100);
/// ```
pub fn random_walk(n: usize, seed: u64) -> Vec<f64> {

    let mut rng_state = seed;
    let mut value = 0.0;
    let mut result = Vec::with_capacity(n);

    for _ in 0..n {
        // Simple LCG random number generator
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let step = if (rng_state & 1) == 0 { 1.0 } else { -1.0 };
        value += step;
        result.push(value);
    }

    result
}

/// Generates a time series with trend and noise.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `trend` - Trend slope
/// * `noise_amplitude` - Amplitude of noise
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::trend_with_noise;
///
/// let data = trend_with_noise(100, 0.1, 0.5, 42);
/// assert_eq!(data.len(), 100);
/// ```
pub fn trend_with_noise(n: usize, trend: f64, noise_amplitude: f64, seed: u64) -> Vec<f64> {
    let mut rng_state = seed;
    (0..n)
        .map(|i| {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (rng_state as f64 / u64::MAX as f64 - 0.5) * noise_amplitude;
            i as f64 * trend + noise
        })
        .collect()
}

/// Generates a periodic signal with multiple frequencies.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `frequencies` - Slice of frequencies to combine
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::multi_frequency;
///
/// let data = multi_frequency(100, &[1.0, 2.0, 3.0]);
/// assert_eq!(data.len(), 100);
/// ```
pub fn multi_frequency(n: usize, frequencies: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| {
            frequencies
                .iter()
                .map(|&f| (2.0 * PI * f * i as f64 / n as f64).sin())
                .sum()
        })
        .collect()
}

/// Generates a chaotic Logistic Map time series.
///
/// The logistic map is: x_{n+1} = r * x_n * (1 - x_n)
///
/// # Arguments
///
/// * `n` - Number of iterations
/// * `r` - Chaos parameter (typically 3.5-4.0 for chaos)
/// * `x0` - Initial condition (between 0 and 1)
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::logistic_map;
///
/// let data = logistic_map(100, 3.9, 0.5);
/// assert_eq!(data.len(), 100);
/// ```
pub fn logistic_map(n: usize, r: f64, x0: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(n);
    let mut x = x0;

    for _ in 0..n {
        result.push(x);
        x = r * x * (1.0 - x);
    }

    result
}

/// Generates a time series with step changes (regime shifts).
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `regimes` - Number of regimes
/// * `levels` - Levels for each regime
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::step_function;
///
/// let data = step_function(100, &[0.0, 1.0, 0.5, 2.0]);
/// assert_eq!(data.len(), 100);
/// ```
pub fn step_function(n: usize, levels: &[f64]) -> Vec<f64> {
    let regime_length = n / levels.len();
    let mut result = Vec::with_capacity(n);

    for (i, &level) in levels.iter().enumerate() {
        let start = i * regime_length;
        let end = if i == levels.len() - 1 {
            n
        } else {
            (i + 1) * regime_length
        };

        for _ in start..end {
            result.push(level);
        }
    }

    result
}

/// Generates a sawtooth wave.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `period` - Period of the sawtooth
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::sawtooth;
///
/// let data = sawtooth(100, 20);
/// assert_eq!(data.len(), 100);
/// ```
pub fn sawtooth(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i % period) as f64 / period as f64)
        .collect()
}

/// Generates a square wave.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `period` - Period of the square wave
///
/// # Examples
///
/// ```rust
/// use rustygraph::datasets::square_wave;
///
/// let data = square_wave(100, 20);
/// assert_eq!(data.len(), 100);
/// ```
pub fn square_wave(n: usize, period: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if (i / (period / 2)) % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_wave() {
        let data = sine_wave(100, 1.0, 1.0);
        assert_eq!(data.len(), 100);
        assert!(data[0].abs() < 0.1); // Should start near zero
    }

    #[test]
    fn test_random_walk() {
        let data1 = random_walk(100, 42);
        let data2 = random_walk(100, 42);
        assert_eq!(data1, data2); // Same seed = same walk
    }

    #[test]
    fn test_logistic_map() {
        let data = logistic_map(100, 3.9, 0.5);
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}

