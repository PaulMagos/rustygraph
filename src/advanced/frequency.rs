//! Advanced features including frequency domain and wavelet analysis.
//!
//! This module provides advanced feature computation capabilities that go beyond
//! simple time-domain features. Requires the `advanced-features` cargo feature.

#[cfg(feature = "advanced-features")]
use rustfft::{FftPlanner, num_complex::Complex};

/// Frequency domain features using FFT.
///
/// Requires `advanced-features` cargo feature.
#[cfg(feature = "advanced-features")]
pub struct FrequencyFeatures;

#[cfg(feature = "advanced-features")]
impl FrequencyFeatures {
    /// Computes FFT coefficients for a time series window.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `center` - Center index
    /// * `window_size` - Size of the window
    ///
    /// # Returns
    ///
    /// Vector of FFT coefficient magnitudes
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "advanced-features")]
    /// # {
    /// use rustygraph::advanced::FrequencyFeatures;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
    /// let fft_mags = FrequencyFeatures::compute_fft_window(&data, 4, 8);
    /// assert_eq!(fft_mags.len(), 8);
    /// # }
    /// ```
    pub fn compute_fft_window(data: &[f64], center: usize, window_size: usize) -> Vec<f64> {
        let start = center.saturating_sub(window_size / 2);
        let end = (center + window_size / 2).min(data.len());
        let window: Vec<f64> = data[start..end].to_vec();

        // Pad to window_size if necessary
        let mut padded = window.clone();
        while padded.len() < window_size {
            padded.push(0.0);
        }

        // Convert to complex numbers
        let mut buffer: Vec<Complex<f64>> = padded
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Perform FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);

        // Return magnitudes
        buffer.iter().map(|c| c.norm()).collect()
    }

    /// Computes dominant frequency from FFT.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `center` - Center index
    /// * `window_size` - Size of the window
    ///
    /// # Returns
    ///
    /// Index of dominant frequency component
    pub fn dominant_frequency(data: &[f64], center: usize, window_size: usize) -> usize {
        let fft_mags = Self::compute_fft_window(data, center, window_size);

        // Skip DC component (index 0) and find max
        fft_mags
            .iter()
            .enumerate()
            .skip(1)
            .take(fft_mags.len() / 2) // Only first half (real frequencies)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Computes spectral energy in a frequency band.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `center` - Center index
    /// * `window_size` - Size of the window
    /// * `band_start` - Start of frequency band
    /// * `band_end` - End of frequency band
    ///
    /// # Returns
    ///
    /// Total energy in the specified band
    pub fn spectral_energy(
        data: &[f64],
        center: usize,
        window_size: usize,
        band_start: usize,
        band_end: usize,
    ) -> f64 {
        let fft_mags = Self::compute_fft_window(data, center, window_size);

        fft_mags[band_start..band_end.min(fft_mags.len())]
            .iter()
            .map(|&x| x * x)
            .sum()
    }
}

/// Simple wavelet-based features using Haar wavelet.
///
/// Provides multi-scale analysis without external dependencies.
pub struct WaveletFeatures;

impl WaveletFeatures {
    /// Computes Haar wavelet transform coefficients.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data (must be power of 2 length)
    ///
    /// # Returns
    ///
    /// Tuple of (approximation coefficients, detail coefficients)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::advanced::WaveletFeatures;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let (approx, detail) = WaveletFeatures::haar_transform(&data);
    /// assert_eq!(approx.len(), 4);
    /// assert_eq!(detail.len(), 4);
    /// ```
    pub fn haar_transform(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 2 {
            return (data.to_vec(), vec![]);
        }

        let mut approx = Vec::with_capacity(n / 2);
        let mut detail = Vec::with_capacity(n / 2);

        for i in (0..n).step_by(2) {
            if i + 1 < n {
                // Approximation (average)
                approx.push((data[i] + data[i + 1]) / 2.0_f64.sqrt());
                // Detail (difference)
                detail.push((data[i] - data[i + 1]) / 2.0_f64.sqrt());
            }
        }

        (approx, detail)
    }

    /// Computes multi-level wavelet decomposition.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `levels` - Number of decomposition levels
    ///
    /// # Returns
    ///
    /// Vector of detail coefficients at each level
    pub fn multi_level_decomposition(data: &[f64], levels: usize) -> Vec<Vec<f64>> {
        let mut current = data.to_vec();
        let mut details = Vec::new();

        for _ in 0..levels {
            if current.len() < 2 {
                break;
            }

            let (approx, detail) = Self::haar_transform(&current);
            details.push(detail);
            current = approx;
        }

        details
    }

    /// Computes wavelet energy at a specific scale.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `level` - Decomposition level
    ///
    /// # Returns
    ///
    /// Energy at the specified scale
    pub fn wavelet_energy(data: &[f64], level: usize) -> f64 {
        let details = Self::multi_level_decomposition(data, level + 1);

        if level < details.len() {
            details[level].iter().map(|&x| x * x).sum()
        } else {
            0.0
        }
    }
}

/// Advanced time-domain features.
pub struct AdvancedFeatures;

impl AdvancedFeatures {
    /// Computes sample entropy (complexity measure).
    ///
    /// ⚠️ **Performance Warning:** This method has **O(n²)** complexity where n is the length
    /// of the data. It compares all pairs of patterns. For long time series (> 10,000 points),
    /// this can be slow. This is the standard complexity for sample entropy calculation.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `m` - Pattern length
    /// * `r` - Tolerance
    ///
    /// # Returns
    ///
    /// Sample entropy value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::advanced::AdvancedFeatures;
    ///
    /// let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
    /// let entropy = AdvancedFeatures::sample_entropy(&data, 2, 0.2);
    /// println!("Sample entropy: {}", entropy);
    /// ```
    pub fn sample_entropy(data: &[f64], m: usize, r: f64) -> f64 {
        let n = data.len();
        if n < m + 1 {
            return 0.0;
        }

        let phi_m = Self::compute_phi(data, m, r);
        let phi_m_plus_1 = Self::compute_phi(data, m + 1, r);

        Self::calculate_entropy_from_phi(phi_m, phi_m_plus_1)
    }

    /// Compute phi statistic for sample entropy
    fn compute_phi(data: &[f64], m: usize, r: f64) -> f64 {
        let n = data.len();
        if n < m {
            return 0.0;
        }

        let count = Self::count_matching_patterns(data, m, r);
        let total_comparisons = (n - m) * (n - m - 1);

        if total_comparisons > 0 {
            count as f64 / total_comparisons as f64
        } else {
            0.0
        }
    }

    /// Count matching patterns within tolerance
    fn count_matching_patterns(data: &[f64], m: usize, r: f64) -> usize {
        let n = data.len();
        let mut count = 0;

        for i in 0..n - m {
            for j in 0..n - m {
                if i != j && Self::patterns_match(data, i, j, m, r) {
                    count += 1;
                }
            }
        }

        count
    }

    /// Check if two patterns match within tolerance
    fn patterns_match(data: &[f64], i: usize, j: usize, m: usize, r: f64) -> bool {
        let max_diff = Self::max_pattern_difference(data, i, j, m);
        max_diff <= r
    }

    /// Find maximum difference between two patterns
    fn max_pattern_difference(data: &[f64], i: usize, j: usize, m: usize) -> f64 {
        (0..m)
            .map(|l| (data[i + l] - data[j + l]).abs())
            .fold(0.0, f64::max)
    }

    /// Calculate final entropy from phi values
    fn calculate_entropy_from_phi(phi_m: f64, phi_m_plus_1: f64) -> f64 {
        if phi_m > 0.0 && phi_m_plus_1 > 0.0 {
            -(phi_m_plus_1 / phi_m).ln()
        } else {
            0.0
        }
    }

    /// Computes Hurst exponent (long-range dependence).
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    ///
    /// # Returns
    ///
    /// Hurst exponent estimate
    pub fn hurst_exponent(data: &[f64]) -> f64 {
        let n = data.len();
        if n < 10 {
            return 0.5;
        }

        let mean = Self::calculate_mean(data);
        let std = Self::calculate_std(data, mean);

        if std == 0.0 {
            return 0.5;
        }

        let deviations = Self::compute_cumulative_deviations(data, mean);
        let range = Self::compute_range(&deviations);
        let rs = range / std;

        Self::calculate_hurst_from_rs(rs, n)
    }

    /// Calculate mean of data
    fn calculate_mean(data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation
    fn calculate_std(data: &[f64], mean: f64) -> f64 {
        let n = data.len();
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        variance.sqrt()
    }

    /// Compute cumulative deviations from mean
    fn compute_cumulative_deviations(data: &[f64], mean: f64) -> Vec<f64> {
        let mut cumsum = 0.0;
        data.iter()
            .map(|&x| {
                cumsum += x - mean;
                cumsum
            })
            .collect()
    }

    /// Compute range of deviations
    fn compute_range(deviations: &[f64]) -> f64 {
        let max = deviations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min = deviations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        max - min
    }

    /// Calculate Hurst exponent from R/S ratio
    fn calculate_hurst_from_rs(rs: f64, n: usize) -> f64 {
        if rs > 0.0 {
            rs.ln() / (n as f64).ln()
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_transform() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (approx, detail) = WaveletFeatures::haar_transform(&data);
        assert_eq!(approx.len(), 2);
        assert_eq!(detail.len(), 2);
    }

    #[test]
    fn test_multi_level_decomposition() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let details = WaveletFeatures::multi_level_decomposition(&data, 2);
        assert_eq!(details.len(), 2);
    }

    #[test]
    fn test_sample_entropy() {
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let entropy = AdvancedFeatures::sample_entropy(&data, 2, 0.2);
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_hurst_exponent() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let hurst = AdvancedFeatures::hurst_exponent(&data);
        assert!(hurst >= 0.0 && hurst <= 1.0);
    }
}
