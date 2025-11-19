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

        let mut phi = vec![0.0; 2];

        for k in 0..2 {
            let m_curr = m + k;
            let mut count = 0;

            for i in 0..n - m_curr {
                for j in 0..n - m_curr {
                    if i != j {
                        let mut max_diff = 0.0;
                        for l in 0..m_curr {
                            let diff = (data[i + l] - data[j + l]).abs();
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }
                        if max_diff <= r {
                            count += 1;
                        }
                    }
                }
            }

            phi[k] = (count as f64) / ((n - m_curr) * (n - m_curr - 1)) as f64;
        }

        if phi[0] > 0.0 && phi[1] > 0.0 {
            -(phi[1] / phi[0]).ln()
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

        // Simplified R/S analysis
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let std: f64 = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();

        if std == 0.0 {
            return 0.5;
        }

        // Cumulative deviations
        let mut cumsum = 0.0;
        let mut deviations = Vec::new();
        for &x in data {
            cumsum += x - mean;
            deviations.push(cumsum);
        }

        let range = deviations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - deviations.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let rs = range / std;

        // Hurst = log(R/S) / log(n)
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

