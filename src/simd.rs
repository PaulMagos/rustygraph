//! SIMD-accelerated numerical operations for performance-critical paths.
//!
//! This module provides SIMD (Single Instruction, Multiple Data) optimizations
//! for numerical computations in visibility graph algorithms and feature computation.
//!
//! Requires the `simd` cargo feature flag.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated operations for f64 arrays.
pub struct SimdOps;

impl SimdOps {
    /// Computes sum of array using SIMD when available.
    ///
    /// Falls back to scalar implementation on non-x86_64 platforms.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::simd::SimdOps;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let sum = SimdOps::sum_f64(&data);
    /// assert_eq!(sum, 36.0);
    /// ```
    #[inline]
    pub fn sum_f64(data: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::sum_f64_avx2(data) }
            } else {
                Self::sum_f64_scalar(data)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::sum_f64_scalar(data)
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_f64_avx2(data: &[f64]) -> f64 {
        let mut sum = _mm256_setzero_pd();
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let values = _mm256_loadu_pd(chunk.as_ptr());
            sum = _mm256_add_pd(sum, values);
        }
        
        // Horizontal sum of SIMD register
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum);
        let simd_sum = result.iter().sum::<f64>();
        
        // Add remainder
        simd_sum + remainder.iter().sum::<f64>()
    }
    
    #[inline]
    fn sum_f64_scalar(data: &[f64]) -> f64 {
        data.iter().sum()
    }
    
    /// Computes mean of array using SIMD when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::simd::SimdOps;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let mean = SimdOps::mean_f64(&data);
    /// assert_eq!(mean, 2.5);
    /// ```
    #[inline]
    pub fn mean_f64(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        Self::sum_f64(data) / data.len() as f64
    }
    
    /// Computes dot product using SIMD when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::simd::SimdOps;
    ///
    /// let a = vec![1.0, 2.0, 3.0, 4.0];
    /// let b = vec![2.0, 3.0, 4.0, 5.0];
    /// let dot = SimdOps::dot_product_f64(&a, &b);
    /// assert_eq!(dot, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 40
    /// ```
    #[inline]
    pub fn dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::dot_product_f64_avx2(a, b) }
            } else {
                Self::dot_product_f64_scalar(a, b)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::dot_product_f64_scalar(a, b)
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = _mm256_setzero_pd();
        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm256_loadu_pd(chunk_a.as_ptr());
            let vb = _mm256_loadu_pd(chunk_b.as_ptr());
            let prod = _mm256_mul_pd(va, vb);
            sum = _mm256_add_pd(sum, prod);
        }
        
        // Horizontal sum
        let mut result = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum);
        let simd_sum = result.iter().sum::<f64>();
        
        // Add remainder
        let remainder_sum: f64 = remainder_a
            .iter()
            .zip(remainder_b.iter())
            .map(|(x, y)| x * y)
            .sum();
        
        simd_sum + remainder_sum
    }
    
    #[inline]
    fn dot_product_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    /// Computes variance using SIMD when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustygraph::simd::SimdOps;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let variance = SimdOps::variance_f64(&data);
    /// assert!((variance - 2.0).abs() < 1e-10);
    /// ```
    pub fn variance_f64(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let mean = Self::mean_f64(data);
        let squared_diffs: Vec<f64> = data.iter().map(|&x| (x - mean).powi(2)).collect();
        Self::sum_f64(&squared_diffs) / data.len() as f64
    }
    
    /// Computes element-wise addition using SIMD when available.
    ///
    /// Stores result in `result` slice.
    #[inline]
    pub fn add_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        assert_eq!(a.len(), result.len(), "Result array must have same length");
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::add_f64_avx2(a, b, result) }
            } else {
                Self::add_f64_scalar(a, b, result)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::add_f64_scalar(a, b, result)
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let chunks_r = result.chunks_exact_mut(4);
        
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        let remainder_r = chunks_r.into_remainder();
        
        for ((chunk_a, chunk_b), chunk_r) in chunks_a.zip(chunks_b).zip(result.chunks_exact_mut(4)) {
            let va = _mm256_loadu_pd(chunk_a.as_ptr());
            let vb = _mm256_loadu_pd(chunk_b.as_ptr());
            let sum = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(chunk_r.as_mut_ptr(), sum);
        }
        
        // Handle remainder
        for i in 0..remainder_a.len() {
            remainder_r[i] = remainder_a[i] + remainder_b[i];
        }
    }
    
    #[inline]
    fn add_f64_scalar(a: &[f64], b: &[f64], result: &mut [f64]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = SimdOps::sum_f64(&data);
        assert_eq!(sum, 15.0);
    }
    
    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mean = SimdOps::mean_f64(&data);
        assert_eq!(mean, 2.5);
    }
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let dot = SimdOps::dot_product_f64(&a, &b);
        assert_eq!(dot, 40.0);
    }
    
    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = SimdOps::variance_f64(&data);
        assert!((variance - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];
        SimdOps::add_f64(&a, &b, &mut result);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }
    
    #[test]
    fn test_simd_with_non_aligned_length() {
        // Test with length that's not a multiple of 4
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let sum = SimdOps::sum_f64(&data);
        assert_eq!(sum, 28.0);
    }
}

