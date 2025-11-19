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

/// SIMD-accelerated visibility checking operations.
///
/// These functions use SIMD instructions to accelerate the visibility
/// criterion checks in natural visibility graphs.
impl SimdOps {
    /// Checks if a line segment from (i, yi) to (j, yj) is blocked by any intermediate points.
    ///
    /// Uses SIMD to check multiple intermediate points simultaneously.
    ///
    /// # Arguments
    ///
    /// * `yi` - Value at point i
    /// * `yj` - Value at point j
    /// * `intermediate` - Slice of intermediate values to check
    /// * `i` - Index of start point
    /// * `j` - Index of end point
    ///
    /// # Returns
    ///
    /// `true` if the line is not blocked (visible), `false` otherwise
    #[inline]
    pub fn is_visible_natural_simd(yi: f64, yj: f64, intermediate: &[f64], start_idx: usize, end_idx: usize) -> bool {
        // NOTE: Parameters are (value_at_start, value_at_end, intermediate_values, start_index, end_index)
        // where start_idx < end_idx
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && intermediate.len() >= 4 {
                unsafe { Self::is_visible_natural_avx2(yi, yj, intermediate, start_idx, end_idx) }
            } else {
                Self::is_visible_natural_scalar(yi, yj, intermediate, start_idx, end_idx, start_idx + 1)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && intermediate.len() >= 2 {
                unsafe { Self::is_visible_natural_neon(yi, yj, intermediate, start_idx, end_idx) }
            } else {
                Self::is_visible_natural_scalar(yi, yj, intermediate, start_idx, end_idx, start_idx + 1)
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::is_visible_natural_scalar(yi, yj, intermediate, start_idx, end_idx, start_idx + 1)
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn is_visible_natural_avx2(
        yi: f64,
        yj: f64,
        intermediate: &[f64],
        start_idx: usize,
        end_idx: usize,
    ) -> bool {
        let yi_vec = _mm256_set1_pd(yi);
        let slope = (yj - yi) / ((end_idx - start_idx) as f64);
        let slope_vec = _mm256_set1_pd(slope);
        let chunks = intermediate.chunks_exact(4);
        let remainder = chunks.remainder();
        // Check 4 points at a time using SIMD
        for (idx, chunk) in chunks.enumerate() {
            let k_base = start_idx + 1 + idx * 4;
            // Load 4 intermediate values
            let yk = _mm256_loadu_pd(chunk.as_ptr());
            // Compute expected line heights for positions k, k+1, k+2, k+3
            // NOTE: _mm256_set_pd(d, c, b, a) creates vector [a, b, c, d] in memory order
            // So we need to pass parameters in REVERSE order
            let offsets = _mm256_set_pd(
                (k_base + 3 - start_idx) as f64,  // Lane 3
                (k_base + 2 - start_idx) as f64,  // Lane 2
                (k_base + 1 - start_idx) as f64,  // Lane 1
                (k_base - start_idx) as f64,      // Lane 0
            );
            // line_height = yi + slope * offset
            let line_height = _mm256_fmadd_pd(slope_vec, offsets, yi_vec);
            // Check if any yk >= line_height (blocked)
            let cmp = _mm256_cmp_pd(yk, line_height, _CMP_GE_OQ);
            let mask = _mm256_movemask_pd(cmp);
            if mask != 0 {
                return false; // At least one point blocks the view
            }
        }
        // Check remainder with scalar code
        let offset = intermediate.len() - remainder.len();
        let remainder_start_idx = start_idx + 1 + offset;
        Self::is_visible_natural_scalar(yi, yj, remainder, start_idx, end_idx, remainder_start_idx)
    }
    fn is_visible_natural_scalar(
        yi: f64,
        yj: f64,
        intermediate: &[f64],
        original_start_idx: usize,
        original_end_idx: usize,
        actual_start_idx: usize,
    ) -> bool {
        // Use ORIGINAL indices for slope calculation
        let slope = (yj - yi) / ((original_end_idx - original_start_idx) as f64);
        for (offset, &yk) in intermediate.iter().enumerate() {
            let k = actual_start_idx + offset;
            let line_height = yi + slope * ((k - original_start_idx) as f64);
            if yk >= line_height {
                return false;
            }
        }
        true
    }
    /// Checks horizontal visibility using SIMD.
    ///
    /// Two points are visible if all intermediate values are strictly less than min(yi, yj).
    pub fn is_visible_horizontal_simd(yi: f64, yj: f64, intermediate: &[f64]) -> bool {
        let min_h = yi.min(yj);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && intermediate.len() >= 4 {
                unsafe { Self::is_visible_horizontal_avx2(min_h, intermediate) }
            } else {
                intermediate.iter().all(|&yk| yk < min_h)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && intermediate.len() >= 2 {
                unsafe { Self::is_visible_horizontal_neon(min_h, intermediate) }
            } else {
                intermediate.iter().all(|&yk| yk < min_h)
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            intermediate.iter().all(|&yk| yk < min_h)
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn is_visible_horizontal_avx2(min_h: f64, intermediate: &[f64]) -> bool {
        let min_vec = _mm256_set1_pd(min_h);
        let chunks = intermediate.chunks_exact(4);
        let remainder = chunks.remainder();
        // Check 4 values at a time
        for chunk in chunks {
            let values = _mm256_loadu_pd(chunk.as_ptr());
            // Check if any value >= min_h
            let cmp = _mm256_cmp_pd(values, min_vec, _CMP_GE_OQ);
            let mask = _mm256_movemask_pd(cmp);
            if mask != 0 {
                return false; // At least one value blocks
            }
        }
        // Check remainder
        remainder.iter().all(|&yk| yk < min_h)
    }
}
// ============================================================================
// ARM NEON SIMD Implementations
// ============================================================================
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
impl SimdOps {
    /// ARM NEON implementation of natural visibility check.
    /// 
    /// Processes 2 f64 values per iteration using NEON instructions.
    #[target_feature(enable = "neon")]
    unsafe fn is_visible_natural_neon(
        yi: f64,
        yj: f64,
        intermediate: &[f64],
        start_idx: usize,
        end_idx: usize,
    ) -> bool {
        let slope = (yj - yi) / ((end_idx - start_idx) as f64);
        let yi_vec = vdupq_n_f64(yi);
        let slope_vec = vdupq_n_f64(slope);
        let chunks = intermediate.chunks_exact(2);
        let remainder = chunks.remainder();
        // Check 2 points at a time using NEON
        for (idx, chunk) in chunks.enumerate() {
            let k_base = start_idx + 1 + idx * 2;
            // Load 2 intermediate values
            let yk = vld1q_f64(chunk.as_ptr());
            // Compute expected line heights for positions k and k+1
            let offset0 = (k_base - start_idx) as f64;
            let offset1 = (k_base + 1 - start_idx) as f64;
            let offsets = vsetq_lane_f64::<1>(offset1, vdupq_n_f64(offset0));
            // line_height = yi + slope * offset
            let slope_times_offset = vmulq_f64(slope_vec, offsets);
            let line_height = vaddq_f64(yi_vec, slope_times_offset);
            // Check if any yk >= line_height (blocked)
            let cmp = vcgeq_f64(yk, line_height);
            // Extract comparison results - check if ANY lane is true
            // vcgeq_f64 returns uint64x2_t with 0xFFFFFFFFFFFFFFFF for true, 0 for false
            let mask0 = vgetq_lane_u64(cmp, 0);
            let mask1 = vgetq_lane_u64(cmp, 1);
            if mask0 != 0 || mask1 != 0 {
                return false; // At least one point blocks the view
            }
        }
        // Check remainder with scalar code
        let offset = intermediate.len() - remainder.len();
        let remainder_start_idx = start_idx + 1 + offset;
        Self::is_visible_natural_scalar(yi, yj, remainder, start_idx, end_idx, remainder_start_idx)
    }
    /// ARM NEON implementation of horizontal visibility check.
    /// 
    /// Processes 2 f64 values per iteration using NEON instructions.
    #[target_feature(enable = "neon")]
    unsafe fn is_visible_horizontal_neon(min_h: f64, intermediate: &[f64]) -> bool {
        let min_vec = vdupq_n_f64(min_h);
        let chunks = intermediate.chunks_exact(2);
        let remainder = chunks.remainder();
        // Check 2 values at a time
        for chunk in chunks {
            let values = vld1q_f64(chunk.as_ptr());
            // Check if any value >= min_h
            let cmp = vcgeq_f64(values, min_vec);
            // Extract comparison results
            let mask = vgetq_lane_u64(vreinterpretq_u64_u8(vandq_u8(
                vreinterpretq_u8_u64(cmp),
                vdupq_n_u8(1)
            )), 0);
            if mask != 0 {
                return false; // At least one value blocks
            }
        }
        // Check remainder
        remainder.iter().all(|&yk| yk < min_h)
    }
}
