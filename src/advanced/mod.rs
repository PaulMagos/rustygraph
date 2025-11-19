//! Advanced signal processing features.
//!
//! This module provides advanced features like FFT analysis and wavelet transforms.
//! Requires the `advanced-features` cargo feature.

#[cfg(feature = "advanced-features")]
pub mod frequency;

#[cfg(feature = "advanced-features")]
pub use self::frequency::{FrequencyFeatures, WaveletFeatures, AdvancedFeatures};

