fn main() {
    println!("Checking which SIMD features are available:");

    #[cfg(target_arch = "x86_64")]
    {
        println!("  Architecture: x86_64");
        if is_x86_feature_detected!("avx2") {
            println!("  AVX2: AVAILABLE ✅");
        } else {
            println!("  AVX2: NOT AVAILABLE ❌");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("  Architecture: aarch64 (ARM)");
        if std::arch::is_aarch64_feature_detected!("neon") {
            println!("  NEON: AVAILABLE ✅");
        } else {
            println!("  NEON: NOT AVAILABLE ❌");
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("  Architecture: Other (no SIMD)");
    }
}

