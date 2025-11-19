# Compilation Fix: SIMD Borrow Checker Error

## Issue

Compilation error on Linux CI:
```
error[E0499]: cannot borrow `*result` as mutable more than once at a time
   --> src/performance/simd.rs:209:73
```

## Root Cause

In the AVX2 SIMD implementation, we were trying to borrow `result` mutably twice:
1. First with `result.chunks_exact_mut(4)` to get the chunks iterator
2. Second with `result.chunks_exact_mut(4)` in the for loop

This violates Rust's borrow checker rules.

## Fix

Changed from:
```rust
let chunks_r = result.chunks_exact_mut(4);
// ...
for ((chunk_a, chunk_b), chunk_r) in chunks_a.zip(chunks_b).zip(result.chunks_exact_mut(4)) {
    // This creates a second mutable borrow!
}
```

To:
```rust
let mut chunks_r = result.chunks_exact_mut(4);
// ...
for ((chunk_a, chunk_b), chunk_r) in chunks_a.zip(chunks_b).zip(&mut chunks_r) {
    // Now we're using the existing mutable borrow
}
```

## Solution Details

1. Create `chunks_r` with `mut` binding
2. Use `&mut chunks_r` in the iterator chain instead of calling `chunks_exact_mut` again
3. The remainder is still properly handled with `chunks_r.into_remainder()`

## Verification

✅ Builds successfully on Linux (x86_64)
✅ Builds successfully on macOS (ARM64)
✅ No functionality changed
✅ SIMD optimizations still work correctly

## Files Changed

- `src/performance/simd.rs` - Fixed double mutable borrow in `add_f64_avx2`

## Impact

- **Scope:** Compilation fix only
- **Functionality:** No change to behavior
- **Performance:** No impact
- **Compatibility:** Fixes Linux CI builds

---

**Status:** ✅ Fixed
**Date:** November 19, 2025

