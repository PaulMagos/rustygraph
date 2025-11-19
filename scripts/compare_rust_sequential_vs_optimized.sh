#!/bin/bash
# Compare Sequential vs Parallel+SIMD Rust Performance

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     Rust Performance: Sequential vs Parallel+SIMD Comparison      ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "This will compile and run the benchmark twice:"
echo "  1. WITHOUT optimizations (sequential only)"
echo "  2. WITH optimizations (parallel + SIMD)"
echo ""
echo "Press Enter to continue..."
read

echo "════════════════════════════════════════════════════════════════════"
echo "PART 1: Sequential Rust (NO parallel, NO SIMD)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

cargo run --example rust_benchmark --no-default-features --release > /tmp/sequential_output.txt 2>&1
cat /tmp/sequential_output.txt

echo ""
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "PART 2: Optimized Rust (WITH parallel, WITH SIMD)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

cargo run --example rust_benchmark --features parallel,simd --release > /tmp/optimized_output.txt 2>&1
cat /tmp/optimized_output.txt

echo ""
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "COMPARISON SUMMARY"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Extract timing data and compare
echo "Extracting timing data..."
echo ""

grep "Average:" /tmp/sequential_output.txt | while read line; do
    size=$(echo "$line" | grep -o "Test size: [0-9]*" | grep -o "[0-9]*" || echo "unknown")
    time=$(echo "$line" | grep -o "[0-9.]*ms" | head -1)
    echo "Sequential - $size nodes: $time"
done

echo ""

grep "Average:" /tmp/optimized_output.txt | while read line; do
    size=$(echo "$line" | grep -o "Test size: [0-9]*" | grep -o "[0-9]*" || echo "unknown")
    time=$(echo "$line" | grep -o "[0-9.]*ms" | head -1)
    echo "Optimized  - $size nodes: $time"
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  To see the difference clearly, compare the times above            ║"
echo "║  Speedup = Sequential Time / Optimized Time                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

