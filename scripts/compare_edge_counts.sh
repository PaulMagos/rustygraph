#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Edge Count Comparison: Sequential vs Optimized           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Building both versions..."
cargo build --example correctness_test --no-default-features --release 2>&1 > /dev/null
mv target/release/examples/correctness_test /tmp/correctness_test_seq

cargo build --example correctness_test --features parallel,simd --release 2>&1 > /dev/null
mv target/release/examples/correctness_test /tmp/correctness_test_opt

echo ""
echo "Running tests..."
echo ""

echo "Sequential (no optimizations):" > /tmp/seq_output.txt
/tmp/correctness_test_seq 2>&1 | grep -E "(Testing with|Edges:|✅|❌)" >> /tmp/seq_output.txt

echo "Optimized (SIMD + parallel):" > /tmp/opt_output.txt
/tmp/correctness_test_opt 2>&1 | grep -E "(Testing with|Edges:|✅|❌)" >> /tmp/opt_output.txt

echo "═══════════════════════════════════════════════════════════"
echo "SEQUENTIAL RESULTS:"
echo "═══════════════════════════════════════════════════════════"
cat /tmp/seq_output.txt
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "OPTIMIZED RESULTS:"
echo "═══════════════════════════════════════════════════════════"
cat /tmp/opt_output.txt
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "COMPARISON:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Size  | Sequential | Optimized  | Difference"
echo "------|------------|------------|------------"

# Extract edge counts
for size in 50 100 200 500 1000 2000 5000; do
    seq_edges=$(grep -A 1 "Testing with $size nodes" /tmp/seq_output.txt | grep "Edges:" | grep -o '[0-9]*')
    opt_edges=$(grep -A 1 "Testing with $size nodes" /tmp/opt_output.txt | grep "Edges:" | grep -o '[0-9]*')

    if [ -n "$seq_edges" ] && [ -n "$opt_edges" ]; then
        diff=$((seq_edges - opt_edges))
        printf "%5d | %10s | %10s | %+d\n" "$size" "$seq_edges" "$opt_edges" "$diff"
    fi
done

echo ""
echo "If differences are non-zero, there's still a bug!"

