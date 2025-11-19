#include <metal_stdlib>
using namespace metal;

/// Optimized Natural Visibility Kernel - O(n) per thread using monotonic stack approach
/// Each thread maintains a local stack and processes forward visibility
kernel void natural_visibility_kernel(
    device const float* data [[buffer(0)]],
    device uint2* edges [[buffer(1)]],
    device atomic_uint* edge_count [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    // Thread-local stack for monotonic envelope (max 256 elements should be enough for local processing)
    uint stack[256];
    uint stack_size = 0;

    // Process visibility from this node forward
    // We'll check visibility to nearby nodes (within a window)
    // This approximates the O(n) algorithm while maintaining parallelism
    const uint WINDOW_SIZE = min(n - gid, 256u);

    for (uint offset = 1; offset < WINDOW_SIZE; offset++) {
        uint j = gid + offset;
        if (j >= n) break;

        float yi = data[gid];
        float yj = data[j];

        // Check visibility using the stack-based approach
        bool visible = true;

        // Update stack: remove points that would be hidden by line from gid to j
        while (stack_size >= 2) {
            uint k = stack[stack_size - 1];
            uint m = stack[stack_size - 2];

            float yk = data[k];
            float ym = data[m];

            // Check if k is hidden by the line from m to j
            float slope_mj = (yj - ym) / float(j - m);
            float slope_mk = (yk - ym) / float(k - m);

            if (slope_mj > slope_mk) {
                stack_size--;
            } else {
                break;
            }
        }

        // Check visibility against all points in stack
        for (uint s = 0; s < stack_size; s++) {
            uint k = stack[s];
            float yk = data[k];

            // Line from gid to j
            float line_height = yi + (yj - yi) * float(k - gid) / float(j - gid);

            if (yk >= line_height - 1e-6) {
                visible = false;
                break;
            }
        }

        if (visible) {
            uint idx = atomic_fetch_add_explicit(edge_count, 1, memory_order_relaxed);
            edges[idx] = uint2(gid, j);
        }

        // Add current point to stack if there's room
        if (stack_size < 256) {
            stack[stack_size++] = j;
        }
    }
}

/// Horizontal Visibility Kernel
/// Computes visibility based on horizontal criterion (intermediate values must be lower)
kernel void horizontal_visibility_kernel(
    device const float* data [[buffer(0)]],
    device uint2* edges [[buffer(1)]],
    device atomic_uint* edge_count [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    // Each thread processes visibility from node gid to all later nodes
    for (uint j = gid + 1; j < n; j++) {
        bool visible = true;
        float yi = data[gid];
        float yj = data[j];
        float min_h = min(yi, yj);

        // Check all intermediate points
        for (uint k = gid + 1; k < j; k++) {
            float yk = data[k];

            if (yk >= min_h) {
                visible = false;
                break;
            }
        }

        if (visible) {
            uint idx = atomic_fetch_add_explicit(edge_count, 1, memory_order_relaxed);
            edges[idx] = uint2(gid, j);
        }
    }
}

/// Optimized Natural Visibility with Shared Memory
/// Uses threadgroup memory for better performance on larger datasets
kernel void natural_visibility_kernel_optimized(
    device const float* data [[buffer(0)]],
    device uint2* edges [[buffer(1)]],
    device atomic_uint* edge_count [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    // Shared memory for data caching
    threadgroup float shared_data[256];

    // Load data into shared memory
    if (lid < min(256u, n)) {
        uint data_idx = gid - lid + lid;
        if (data_idx < n) {
            shared_data[lid] = data[data_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= n) return;

    float yi = shared_data[lid];

    for (uint j = gid + 1; j < n; j++) {
        bool visible = true;
        float yj = data[j];
        float slope = (yj - yi) / float(j - gid);

        for (uint k = gid + 1; k < j; k++) {
            float yk = data[k];
            float line_height = yi + slope * float(k - gid);

            if (yk >= line_height) {
                visible = false;
                break;
            }
        }

        if (visible) {
            uint idx = atomic_fetch_add_explicit(edge_count, 1, memory_order_relaxed);
            edges[idx] = uint2(gid, j);
        }
    }
}

