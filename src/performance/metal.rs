//! Metal shader implementations for Apple Silicon GPU acceleration.
//!
//! This module provides Metal compute shaders for visibility graph computation
//! on Apple Silicon GPUs, leveraging the unified memory architecture and
//! Neural Engine capabilities.

#[cfg(target_os = "macos")]
use metal::*;

/// Metal compute pipeline for visibility graph construction
#[cfg(target_os = "macos")]
pub struct MetalVisibilityPipeline {
    device: Device,
    command_queue: CommandQueue,
    natural_visibility_pipeline: ComputePipelineState,
    horizontal_visibility_pipeline: ComputePipelineState,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalVisibilityPipeline {
    /// Creates a new Metal compute pipeline
    pub fn new() -> Result<Self, String> {
        use metal::*;

        // Get default Metal device
        let device = Device::system_default()
            .ok_or("No Metal device found")?;

        // Create command queue
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = Self::create_shader_library(&device)?;

        // Create compute pipelines
        let natural_visibility_pipeline = Self::create_pipeline(
            &device,
            &library,
            "natural_visibility_kernel"
        )?;

        let horizontal_visibility_pipeline = Self::create_pipeline(
            &device,
            &library,
            "horizontal_visibility_kernel"
        )?;

        Ok(Self {
            device,
            command_queue,
            natural_visibility_pipeline,
            horizontal_visibility_pipeline,
        })
    }

    fn create_shader_library(device: &metal::Device) -> Result<metal::Library, String> {
        let shader_source = include_str!("shaders/visibility.metal");

        let options = metal::CompileOptions::new();
        device.new_library_with_source(shader_source, &options)
            .map_err(|e| format!("Failed to compile Metal shaders: {}", e))
    }

    fn create_pipeline(
        device: &metal::Device,
        library: &metal::Library,
        function_name: &str,
    ) -> Result<metal::ComputePipelineState, String> {
        let function = library.get_function(function_name, None)
            .map_err(|e| format!("Failed to get function {}: {}", function_name, e))?;

        device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline state: {}", e))
    }

    /// Computes natural visibility edges using Metal GPU
    pub fn compute_natural_visibility(
        &self,
        data: &[f64],
    ) -> Result<Vec<(usize, usize)>, String> {
        use metal::*;

        let n = data.len();

        // Convert f64 to f32 for Metal (Metal doesn't support double precision)
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        // Create Metal buffers
        let data_buffer = self.device.new_buffer_with_data(
            data_f32.as_ptr() as *const _,
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Allocate output buffer for edges (worst case: n*(n-1)/2 edges)
        let max_edges = n * (n - 1) / 2;
        let edge_buffer = self.device.new_buffer(
            (max_edges * 2 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let edge_count_buffer = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.natural_visibility_pipeline);
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&edge_buffer), 0);
        encoder.set_buffer(2, Some(&edge_count_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &n as *const usize as *const _);

        // Calculate thread configuration
        let thread_group_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let thread_groups = MTLSize {
            width: ((n + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        // Execute and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let edge_count_ptr = edge_count_buffer.contents() as *const u32;
        let edge_count = unsafe { *edge_count_ptr } as usize;

        let edges_ptr = edge_buffer.contents() as *const u32;
        let mut edges = Vec::with_capacity(edge_count);

        for i in 0..edge_count {
            unsafe {
                let src = *edges_ptr.add(i * 2) as usize;
                let dst = *edges_ptr.add(i * 2 + 1) as usize;
                edges.push((src, dst));
            }
        }

        Ok(edges)
    }

    /// Computes horizontal visibility edges using Metal GPU
    pub fn compute_horizontal_visibility(
        &self,
        data: &[f64],
    ) -> Result<Vec<(usize, usize)>, String> {
        use metal::*;

        let n = data.len();

        // Convert f64 to f32 for Metal (Metal doesn't support double precision)
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        // Similar to natural visibility but uses horizontal kernel
        let data_buffer = self.device.new_buffer_with_data(
            data_f32.as_ptr() as *const _,
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let max_edges = n * (n - 1) / 2;
        let edge_buffer = self.device.new_buffer(
            (max_edges * 2 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let edge_count_buffer = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.horizontal_visibility_pipeline);
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&edge_buffer), 0);
        encoder.set_buffer(2, Some(&edge_count_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &n as *const usize as *const _);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_groups = MTLSize { width: ((n + 255) / 256) as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let edge_count_ptr = edge_count_buffer.contents() as *const u32;
        let edge_count = unsafe { *edge_count_ptr } as usize;

        let edges_ptr = edge_buffer.contents() as *const u32;
        let mut edges = Vec::with_capacity(edge_count);

        for i in 0..edge_count {
            unsafe {
                let src = *edges_ptr.add(i * 2) as usize;
                let dst = *edges_ptr.add(i * 2 + 1) as usize;
                edges.push((src, dst));
            }
        }

        Ok(edges)
    }

    /// Batch process multiple time series to amortize GPU overhead
    /// Returns a vector of edge lists, one for each input series
    pub fn compute_natural_visibility_batch(
        &self,
        data_batch: &[&[f64]],
    ) -> Result<Vec<Vec<(usize, usize)>>, String> {
        use metal::*;

        if data_batch.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_results = Vec::with_capacity(data_batch.len());
        
        // Process all series using a single command buffer for efficiency
        let command_buffer = self.command_queue.new_command_buffer();

        for data in data_batch {
            let n = data.len();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

            // Create buffers for this series
            let data_buffer = self.device.new_buffer_with_data(
                data_f32.as_ptr() as *const _,
                (n * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let max_edges = n * (n - 1) / 2;
            let edge_buffer = self.device.new_buffer(
                (max_edges * 2 * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let edge_count_buffer = self.device.new_buffer(
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Encode compute command
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.natural_visibility_pipeline);
            encoder.set_buffer(0, Some(&data_buffer), 0);
            encoder.set_buffer(1, Some(&edge_buffer), 0);
            encoder.set_buffer(2, Some(&edge_count_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &n as *const usize as *const _);

            let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
            let thread_groups = MTLSize { width: ((n + 255) / 256) as u64, height: 1, depth: 1 };

            encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            encoder.end_encoding();

            // Store buffers for reading later
            all_results.push((edge_buffer, edge_count_buffer));
        }

        // Execute all compute commands in one batch
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read all results
        let mut results = Vec::with_capacity(data_batch.len());
        for (edge_buffer, edge_count_buffer) in all_results {
            let edge_count_ptr = edge_count_buffer.contents() as *const u32;
            let edge_count = unsafe { *edge_count_ptr } as usize;

            let edges_ptr = edge_buffer.contents() as *const u32;
            let mut edges = Vec::with_capacity(edge_count);

            for i in 0..edge_count {
                unsafe {
                    let src = *edges_ptr.add(i * 2) as usize;
                    let dst = *edges_ptr.add(i * 2 + 1) as usize;
                    edges.push((src, dst));
                }
            }

            results.push(edges);
        }

        Ok(results)
    }

    /// Batch process horizontal visibility for multiple time series
    pub fn compute_horizontal_visibility_batch(
        &self,
        data_batch: &[&[f64]],
    ) -> Result<Vec<Vec<(usize, usize)>>, String> {
        use metal::*;

        if data_batch.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_results = Vec::with_capacity(data_batch.len());
        let command_buffer = self.command_queue.new_command_buffer();

        for data in data_batch {
            let n = data.len();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

            let data_buffer = self.device.new_buffer_with_data(
                data_f32.as_ptr() as *const _,
                (n * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let max_edges = n * (n - 1) / 2;
            let edge_buffer = self.device.new_buffer(
                (max_edges * 2 * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let edge_count_buffer = self.device.new_buffer(
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.horizontal_visibility_pipeline);
            encoder.set_buffer(0, Some(&data_buffer), 0);
            encoder.set_buffer(1, Some(&edge_buffer), 0);
            encoder.set_buffer(2, Some(&edge_count_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &n as *const usize as *const _);

            let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
            let thread_groups = MTLSize { width: ((n + 255) / 256) as u64, height: 1, depth: 1 };

            encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            encoder.end_encoding();

            all_results.push((edge_buffer, edge_count_buffer));
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut results = Vec::with_capacity(data_batch.len());
        for (edge_buffer, edge_count_buffer) in all_results {
            let edge_count_ptr = edge_count_buffer.contents() as *const u32;
            let edge_count = unsafe { *edge_count_ptr } as usize;

            let edges_ptr = edge_buffer.contents() as *const u32;
            let mut edges = Vec::with_capacity(edge_count);

            for i in 0..edge_count {
                unsafe {
                    let src = *edges_ptr.add(i * 2) as usize;
                    let dst = *edges_ptr.add(i * 2 + 1) as usize;
                    edges.push((src, dst));
                }
            }

            results.push(edges);
        }

        Ok(results)
    }
}

// Stub implementation for non-Apple Silicon platforms
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub struct MetalVisibilityPipeline;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl MetalVisibilityPipeline {
    pub fn new() -> Result<Self, String> {
        Err("Metal is only available on Apple Silicon".to_string())
    }
}

