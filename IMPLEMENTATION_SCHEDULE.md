# RustyGraph Implementation Schedule

## Quick Start - Your First Hour

### Goal: Get familiar with the codebase and make your first contribution

**Start here** if you're new to the project:

1. **Read the documentation** (15 min)
   - Browse `README.md` for overview
   - Open `cargo doc --open` to see the full API
   - Look at `examples/basic_usage.rs` to understand usage

2. **Understand the architecture** (15 min)
   - Read `ARCHITECTURE.md` sections on data flow
   - Review module dependencies diagram
   - Understand the builder pattern in `visibility_graph.rs`

3. **Set up your environment** (10 min)
   ```bash
   cd /Users/paulmagos/RustroverProjects/rustygraph
   cargo build          # Should compile with warnings
   cargo test           # Will fail - tests not implemented yet
   cargo doc --open     # View documentation
   ```

4. **Make a small contribution** (20 min)
   - Pick one simple task from Week 1 below
   - Write a simple test
   - Run `cargo check` to verify

---

## Week 1: Core Algorithm - Natural Visibility (High Priority)

**Time Estimate**: 8-12 hours  
**Difficulty**: ⭐⭐⭐⚫⚫ (Medium)

### Day 1-2: Natural Visibility Algorithm (4-6 hours)

**File**: `src/algorithms/natural.rs`

**Tasks**:
1. ✅ **Study the algorithm** (1 hour)
   - Read the referenced paper (Lacasa et al. 2008)
   - Understand visibility criterion: `yk < yi + (yj - yi) * (tk - ti) / (tj - ti)`
   - Review monotonic stack data structure

2. ✅ **Implement naive version first** (1-2 hours)
   ```rust
   // Start with O(n²) version for correctness
   pub fn compute_edges_naive<T>(series: &[T]) -> Vec<(usize, usize)>
   where
       T: PartialOrd + Copy,
   {
       let mut edges = Vec::new();
       let n = series.len();
       
       for i in 0..n {
           for j in (i+1)..n {
               if is_visible(series, i, j) {
                   edges.push((i, j));
               }
           }
       }
       edges
   }
   
   fn is_visible<T>(series: &[T], i: usize, j: usize) -> bool
   where
       T: PartialOrd + Copy,
   {
       // Check if all intermediate points satisfy visibility
       // TODO: Implement this
   }
   ```

3. ✅ **Add tests for naive version** (1 hour)
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_monotonic_increasing() {
           let series = vec![1.0, 2.0, 3.0, 4.0];
           let edges = compute_edges_naive(&series);
           // Should have edges: (0,1), (1,2), (2,3), (0,2), (1,3), (0,3)
           assert_eq!(edges.len(), 6);
       }

       #[test]
       fn test_single_peak() {
           let series = vec![1.0, 3.0, 2.0];
           let edges = compute_edges_naive(&series);
           // Should have edges: (0,1), (1,2), (0,2)
           assert_eq!(edges.len(), 3);
       }
   }
   ```

4. ✅ **Optimize to O(n) with monotonic stack** (2-3 hours)
   ```rust
   pub fn compute_edges<T>(series: &[T]) -> Vec<(usize, usize)>
   where
       T: PartialOrd + Copy,
   {
       let mut edges = Vec::new();
       
       // Left visibility scan
       edges.extend(left_visibility(series));
       
       // Right visibility scan
       edges.extend(right_visibility(series));
       
       edges
   }
   
   fn left_visibility<T>(series: &[T]) -> Vec<(usize, usize)>
   where
       T: PartialOrd + Copy,
   {
       // TODO: Implement with monotonic stack
   }
   ```

**Validation**:
```bash
cargo test algorithms::natural::tests
cargo bench natural_visibility  # After benchmarks are added
```

### Day 3: Wire Up Natural Visibility (2-3 hours)

**File**: `src/visibility_graph.rs`

**Tasks**:
1. ✅ **Implement `natural_visibility()` method** (1 hour)
   ```rust
   pub fn natural_visibility(self) -> Result<VisibilityGraph<T>, GraphError> {
       // Check for empty series
       if self.series.is_empty() {
           return Err(GraphError::EmptyTimeSeries);
       }
       
       // Extract values (handle Option<T>)
       let values: Vec<T> = self.series.values
           .iter()
           .filter_map(|&v| v)
           .collect();
       
       if values.is_empty() {
           return Err(GraphError::AllValuesMissing);
       }
       
       // Call algorithm
       let edges = crate::algorithms::natural::compute_edges(&values);
       
       // Build adjacency list
       let adjacency = build_adjacency_list(self.series.len(), &edges);
       
       // Compute features if requested
       let node_features = if let Some(feature_set) = self.feature_set {
           compute_node_features(&self.series, &feature_set)?
       } else {
           vec![HashMap::new(); self.series.len()]
       };
       
       Ok(VisibilityGraph {
           node_count: self.series.len(),
           edges,
           adjacency,
           node_features,
       })
   }
   ```

2. ✅ **Implement helper functions** (1 hour)
   ```rust
   fn build_adjacency_list(node_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
       let mut adjacency = vec![Vec::new(); node_count];
       for &(src, dst) in edges {
           adjacency[src].push(dst);
           adjacency[dst].push(src); // Undirected
       }
       adjacency
   }
   ```

3. ✅ **Add integration tests** (1 hour)
   - Create `tests/integration_test.rs`
   - Test full pipeline: series → graph → analysis

**Validation**:
```bash
cargo test visibility_graph::tests
cargo run --example basic_usage  # Should work now!
```

### Day 4: Code Review & Documentation (1-2 hours)

**Tasks**:
1. Run `cargo clippy` and fix warnings
2. Add doc comments with examples
3. Verify all tests pass
4. Update CHANGELOG.md

---

## Week 2: Horizontal Visibility & Missing Data (High Priority)

**Time Estimate**: 8-10 hours  
**Difficulty**: ⭐⭐⚫⚫⚫ (Easy-Medium)

### Day 5-6: Horizontal Visibility (3-4 hours)

**File**: `src/algorithms/horizontal.rs`

**Why second**: Simpler algorithm, can reuse patterns from natural visibility

**Tasks**:
1. ✅ **Implement algorithm** (2 hours)
   ```rust
   pub fn compute_edges<T>(series: &[T]) -> Vec<(usize, usize)>
   where
       T: PartialOrd + Copy,
   {
       let mut edges = Vec::new();
       let n = series.len();
       
       for i in 0..n {
           // Scan right
           let mut j = i + 1;
           while j < n {
               if is_horizontally_visible(series, i, j) {
                   edges.push((i, j));
                   j += 1;
               } else {
                   break; // Higher point blocks further visibility
               }
           }
       }
       
       edges
   }
   
   fn is_horizontally_visible<T>(series: &[T], i: usize, j: usize) -> bool
   where
       T: PartialOrd + Copy,
   {
       let min_height = if series[i] < series[j] { series[i] } else { series[j] };
       
       for k in (i+1)..j {
           if series[k] >= min_height {
               return false;
           }
       }
       true
   }
   ```

2. ✅ **Add tests** (1 hour)
3. ✅ **Wire up in graph builder** (1 hour)

### Day 7-8: Missing Data Handling (4-5 hours)

**Files**: 
- `src/features/missing_data.rs`
- `src/time_series.rs`

**Tasks**:
1. ✅ **Implement LinearInterpolation** (1 hour)
   ```rust
   impl MissingDataStrategy {
       pub fn impute<T>(&self, series: &[Option<T>], index: usize) -> Option<T>
       where
           T: Copy + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T>,
       {
           match self {
               MissingDataStrategy::LinearInterpolation => {
                   // Find prev valid
                   let prev_idx = (0..index).rev().find(|&i| series[i].is_some())?;
                   let prev_val = series[prev_idx]?;
                   
                   // Find next valid
                   let next_idx = (index+1..series.len()).find(|&i| series[i].is_some())?;
                   let next_val = series[next_idx]?;
                   
                   // Linear interpolation
                   Some((prev_val + next_val) / 2.0)
               }
               // Other strategies...
               _ => todo!()
           }
       }
   }
   ```

2. ✅ **Implement ForwardFill** (30 min)
3. ✅ **Implement BackwardFill** (30 min)
4. ✅ **Implement handle_missing() in TimeSeries** (1 hour)
   ```rust
   impl<T> TimeSeries<T> {
       pub fn handle_missing(&self, strategy: MissingDataStrategy) 
           -> Result<Self, ImputationError>
       where
           T: Copy + /* trait bounds */,
       {
           let mut new_values = self.values.clone();
           
           for i in 0..new_values.len() {
               if new_values[i].is_none() {
                   new_values[i] = strategy.impute(&self.values, i)
                       .ok_or(ImputationError::AllStrategiesFailed { index: i })?;
               }
           }
           
           Ok(TimeSeries {
               timestamps: self.timestamps.clone(),
               values: new_values,
           })
       }
   }
   ```

5. ✅ **Add tests** (1-2 hours)

**Validation**:
```bash
cargo test missing_data
cargo run --example with_features  # Should work partially
```

---

## Week 3: Basic Features (Medium Priority)

**Time Estimate**: 6-8 hours  
**Difficulty**: ⭐⭐⚫⚫⚫ (Easy-Medium)

### Day 9-11: Implement Core Features (6-8 hours)

**File**: `src/features/builtin.rs`

**Order of implementation** (easiest first):

1. ✅ **DeltaForward** (30 min)
   ```rust
   impl<T> Feature<T> for DeltaForwardFeature
   where
       T: Copy + std::ops::Sub<Output = T>,
   {
       fn compute(
           &self,
           series: &[Option<T>],
           index: usize,
           _handler: &dyn MissingDataHandler<T>,
       ) -> Option<T> {
           if index + 1 >= series.len() {
               return None;
           }
           let curr = series[index]?;
           let next = series[index + 1]?;
           Some(next - curr)
       }
       
       fn name(&self) -> &str {
           "delta_forward"
       }
       
       fn requires_neighbors(&self) -> bool {
           true
       }
   }
   ```

2. ✅ **DeltaBackward** (30 min) - Similar pattern
3. ✅ **LocalSlope** (1 hour) - Needs time spacing
4. ✅ **IsLocalMax** (1 hour) - Comparison logic
5. ✅ **IsLocalMin** (1 hour) - Comparison logic
6. ✅ **LocalMean** (1-2 hours) - Window operations
7. ✅ **LocalVariance** (1-2 hours) - More complex math

### Day 12: Feature Integration (2 hours)

**File**: `src/features/mod.rs`

**Tasks**:
1. ✅ **Implement `add_builtin()`** (1 hour)
   ```rust
   impl<T> FeatureSet<T> {
       pub fn add_builtin(mut self, feature: BuiltinFeature) -> Self {
           let feature_impl: Box<dyn Feature<T>> = match feature {
               BuiltinFeature::DeltaForward => Box::new(builtin::DeltaForwardFeature),
               BuiltinFeature::DeltaBackward => Box::new(builtin::DeltaBackwardFeature),
               // ... other features
               _ => todo!("Not implemented yet"),
           };
           self.features.push(feature_impl);
           self
       }
   }
   ```

2. ✅ **Implement `add_function()`** (1 hour)
   ```rust
   struct FunctionFeature<T, F>
   where
       F: Fn(&[Option<T>], usize) -> Option<T> + Send + Sync,
   {
       name: String,
       func: F,
       _phantom: std::marker::PhantomData<T>,
   }
   
   impl<T, F> Feature<T> for FunctionFeature<T, F>
   where
       F: Fn(&[Option<T>], usize) -> Option<T> + Send + Sync,
   {
       fn compute(&self, series: &[Option<T>], index: usize, _: &dyn MissingDataHandler<T>) -> Option<T> {
           (self.func)(series, index)
       }
       
       fn name(&self) -> &str {
           &self.name
       }
   }
   ```

**Validation**:
```bash
cargo test features
cargo run --example with_features  # Should fully work!
```

---

## Week 4: Testing & Polishing (High Priority)

**Time Estimate**: 8-10 hours  
**Difficulty**: ⭐⭐⭐⚫⚫ (Medium)

### Day 13-15: Comprehensive Testing (6-8 hours)

**Create**: `tests/` directory with integration tests

**Tasks**:
1. ✅ **Algorithm correctness tests** (2 hours)
   - Test against known graph structures
   - Validate mathematical properties
   - Edge cases (empty, single point, duplicates)

2. ✅ **Feature computation tests** (2 hours)
   - Test each feature independently
   - Test with missing data
   - Test boundary behavior

3. ✅ **Integration tests** (2 hours)
   - Full pipeline: raw data → clean → graph → features
   - Test error handling
   - Test builder pattern

4. ✅ **Property-based tests** (2 hours) - Optional but recommended
   ```bash
   # Add to Cargo.toml
   [dev-dependencies]
   proptest = "1.0"
   ```

### Day 16: Documentation & Cleanup (2 hours)

**Tasks**:
1. ✅ Run `cargo clippy` and fix all warnings
2. ✅ Run `cargo fmt` to format code
3. ✅ Add doc examples that compile
4. ✅ Update CHANGELOG.md with v0.2.0 notes
5. ✅ Update TODO.md to mark completed items

**Final validation**:
```bash
cargo build --release
cargo test
cargo test --doc
cargo clippy
cargo doc --open
cargo run --example basic_usage
cargo run --example with_features
```

---

## Priority Matrix

```
┌─────────────────────────────────────────────────────────────┐
│  PRIORITY MATRIX                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HIGH PRIORITY (Do First)                                   │
│  ├─ Week 1: Natural Visibility Algorithm       [8-12h] ⭐⭐⭐│
│  ├─ Week 2: Horizontal Visibility              [3-4h]  ⭐⭐ │
│  ├─ Week 2: Missing Data (Basic)               [4-5h]  ⭐⭐ │
│  └─ Week 4: Testing                            [6-8h]  ⭐⭐⭐│
│                                                             │
│  MEDIUM PRIORITY (Do Second)                                │
│  ├─ Week 3: Basic Features (3-5 features)      [6-8h]  ⭐⭐ │
│  └─ Week 3: Feature Integration                [2h]    ⭐   │
│                                                             │
│  LOW PRIORITY (Do Later)                                    │
│  ├─ Remaining Features                         [6-8h]  ⭐   │
│  ├─ Advanced Missing Data Strategies           [4-5h]  ⭐   │
│  └─ Graph Export Methods                       [2-3h]  ⭐   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Recommended Implementation Order

### 🚀 Sprint 1: Minimum Viable Product (MVP) - 2 weeks

**Goal**: Working visibility graphs with basic functionality

1. ✅ **Natural visibility algorithm** → `algorithms/natural.rs`
2. ✅ **Wire up graph construction** → `visibility_graph.rs`
3. ✅ **Horizontal visibility algorithm** → `algorithms/horizontal.rs`
4. ✅ **Basic missing data (LinearInterpolation, ForwardFill)** → `missing_data.rs`, `time_series.rs`
5. ✅ **3 basic features (DeltaForward, DeltaBackward, LocalSlope)** → `builtin.rs`
6. ✅ **Integration tests** → `tests/`

**Deliverable**: v0.2.0 - Working graphs with 3 features

### 🎯 Sprint 2: Feature Complete - 1 week

**Goal**: All documented features implemented

7. ✅ **Remaining features** → `builtin.rs`
8. ✅ **Remaining missing data strategies** → `missing_data.rs`
9. ✅ **Custom feature support** → `features/mod.rs`
10. ✅ **Comprehensive tests** → `tests/`

**Deliverable**: v0.3.0 - Full feature set

### ⚡ Sprint 3: Polish & Performance - 1 week

**Goal**: Production-ready quality

11. ✅ **Benchmarks** → `benches/`
12. ✅ **Documentation improvements**
13. ✅ **Performance optimization**
14. ✅ **Error handling refinement**

**Deliverable**: v0.4.0 - Production ready

---

## Daily Workflow

### Morning Routine (15 min)
```bash
cd /Users/paulmagos/RustroverProjects/rustygraph
git pull
cargo build
cargo test
# Review TODO.md and pick today's task
```

### Development Cycle (Pomodoro style)
1. **Code** (25 min) - Implement one function
2. **Test** (5 min) - Write test for that function
3. **Break** (5 min)
4. **Repeat** 4x
5. **Long break** (15 min) - Review, commit, push

### Evening Routine (10 min)
```bash
cargo clippy
cargo fmt
cargo test
git add -A
git commit -m "Implement [feature]: [description]"
git push
# Update TODO.md progress
```

---

## Dependency Graph

```
                    ┌──────────────────┐
                    │ Natural Visibility│
                    │   (Week 1)        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Graph Builder    │
                    │   (Week 1)        │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
       ┌────────────────┐       ┌────────────────┐
       │   Horizontal   │       │ Missing Data   │
       │  Visibility    │       │   (Basic)      │
       │   (Week 2)     │       │   (Week 2)     │
       └────────────────┘       └────────┬───────┘
                                         │
                                         ▼
                                ┌────────────────┐
                                │    Features    │
                                │   (Week 3)     │
                                └────────────────┘
                                         │
                                         ▼
                                ┌────────────────┐
                                │    Testing     │
                                │   (Week 4)     │
                                └────────────────┘
```

---

## Where to Start RIGHT NOW

### Option A: Jump In (Recommended for experienced Rust devs)

1. **Open**: `src/algorithms/natural.rs`
2. **Replace the `compute_edges()` function** with naive implementation
3. **Add 3 simple tests**
4. **Run**: `cargo test algorithms::natural::tests`
5. **Iterate** until tests pass

### Option B: Learn First (Recommended for learning Rust)

1. **Read**: Lacasa et al. 2008 paper (link in algorithms/natural.rs)
2. **Study**: `ARCHITECTURE.md` data flow section
3. **Experiment**: Write a standalone Rust program that computes visibility
4. **Integrate**: Copy working code into the library

### Option C: Start Small (Recommended for beginners)

1. **Pick**: DeltaForward feature (simplest)
2. **Implement**: `src/features/builtin.rs::DeltaForwardFeature`
3. **Test**: Write one test
4. **Repeat**: Do DeltaBackward next

---

## Resources & References

### Academic Papers
- **Natural Visibility**: Lacasa et al. (2008) PNAS - "From time series to complex networks"
- **Horizontal Visibility**: Luque et al. (2009) Physical Review E

### Code Examples
- **Monotonic Stack**: [Wikipedia](https://en.wikipedia.org/wiki/Stack_(abstract_data_type)#Monotonic_stack)
- **Builder Pattern**: Already implemented in `visibility_graph.rs`

### Rust Resources
- **Trait Bounds**: [Rust Book Chapter 10](https://doc.rust-lang.org/book/ch10-00-generics.html)
- **Error Handling**: [Rust Book Chapter 9](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- **Testing**: [Rust Book Chapter 11](https://doc.rust-lang.org/book/ch11-00-testing.html)

---

## Progress Tracking

Create a file `PROGRESS.md` to track your implementation:

```markdown
# Implementation Progress

## Week 1
- [x] Read documentation
- [ ] Natural visibility algorithm
  - [ ] Naive implementation
  - [ ] Tests
  - [ ] O(n) optimization
- [ ] Graph builder integration
- [ ] Integration tests

## Week 2
...
```

---

## Getting Help

### When Stuck:
1. **Check**: `ARCHITECTURE.md` for design decisions
2. **Review**: Similar code in other modules
3. **Test**: Write a failing test to clarify requirements
4. **Simplify**: Break down into smaller functions
5. **Ask**: Open a GitHub issue with specific question

### Common Pitfalls:
- **Generic constraints**: Start concrete (f64), generify later
- **Lifetimes**: Use `'_` or let compiler infer
- **Option<T>**: Use `?` operator liberally
- **Tests**: Write them FIRST before implementation

---

**Ready to start? Your first task is in** `src/algorithms/natural.rs` **line 67** 🚀

**Estimated time to working MVP**: 20-30 hours over 2-3 weeks  
**Current completion**: ~15% (API design done)  
**Next milestone**: v0.2.0 with working graphs

Last Updated: 2025-11-06

