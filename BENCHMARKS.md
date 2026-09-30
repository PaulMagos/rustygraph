# RustyGraph comparison
Machine: arm64 · 10 cores · Python 3.11.15 · load average at start 2.7

### Natural VG, single series (speed-up of RustyGraph in parentheses)
| series | n | RustyGraph 0.5 | ts2vg | pyunicorn | networkx | RustyGraph 0.4.1 |
|---|---|---|---|---|---|---|
| noise | 1,000 | **0.03 ms** | 0.18 ms (7×) | 5.67 ms (220×) | 411.57 ms (15958×) | 19.68 ms (763×) |
| noise | 10,000 | **0.38 ms** | 1.89 ms (5×) | 492.48 ms (1292×) | skipped | skipped |
| noise | 100,000 | **3.13 ms** | 21.24 ms (7×) | skipped | skipped | skipped |
| noise | 1,000,000 | **25.28 ms** | 254.02 ms (10×) | skipped | skipped | skipped |
| random walk | 1,000 | **0.07 ms** | 0.43 ms (6×) | 9.95 ms (140×) | 639.68 ms (9010×) | 21.78 ms (307×) |
| random walk | 10,000 | **1.46 ms** | 8.02 ms (6×) | 1.42 s (975×) | skipped | skipped |
| random walk | 100,000 | **10.56 ms** | 184.57 ms (17×) | skipped | skipped | skipped |
| random walk | 1,000,000 | **188.62 ms** | 4.12 s (22×) | skipped | skipped | skipped |
| concave (worst case) | 1,000 | **0.22 ms** | 1.34 ms (6×) | 4.08 ms (18×) | 339.15 ms (1534×) | 19.74 ms (89×) |
| concave (worst case) | 10,000 | **3.38 ms** | 128.07 ms (38×) | 340.30 ms (101×) | skipped | skipped |
| concave (worst case) | 100,000 | **10.09 ms** | 11.75 s (1165×) | skipped | skipped | skipped |
| concave (worst case) | 1,000,000 | **77.98 ms** | skipped | skipped | skipped | skipped |

### Horizontal VG, single series (speed-up of RustyGraph in parentheses)
| series | n | RustyGraph 0.5 | ts2vg | pyunicorn | RustyGraph 0.4.1 |
|---|---|---|---|---|---|
| noise | 1,000 | **0.01 ms** | 0.08 ms (10×) | 4.92 ms (612×) | 0.62 ms (77×) |
| noise | 10,000 | **0.08 ms** | 1.21 ms (14×) | 386.21 ms (4609×) | 16.50 ms (197×) |
| noise | 100,000 | **1.07 ms** | 12.83 ms (12×) | skipped | 827.42 ms (770×) |
| noise | 1,000,000 | **14.39 ms** | 145.43 ms (10×) | skipped | skipped |

### Vector VG (multivariate)
| series | RustyGraph 0.5 | vector-vis-graph (numba, all cores) |
|---|---|---|
| 1,000×8 noise | **0.38 ms** | 30.93 ms (82×) |
| 5,000×8 walk | **1.05 ms** | 3.05 s (2917×) |
| 10,000×36 walk | **14.00 ms** | 29.70 s (2121×) |
| 2,000×64 noise | **6.37 ms** | 204.42 ms (32×) |

### Many small windows (GNN / thesis use case)
| windows | kind | RustyGraph 0.5 batch | competitor loop |
|---|---|---|---|
| 10,000 × 23 | natural | **2.04 ms** | 135.41 ms (67×) |
| 10,000 × 64 | natural | **6.48 ms** | 188.78 ms (29×) |
| 1,000 × 23 × 36 | vector | **1.58 ms** | 139.35 ms (88×) vector-vis-graph |

### Streaming / autoregressive generation
| scenario | RustyGraph VisibilityStream | alternative |
|---|---|---|
| natural, unbounded history, 100k pushes | **121.42 ms** (1.21 µs/push) | none of the compared libraries has a streaming API |
| natural, 5k pushes, rebuild-per-step baseline | **6.91 ms** | ts2vg rebuild each step: 12.50 s (1810×) |
| natural, window 64, 20k pushes | **2.20 ms** | ts2vg rebuild window each step: 502.70 ms (229×) |
| vector d=36, window 23, 2k pushes | **1.09 ms** | vector-vis-graph rebuild each step: 292.43 ms (269×) |

### Correctness: graphs differing from exact rational arithmetic
| data | kind | RustyGraph 0.5 | ts2vg |
|---|---|---|---|
| random normal (n ≤ 60) | natural | **0/300** | 0/300 |
| random normal (n ≤ 60) | horizontal | **0/300** | 0/300 |
| integer ties (n ≤ 60) | natural | **0/300** | 0/300 |
| integer ties (n ≤ 60) | horizontal | **0/300** | 0/300 |
| real exchange-rate windows (decimal data) | natural | **0/624** | 34/624 |
| real exchange-rate windows (decimal data) | horizontal | **0/624** | 0/624 |
| random walk n = 200,000 (measured separately) | natural | **0 edges missed** | 6 truly visible edges missed (float rounding) |
RustyGraph 0.4.1 (for reference): horizontal VG wrong on 558/607 random graphs (missing edges); natural VG correct but O(n³).

### Fidelity metrics: marginal metrics (as in most TS-generation papers) vs vg_fidelity
| synthetic data | KS | Wasserstein | MMD (values) | vg_divergence | real-vs-real baseline |
|---|---|---|---|---|---|
| same process (good) | 0.057 | 0.377 | 0.0123 | **0.0003** | 0.0008 |
| time-shuffled real (bad, identical marginals) | 0.000 | 0.000 | 0.0019 | **0.0576** | 0.0008 |
| i.i.d. Gaussian, matched moments (bad) | 0.013 | 0.061 | 0.0090 | **0.0657** | 0.0008 |

### 10,000 windows 23×36, vector VG: Rust vs PyTorch engines
| engine | time | notes |
|---|---|---|
| Rust batch (CPU) | 14.99 ms | exact |
| torch exact, CPU float64 | 319.16 ms | exact, bit-identical |
| torch exact, MPS float32 | 497.10 ms | exact on device |
| torch soft, MPS float32 (forward) | 308.84 ms | differentiable |
