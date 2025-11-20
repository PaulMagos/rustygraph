# Publishing Checklist 📋

Use this checklist when publishing a new version of rustygraph to PyPI.

## Pre-Release Checklist

- [ ] **Code is ready**
  - [ ] All features implemented and tested
  - [ ] Tests passing: `cargo test`
  - [ ] Examples work: `cargo run --example basic_usage`
  - [ ] Python tests pass: `pytest python/tests/`

- [ ] **Documentation updated**
  - [ ] CHANGELOG.md updated with new version
  - [ ] README.md reflects new features
  - [ ] Doc comments accurate: `cargo doc --open`
  - [ ] Python docstrings up to date

- [ ] **Version numbers**
  - [ ] Use helper: `python3 .github/workflows/bump_version.py X.Y.Z`
  - [ ] Or manually update both:
    - [ ] `pyproject.toml` version = "X.Y.Z"
    - [ ] `Cargo.toml` version = "X.Y.Z"
  - [ ] Versions match exactly

- [ ] **Local testing**
  - [ ] Build Rust: `cargo build --release`
  - [ ] Build Python: `maturin build --release --features python-bindings`
  - [ ] Test Python: `maturin develop && python python/examples/comprehensive_example.py`

## Publishing Steps

- [ ] **Commit version bump**
  ```bash
  git add pyproject.toml Cargo.toml CHANGELOG.md
  git commit -m "Bump version to X.Y.Z"
  ```

- [ ] **Create and push tag**
  ```bash
  git tag vX.Y.Z
  git push origin main
  git push origin vX.Y.Z
  ```

- [ ] **Monitor workflow**
  - [ ] Go to GitHub → Actions tab
  - [ ] Watch `Publish Python Package on Tag` workflow
  - [ ] Check all jobs (linux, macos, windows, sdist) succeed
  - [ ] Check release job completes

## Post-Release Verification

- [ ] **Check PyPI**
  - [ ] Visit: https://pypi.org/project/rustygraph/
  - [ ] Verify version X.Y.Z appears
  - [ ] Check supported platforms listed
  - [ ] Review package metadata

- [ ] **Check GitHub Release**
  - [ ] Visit: https://github.com/YOUR_USERNAME/rustygraph/releases
  - [ ] Verify release vX.Y.Z created
  - [ ] Check wheels attached as artifacts
  - [ ] Review release notes

- [ ] **Test installation** (in fresh environment)
  ```bash
  # Create clean environment
  python3 -m venv test_env
  source test_env/bin/activate
  
  # Install from PyPI
  pip install rustygraph
  
  # Verify version
  python -c "import rustygraph; print(rustygraph.__version__)"
  
  # Test basic functionality
  python -c "
  import rustygraph as rg
  series = rg.TimeSeries([1.0, 3.0, 2.0, 4.0])
  graph = series.natural_visibility()
  print(f'Nodes: {graph.node_count()}, Edges: {graph.edge_count()}')
  "
  
  # Cleanup
  deactivate
  rm -rf test_env
  ```

- [ ] **Test on multiple platforms** (if possible)
  - [ ] Linux
  - [ ] macOS
  - [ ] Windows

- [ ] **Update documentation sites** (if applicable)
  - [ ] docs.rs will auto-update for Rust
  - [ ] Update any external documentation
  - [ ] Announce on social media / blog

## Troubleshooting

### If workflow fails:

1. **Check workflow logs** in GitHub Actions
2. **Common fixes**:
   - Invalid token → Check `PYPI_API_TOKEN` secret
   - Build fails → Test locally first
   - Version exists → Bump version number
3. **Re-run**: Delete tag, fix issue, create new tag

### To rollback:

```bash
# Remove tag locally and remotely
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z

# Note: Can't remove from PyPI, but can yank:
# Visit PyPI project settings and yank the release
```

## Quick Commands Reference

```bash
# Version bump
python3 .github/workflows/bump_version.py X.Y.Z

# Build and test locally
cargo build --release
maturin build --release --features python-bindings
maturin develop --release --features python-bindings

# Run tests
cargo test
pytest python/tests/

# Create release
git commit -am "Bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main && git push origin vX.Y.Z

# Check status
# → GitHub Actions tab
# → https://pypi.org/project/rustygraph/
```

## Version Numbering Guide

Use **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **MAJOR** (1.0.0): Breaking changes, API incompatibility
- **MINOR** (0.5.0): New features, backward compatible
- **PATCH** (0.4.1): Bug fixes, backward compatible

Examples:
- `0.4.0` → `0.4.1`: Bug fix
- `0.4.0` → `0.5.0`: New feature
- `0.9.0` → `1.0.0`: Stable release, major milestone
- `1.0.0` → `2.0.0`: Breaking API change

## Notes

- **First time publishing?** Run: `bash .github/workflows/setup-pypi.sh`
- **Need help?** Check: `.github/workflows/README.md`
- **Workflow broken?** See: `PUBLISHING_SETUP.md`

---

**Last Updated**: November 20, 2025
**Workflow Version**: 1.0

