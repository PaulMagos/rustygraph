# Python Package Publishing Workflows

This directory contains GitHub Actions workflows for automatically building and publishing the `rustygraph` Python package to PyPI.

## Workflows

### 1. `publish-python.yml` - Publish on Main Branch Commits
This workflow automatically publishes the package to PyPI when changes are pushed to the `main` branch.

**Triggers:**
- Push to `main` branch with changes in: `src/`, `python/`, `pyproject.toml`, `Cargo.toml`, or `Cargo.lock`
- Manual workflow dispatch

**What it does:**
- Builds wheels for Linux (x86_64, aarch64)
- Builds wheels for macOS (x86_64, aarch64/Apple Silicon)
- Builds wheels for Windows (x64)
- Builds source distribution (sdist)
- Publishes all artifacts to PyPI

### 2. `publish-python-on-tag.yml` - Publish on Version Tags (RECOMMENDED)
This workflow publishes the package when you create and push a version tag.

**Triggers:**
- Push of tags matching `v*` pattern (e.g., `v0.4.0`, `v1.0.0`)
- Manual workflow dispatch

**What it does:**
- Builds wheels for Linux (x86_64, aarch64)
- Builds wheels for macOS (x86_64, aarch64/Apple Silicon)
- Builds wheels for Windows (x64)
- Builds source distribution (sdist)
- Publishes all artifacts to PyPI
- Creates a GitHub Release with the built wheels attached

## Setup Instructions

### 1. Set up PyPI API Token

You need to configure a PyPI API token as a GitHub secret:

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Scroll to "API tokens" section
3. Click "Add API token"
4. Give it a name (e.g., "rustygraph-github-actions")
5. Set scope to "Entire account" or specific to "rustygraph" project
6. Copy the token (starts with `pypi-`)
7. Go to your GitHub repository → Settings → Secrets and variables → Actions
8. Click "New repository secret"
9. Name: `PYPI_API_TOKEN`
10. Value: paste your PyPI token
11. Click "Add secret"

### 2. Alternative: Use Trusted Publishing (Recommended)

PyPI supports "trusted publishing" which doesn't require managing tokens:

1. Go to [PyPI project management](https://pypi.org/manage/project/rustygraph/settings/)
2. Scroll to "Publishing" section
3. Add a new publisher:
   - PyPI Project Name: `rustygraph`
   - Owner: `paulmagos` (your GitHub username/org)
   - Repository name: `rustygraph`
   - Workflow name: `publish-python-on-tag.yml`
   - Environment name: (leave empty)

If using trusted publishing, you can remove the `MATURIN_PYPI_TOKEN` line from the workflow.

## Usage

### Option 1: Publish on Every Commit to Main

The `publish-python.yml` workflow will automatically run when you push to main:

```bash
git add .
git commit -m "Update feature"
git push origin main
```

**Note:** This publishes immediately! Make sure your code is ready for release.

### Option 2: Publish on Tagged Releases (RECOMMENDED)

The `publish-python-on-tag.yml` workflow runs when you create a version tag:

1. Update version in `pyproject.toml`:
   ```toml
   [project]
   version = "0.5.0"
   ```

2. Update version in `Cargo.toml`:
   ```toml
   [package]
   version = "0.5.0"
   ```

3. Commit the changes:
   ```bash
   git add pyproject.toml Cargo.toml
   git commit -m "Bump version to 0.5.0"
   ```

4. Create and push a tag:
   ```bash
   git tag v0.5.0
   git push origin v0.5.0
   ```

5. The workflow will automatically:
   - Build wheels for all platforms
   - Publish to PyPI
   - Create a GitHub Release

### Manual Trigger

Both workflows can be manually triggered:

1. Go to GitHub → Actions
2. Select the workflow
3. Click "Run workflow"
4. Choose the branch
5. Click "Run workflow"

## Monitoring

### Check Workflow Status

1. Go to GitHub repository → Actions tab
2. Click on the workflow run
3. View logs for each job (linux, macos, windows, sdist, release)

### Verify Publication

After successful run:

1. Check PyPI: https://pypi.org/project/rustygraph/
2. Install and test:
   ```bash
   pip install --upgrade rustygraph
   python -c "import rustygraph; print(rustygraph.__version__)"
   ```

## Troubleshooting

### Build Failures

**Error: "Rust compiler not found"**
- The workflow uses maturin-action which includes Rust toolchain
- Check if `Cargo.toml` is valid

**Error: "Python version not supported"**
- Ensure `pyproject.toml` has correct Python version requirements
- The workflow uses Python 3.11 for building

### Publication Failures

**Error: "Invalid API token"**
- Verify `PYPI_API_TOKEN` secret is set correctly
- Make sure token hasn't expired

**Error: "File already exists"**
- The version already exists on PyPI
- Bump the version in `pyproject.toml` and `Cargo.toml`
- The workflow uses `--skip-existing` to avoid this error

**Error: "Package name already taken"**
- If first time publishing, you need to register the package name
- Publish manually first: `maturin publish`

## Best Practices

1. **Use tagged releases** (Option 2) rather than publishing on every commit
2. **Test locally** before pushing:
   ```bash
   maturin build --release
   maturin develop
   pytest python/tests/
   ```
3. **Version management**:
   - Use semantic versioning (MAJOR.MINOR.PATCH)
   - Keep `pyproject.toml` and `Cargo.toml` versions in sync
4. **Update CHANGELOG.md** before releasing
5. **Test the package** after publishing:
   ```bash
   pip install --upgrade rustygraph
   python python/examples/comprehensive_example.py
   ```

## Platform Support

The workflows build for:

- **Linux**: x86_64 (Intel/AMD), aarch64 (ARM)
- **macOS**: x86_64 (Intel), aarch64 (Apple Silicon M1/M2/M3)
- **Windows**: x64

Users on these platforms can install pre-built wheels. Others will compile from source.

## Disabling Auto-Publish

If you want to disable automatic publishing on commits:

1. Rename or delete `publish-python.yml`
2. Keep only `publish-python-on-tag.yml`
3. Publish only when you explicitly create tags

## Related Files

- `pyproject.toml` - Python package metadata and build configuration
- `Cargo.toml` - Rust crate configuration
- `src/integrations/python.rs` - Python bindings implementation
- `python/rustygraph/__init__.py` - Python package entry point

