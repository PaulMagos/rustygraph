#!/usr/bin/env bash
# Setup script for configuring PyPI publishing
# This script guides you through setting up GitHub secrets for automated PyPI publishing

set -e

echo "🚀 RustyGraph - PyPI Publishing Setup"
echo "======================================"
echo ""
echo "This script will help you configure automated PyPI publishing."
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Get repository info
REPO_URL=$(git config --get remote.origin.url)
echo "📦 Repository: $REPO_URL"
echo ""

echo "📋 Setup Checklist:"
echo "=================="
echo ""

echo "✓ Step 1: Create PyPI API Token"
echo "  1. Go to https://pypi.org/manage/account/"
echo "  2. Scroll to 'API tokens' section"
echo "  3. Click 'Add API token'"
echo "  4. Name: 'rustygraph-github-actions'"
echo "  5. Scope: Choose 'Project: rustygraph' (or 'Entire account' if project doesn't exist yet)"
echo "  6. Click 'Add token'"
echo "  7. Copy the token (starts with 'pypi-')"
echo ""

echo "✓ Step 2: Add GitHub Secret"
echo "  1. Go to your repository on GitHub"
echo "  2. Click 'Settings' tab"
echo "  3. In left sidebar, go to 'Secrets and variables' → 'Actions'"
echo "  4. Click 'New repository secret'"
echo "  5. Name: PYPI_API_TOKEN"
echo "  6. Value: Paste your PyPI token"
echo "  7. Click 'Add secret'"
echo ""

echo "✓ Step 3: Alternative - Trusted Publishing (Recommended)"
echo "  Instead of using API tokens, you can use PyPI's trusted publishing:"
echo "  1. Go to https://pypi.org/manage/project/rustygraph/settings/"
echo "     (Note: Project must exist first - publish once manually)"
echo "  2. Scroll to 'Publishing' section"
echo "  3. Click 'Add a new publisher'"
echo "  4. Fill in:"
echo "     - PyPI Project Name: rustygraph"
echo "     - Owner: $(git config --get remote.origin.url | sed -E 's/.*[:/]([^/]+)\/[^/]+\.git/\1/')"
echo "     - Repository name: rustygraph"
echo "     - Workflow name: publish-python-on-tag.yml"
echo "     - Environment name: (leave empty)"
echo "  5. Click 'Add'"
echo ""
echo "  If using trusted publishing, you can remove the MATURIN_PYPI_TOKEN"
echo "  environment variable from the workflow file."
echo ""

echo "✓ Step 4: Choose Publishing Strategy"
echo ""
echo "  Two workflows are available:"
echo ""
echo "  A) publish-python.yml - Publish on every commit to main"
echo "     Pros: Automatic, no extra steps"
echo "     Cons: Publishes immediately, version must be bumped each time"
echo ""
echo "  B) publish-python-on-tag.yml - Publish on version tags (RECOMMENDED)"
echo "     Pros: Controlled releases, includes GitHub Release creation"
echo "     Cons: Requires manual tag creation"
echo ""
echo "  Recommended: Use option B (tag-based publishing)"
echo ""

echo "✓ Step 5: First Time Publishing"
echo ""
echo "  If this is your first time publishing 'rustygraph' to PyPI:"
echo "  1. Build the package locally:"
echo "     pip install maturin"
echo "     maturin build --release --features python-bindings"
echo ""
echo "  2. Publish manually (to claim the name):"
echo "     maturin publish --username __token__ --password YOUR_PYPI_TOKEN"
echo ""
echo "  3. After first manual publish, workflows will handle future releases"
echo ""

echo "✓ Step 6: Test the Workflow"
echo ""
echo "  For tag-based publishing:"
echo "  1. Update version in pyproject.toml and Cargo.toml"
echo "  2. Commit: git commit -am 'Bump version to X.Y.Z'"
echo "  3. Tag: git tag vX.Y.Z"
echo "  4. Push: git push origin vX.Y.Z"
echo "  5. Watch the workflow: https://github.com/YOUR_USERNAME/rustygraph/actions"
echo ""

echo "📚 Documentation"
echo "==============="
echo ""
echo "For more details, see:"
echo "  - .github/workflows/README.md"
echo "  - docs/PYTHON_BUILD_GUIDE.md"
echo ""

echo "✅ Setup guide complete!"
echo ""
echo "Next steps:"
echo "  1. Set up PyPI API token or trusted publishing (see above)"
echo "  2. Choose your publishing strategy"
echo "  3. Test with a version tag"
echo ""

