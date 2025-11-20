#!/usr/bin/env python3
"""
Version bumper for rustygraph
Keeps version numbers in sync across pyproject.toml and Cargo.toml
"""

import re
import sys
from pathlib import Path


def update_version_in_file(file_path: Path, new_version: str) -> bool:
    """Update version in a TOML file."""
    content = file_path.read_text()

    # Pattern for version = "X.Y.Z"
    pattern = r'(version\s*=\s*")[^"]+(")'
    replacement = rf'\g<1>{new_version}\g<2>'

    new_content = re.sub(pattern, replacement, content, count=1)

    if content == new_content:
        return False

    file_path.write_text(new_content)
    return True


def get_current_version(file_path: Path) -> str | None:
    """Extract current version from a TOML file."""
    content = file_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else None


def validate_version(version: str) -> bool:
    """Check if version follows semantic versioning."""
    pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$'
    return bool(re.match(pattern, version))


def main():
    if len(sys.argv) != 2:
        print("Usage: bump_version.py VERSION")
        print("Example: bump_version.py 0.5.0")
        sys.exit(1)

    new_version = sys.argv[1]

    # Validate version format
    if not validate_version(new_version):
        print(f"❌ Invalid version format: {new_version}")
        print("   Must follow semantic versioning: MAJOR.MINOR.PATCH")
        print("   Examples: 0.5.0, 1.0.0, 1.2.3-beta")
        sys.exit(1)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    cargo_path = project_root / "Cargo.toml"

    # Check files exist
    if not pyproject_path.exists():
        print(f"❌ Not found: {pyproject_path}")
        sys.exit(1)
    if not cargo_path.exists():
        print(f"❌ Not found: {cargo_path}")
        sys.exit(1)

    # Get current versions
    pyproject_version = get_current_version(pyproject_path)
    cargo_version = get_current_version(cargo_path)

    print("📦 RustyGraph Version Bumper")
    print("=" * 40)
    print(f"Current versions:")
    print(f"  pyproject.toml: {pyproject_version}")
    print(f"  Cargo.toml:     {cargo_version}")
    print(f"\nNew version: {new_version}")
    print()

    # Confirm
    response = input("Proceed? [y/N] ").strip().lower()
    if response not in ('y', 'yes'):
        print("Aborted.")
        sys.exit(0)

    # Update files
    print("\nUpdating files...")

    if update_version_in_file(pyproject_path, new_version):
        print(f"✅ Updated {pyproject_path.name}")
    else:
        print(f"⚠️  No changes in {pyproject_path.name}")

    if update_version_in_file(cargo_path, new_version):
        print(f"✅ Updated {cargo_path.name}")
    else:
        print(f"⚠️  No changes in {cargo_path.name}")

    print("\n📋 Next steps:")
    print(f"  1. Review changes: git diff")
    print(f"  2. Commit: git commit -am 'Bump version to {new_version}'")
    print(f"  3. Tag: git tag v{new_version}")
    print(f"  4. Push: git push origin main && git push origin v{new_version}")
    print(f"\nThe GitHub Action will automatically publish to PyPI.")


if __name__ == "__main__":
    main()

