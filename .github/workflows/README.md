# Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `rust.yml` | PR to `main` | `cargo build` + `cargo test` |
| `release.yml` | push to `main`, manual | Releases when the version prefix rule matches (below); runs its own tests first |

## Releasing

A push to `main` is a release when the **head commit subject** or the **title of the
merged PR** starts with a semantic version, optionally prefixed with `v`:

```text
v0.5.1: fix horizontal stream window        -> releases 0.5.1
0.6.0 — vector motifs                       -> releases 0.6.0
v1.0.0-rc.1 first release candidate         -> releases 1.0.0-rc.1 (GitHub pre-release)
feat: something                             -> no release
```

This works for squash, merge and rebase merges, and for direct pushes.

Steps:

1. Bump both manifests: `python .github/workflows/bump_version.py 0.5.1`
   (keeps `Cargo.toml` and `pyproject.toml` in sync).
2. Open a PR titled `v0.5.1: <summary>` (or commit with that subject) and merge it to `main`.
3. `release.yml` then:
   - checks the version equals `Cargo.toml` **and** `pyproject.toml`, and that tag
     `v0.5.1` does not exist yet (otherwise it fails with an explicit error);
   - runs `cargo test --release`;
   - builds wheels (Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64) and the sdist;
   - only if every build succeeded: uploads to PyPI (`pyrustygraph`) and publishes the crate
     to crates.io (`rustygraph`), skipping either if that version already exists there;
   - creates tag `v0.5.1` and a GitHub Release with generated notes and the built wheels.

Manual release: Actions → Release → Run workflow → enter the version.

## Secrets

| Secret | Used for |
|---|---|
| `PYPI_API_TOKEN` | PyPI upload |
| `CARGO_REGISTRY_TOKEN` | crates.io publish |

`GITHUB_TOKEN` (automatic) creates the tag and release.
