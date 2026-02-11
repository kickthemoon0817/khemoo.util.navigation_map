# khemoo.util.navigation_map

Isaac Sim `orthographic` & `occupancy` map generator with GUI.

See [`docs/README.md`](docs/README.md) for full documentation.

## Versioning

When releasing a new version, update **all three** of the following:

1. **`config/extension.toml`** — Update the `version` field under `[package]`
2. **`docs/VERSION.md`** — Add a new section for the release with changelog
3. **Git tag** — Tag the merge commit on `main` and push it

```bash
# After merging the release PR to main:
git tag -a vX.Y.Z -m "vX.Y.Z: short description"
git push origin vX.Y.Z
```

The version string must match across `extension.toml` and the git tag (e.g. `0.2.0` and `v0.2.0`).
