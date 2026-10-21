# Releasing

Roomer is distributed through [HACS](https://hacs.xyz). HACS offers users the versions published as **GitHub releases**, so a release is not finished until the GitHub release exists. Follow these steps in order.

## 1. Pick the version

Roomer uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`.

- **Patch** (`0.2.0` → `0.2.1`): bug fixes only.
- **Minor** (`0.2.0` → `0.3.0`): new features, or anything users must act on while the project is still pre-1.0.
- **Major**: breaking changes once the project reaches 1.0.

Breaking changes to config entries, entity IDs, services or events should always be called out in the changelog, whatever the version number.

## 2. Bump the version in both files

The version lives in two places and they must match:

- [`custom_components/roomer/manifest.json`](https://github.com/owens-ben/roomer/blob/main/custom_components/roomer/manifest.json): `"version"`. Home Assistant and HACS read this one.
- [`pyproject.toml`](https://github.com/owens-ben/roomer/blob/main/pyproject.toml): `version` under `[project]`.

```json
  "version": "0.3.0"
```

```toml
[project]
name = "roomer"
version = "0.3.0"
```

Check that they agree:

```bash
grep '"version"' custom_components/roomer/manifest.json
grep '^version' pyproject.toml
```

## 3. Update `CHANGELOG.md`

[`CHANGELOG.md`](https://github.com/owens-ben/roomer/blob/main/CHANGELOG.md) follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Throughout development, changes collect under `## [Unreleased]`. At release time:

1. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and add a fresh, empty `## [Unreleased]` above it.
2. Keep entries under the standard headings: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`. Prefix breaking changes with `**BREAKING:**` and say what users have to do, as the existing entries do.
3. Update the compare links at the bottom of the file:

```markdown
## [Unreleased]

## [0.3.0] - 2026-10-21

### Added
- ...

[Unreleased]: https://github.com/owens-ben/roomer/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/owens-ben/roomer/compare/v0.2.0...v0.3.0
```

## 4. Check CI on `main`

Merge the version bump and changelog through a pull request. Before you tag, make sure these workflows pass on `main`:

| Workflow | File | What it checks |
| --- | --- | --- |
| Tests | [`.github/workflows/test.yml`](https://github.com/owens-ben/roomer/blob/main/.github/workflows/test.yml) | `pytest` on Python 3.11, 3.12 and 3.13, plus coverage. Ruff runs too but cannot fail the build yet (`continue-on-error`). |
| Validate | [`.github/workflows/validate.yml`](https://github.com/owens-ben/roomer/blob/main/.github/workflows/validate.yml) | **hassfest** (Home Assistant's manifest and integration structure checks) and the **HACS action** (`category: integration`, with the `brands` check ignored until Roomer is submitted to `home-assistant/brands`). It also runs nightly and can be started by hand from the Actions tab (`workflow_dispatch`). |

Treat a failing Validate run as a release blocker: it usually means HACS or HA will reject or mis-read the integration. Common causes are a malformed `manifest.json`, keys out of order or a missing `version`.

## 5. Tag the release

Tags use the form `vX.Y.Z`, matching the manifest version with a `v` prefix:

```bash
git checkout main
git pull
git tag -a v0.3.0 -m "Roomer 0.3.0"
git push origin v0.3.0
```

## 6. Create the GitHub release

A tag on its own is not enough for HACS: it reads **releases**. Create one from the tag, either on GitHub (**Releases → Draft a new release**, choose the tag) or with the GitHub CLI:

```bash
gh release create v0.3.0 --title "v0.3.0" --notes-file release-notes.md
```

For the notes, paste the version's section from `CHANGELOG.md`. Mark the release as a **pre-release** if it is not ready for everyone. HACS only shows pre-releases to users who have opted in to beta versions for Roomer.

After publishing, HACS users see the update within HACS's normal refresh cycle. To check it end to end, open HACS in a test HA instance, find Roomer, and confirm the new version is offered.

## Documentation

This site is built with MkDocs Material and deploys itself. [`.github/workflows/docs.yml`](https://github.com/owens-ben/roomer/blob/main/.github/workflows/docs.yml) runs on every push to `main` that touches `docs/**`, `mkdocs.yml` or the workflow file. It installs `docs/requirements.txt`, runs `mkdocs build --strict` and publishes the result to GitHub Pages. Pull requests that touch the docs get the build (so broken links or nav entries fail the PR) without the deploy. You can also start a deploy by hand with `workflow_dispatch`.

Docs are not versioned: the site always reflects `main`. If a release changes behaviour, update the docs in the same PR as the code.

To preview locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Release checklist

- Version bumped in `custom_components/roomer/manifest.json` **and** `pyproject.toml`, and they match
- `CHANGELOG.md`: `[Unreleased]` moved to `[X.Y.Z] - date`, compare links updated
- Tests and Validate (hassfest and HACS) green on `main`
- Tag `vX.Y.Z` pushed
- GitHub release created from the tag, with the changelog section as notes
- Docs updated for any behaviour change
