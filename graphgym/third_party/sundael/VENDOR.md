# Vendored: sundael

This directory is a **verbatim copy** of the `sundael` PV/consumption
disaggregation package, vendored so the fleet analysis
(`scratch_sundael_transfer.py` + `notebooks/sundael_metrics.py`) is replicable
on the server without a separate checkout of the sundael repo.

| | |
|---|---|
| Upstream | `git@github.com:PabloGarcia00/sundael.git` |
| Version | `1.0.0` |
| Commit | `50e75abc8009b3dcba5fd0720daf811397d8f564` (2026-09-17) |
| License | **MPL-2.0** (see `LICENSE`) |

## License note

sundael is licensed under the **Mozilla Public License 2.0**. MPL-2.0 is
file-level copyleft: these files keep their MPL-2.0 SPDX headers and remain
under MPL-2.0 even inside this repository. Modifications to the vendored files
themselves must stay MPL-2.0; the surrounding graphgym code is unaffected.

## Local modifications

Kept to the minimum needed to run as a plain (non-pip-installed) subpackage:

- `__init__.py` — `importlib.metadata.version()` is wrapped in a try/except so
  it falls back to the pinned version string instead of raising
  `PackageNotFoundError` (the vendored copy has no distribution metadata).

Everything else is byte-for-byte upstream.

## Environment

sundael needs `pandas<3` + `numba`, which **conflict with graphgym's main
`.venv`** (pandas 3 / numpy 2.5). Build a dedicated venv from the pinned
`requirements.txt`:

```bash
cd graphgym
uv venv --python 3.13 .venv-sundael
.venv-sundael/bin/python -m pip install -r third_party/sundael/requirements.txt
```

## Running the analysis

1. **Disaggregate** (dedicated venv) — writes prediction parquets +
   membership JSON to `results/sundael_transfer/`:
   ```bash
   .venv-sundael/bin/python scratch_sundael_transfer.py
   ```
   `scratch_sundael_transfer.py` puts `third_party/` on `sys.path` and does
   `from sundael import ...`, so it resolves to this vendored copy — no
   external sundael checkout required. Data paths (gold layer, zipcode coords,
   output dir) can be overridden with the `SUNDAEL_GOLD`, `SUNDAEL_COORDS`, and
   `SUNDAEL_OUT` env vars.

2. **Score** (graphgym main `.venv`) — appends the sundael rows to the metric
   tables:
   ```bash
   .venv/bin/python notebooks/sundael_metrics.py
   ```

## Updating

Re-copy `src/sundael/{*.py,data}` + `LICENSE` from the upstream repo, re-apply
the `__init__.py` guard above, and bump the version/commit in this file and in
`__init__.py`'s fallback string.
