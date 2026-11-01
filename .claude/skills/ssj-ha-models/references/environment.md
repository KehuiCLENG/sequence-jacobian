# Environment setup and verification

The SSJ toolkit is pure Python on top of NumPy/SciPy/Numba. No compilers, no
GPU, no exotic dependencies. Numba JIT-compiles a handful of inner loops on
first use.

## Requirements

- Python ≥ 3.8 (3.11 or 3.12 recommended; verified working on 3.12.7).
- Core: `numpy`, `scipy`, `numba` (these are the declared dependencies of
  `sequence-jacobian`).
- Practical additions: `matplotlib`, `pandas`, `pytest`, `ipykernel`/`jupyter`
  (workshop notebooks), `pdfplumber`/`pypdf` only if reading paper PDFs.
- Optional: `graphviz` (Python package **and** the Graphviz binaries) for
  `sj.drawdag`. Everything else works without it.

## Setup A — inside the `sequence-jacobian` repo (preferred for development)

A venv may already exist at the repo root as `.venv/` with the package
installed in editable mode. If not, create it:

```powershell
# Windows (py launcher)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install numpy scipy numba matplotlib pandas pytest ipykernel
.venv\Scripts\python.exe -m pip install -e .
```

```bash
# macOS/Linux
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy scipy numba matplotlib pandas pytest ipykernel
.venv/bin/python -m pip install -e .
```

Editable install (`-e .`) means source edits under `src/sequence_jacobian/`
take effect without reinstalling — important when debugging the toolkit itself.

## Setup B — standalone project

```bash
python -m venv .venv && .venv/bin/pip install sequence-jacobian matplotlib pandas pytest
```

PyPI ships version 1.0.0, which matches the repo's master branch API used
throughout this skill (post-1.0 API: `sj.simple`, `sj.het`, `sj.solved`,
`sj.combine`/`create_model`, `solve_*` methods). Pre-1.0 tutorials found
online use a different API — do not mix them.

## Verification (mandatory before real work)

Run this skill's `scripts/check_env.py` with the venv's python:

```powershell
.venv\Scripts\python.exe <skill>/scripts/check_env.py
```

It prints package versions, then solves the Krusell–Smith model (steady state →
GE Jacobian → linear IRF → nonlinear IRF) and the one-asset HANK model, with
assertions on market clearing and linear/nonlinear agreement. Expected runtime:
~10–30 s (mostly numba compilation). It must end with `ALL CHECKS PASSED`.

For a deeper certification (once per machine), run the package test suite:

```powershell
.venv\Scripts\python.exe -m pytest tests/base -q      # from the repo root; 40 tests, ~30 s
```

All 40 must pass. Deprecation warnings about `sj.agrid`/`sj.markov_rouwenhorst`
are expected and harmless (but write new code with `sj.grids.*`).

## Numba notes

- First call into a numba-compiled routine (e.g. the first steady state or
  Jacobian) pays a compilation cost of seconds; subsequent calls are fast.
  Do not misread the first call as "slow code".
- If numba fails to import or errors at JIT time, the usual cause is a
  numpy/numba version mismatch. Fix by installing a matched pair, e.g.
  `pip install -U numba` (pip will constrain numpy appropriately —
  numba 0.66 pairs with numpy ≤ 2.4.x).
- Numba caches per-process; long-running notebook kernels keep the warm state.

## Reproducibility

Record with any set of results:
```python
import sys, numpy, scipy, numba, sequence_jacobian
print(sys.version, numpy.__version__, scipy.__version__, numba.__version__)
```
plus `pip freeze > requirements-lock.txt`, the git commit of the model code,
and the seed of any simulation (`np.random.seed(...)` or a `Generator`).
