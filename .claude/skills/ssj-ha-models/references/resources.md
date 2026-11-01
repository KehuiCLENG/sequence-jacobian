# Resources: local materials and canonical references

## Local, inside the `sequence-jacobian` repo (F:\hp\Documents\GitHub\sequence-jacobian)

### Package source (the ground truth for API questions)
- `src/sequence_jacobian/` — read the source when in doubt; it is compact.
  - `examples/{rbc,krusell_smith,hank,two_asset}.py` — complete tested models.
  - `hetblocks/{hh_sim,hh_labor,hh_twoasset}.py` — household blocks.
  - `blocks/het_block.py` — the fake news algorithm as implemented.
- `tests/base/` — usage patterns with assertions (`test_workflow.py` shows the
  full two-model workflow incl. `ss_initial` and internals;
  `test_dchoice.py` shows StageBlock/DC-EGM;
  `test_estimation.py`, `test_transitional_dynamics.py`, ...).

### Official intro notebooks (`notebooks/`)
| Notebook | Teaches |
|---|---|
| `rbc.ipynb` | blocks, DAG, linear & nonlinear IRFs |
| `krusell_smith.ipynb` | HetBlocks, manual GE Jacobians, log-likelihood |
| `hank.ipynb` | adapting an off-the-shelf HetBlock; richer calibration |
| `two_asset.ipynb` | SolvedBlocks, saved Jacobians, method options |
| `labor_search.ipynb` | multiple exogenous states; shocks to transition matrices |

### Goethe 2026 workshop (`goethe-workshop-2026/`)
Markdown transcriptions of all lecture slides in `markdown/` (searchable),
PDFs in `docs/`, executable notebooks in `notebooks/`. Topic map:
- Foundations: `lecture1_sim` (SIM steady state), `lecture2_fiscal_policy`
  (HANK & fiscal, IKC), `lecture3_jacobians` (fake news, from scratch —
  `sim_fake_news.py`), `lecture4_monetary`, `lecture5_ssj_toolkit`
  (**the** toolkit tutorial), `lecture6_monetary_topics`,
  `lecture7_HANK_tutorial`.
- Estimation & stochastics: `lecture8_stochastic` (+ `estimation/routines.py`).
- Advanced: `lecture9_portfolios` (portfolio choice extension),
  `lecture10_determinacy_largescale` (quasi-Toeplitz structure, winding number
  — `winding_number.py`; truncation-free & 190-country solves),
  `lecture11_heterogeneous_firms` (HANK + het firms, MPIs),
  `lecture12_euro_hank`, `lecture13_smooth` (smooth/spline SIM methods,
  `smooth_sim/`), `lecture14_higher_order` (2nd-order sequence space),
  `lecture15_open_economy`, `lecture16_pricing_models` (Calvo/menu costs in
  sequence space), `lecture17_info_frictions` (deviations from FIRE, uses fake
  news matrix directly), `lecture18_optimal_longrun_policy`.
- `supplements/` — SIM computation & speed notebooks explaining
  `sim_steady_state(_fast).py`.

### NBER 2023 workshop (`nber-workshop-2023/`)
Lectures (PDF + some notebooks) and hands-on tutorials with/without blanks:
Tutorial 1 fiscal, Tutorial 2 monetary, Tutorial 3 estimating HANK (complete
Bayesian pipeline: `estimation/{model,routines,plots}.py`, `us_data.csv`),
Tutorial 4 discrete choice, Tutorial 5 information frictions.

### Paper PDFs at repo root
- `SSJ.pdf` — Auclert, Bardóczy, Rognlie, Straub (2021), *Using the
  Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models*,
  Econometrica 89(5). The method's source of truth: framework (§3.1), fake news
  algorithm (§3.2), DAG/GE (§4), estimation (§5), nonlinear transitions (§6).
- `sequence_slides.pdf` — the authors' slide deck for the paper.

## Online

- Toolkit repo: https://github.com/shade-econ/sequence-jacobian (issues
  tracker is useful for edge cases).
- Replication archive for the paper: https://github.com/shade-econ/ssj-replication.
- Workshops: https://github.com/shade-econ/nber-workshop-2025 (spring 2025),
  https://github.com/shade-econ/nber-workshop-2023.
- Adrien Auclert's teaching page (course materials, links):
  https://aauclert.people.stanford.edu/teaching
- Key companion papers (with SSJ-based replication code):
  - *The Intertemporal Keynesian Cross* (Auclert, Rognlie, Straub; JPE 2024)
    — iMPCs, fiscal policy: https://shade-econ.github.io/ikc/ikc.pdf
  - *Fiscal and Monetary Policy with Heterogeneous Agents* (Annual Review of
    Economics 2025) — canonical model:
    https://shade-econ.github.io/annual-review/annual_review_hank.pdf
    (repo: https://github.com/shade-econ/annual-review)
  - Determinacy in the sequence space:
    https://shade-econ.github.io/determinacy_sequence_space.pdf
  - Ungated SSJ paper: https://shade-econ.github.io/sequence_space_jacobian.pdf
- Related toolkits for cross-checking results: HARK, BASEforHANK (Julia),
  econpizza (JAX). Use them for independent verification, not as API guides —
  conventions differ.

## Citation

When producing research output with this method, cite:
Auclert, A., B. Bardóczy, M. Rognlie, and L. Straub (2021): "Using the
Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models,"
*Econometrica*, 89(5), 2375–2408. https://doi.org/10.3982/ECTA17434
— plus the model-specific papers whose blocks/calibrations you adapt.
