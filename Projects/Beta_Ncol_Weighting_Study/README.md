# Beta Ncol Weighting Study

Compares beta versus column density for different excitation and weighting assumptions.

## Scripts

- `effect_of_weights.py`: main calculation script that writes beta-vs-Ncol text tables.
- `plot_weights_study.py`: plotting helper for the generated weighting tables.

## Run

From the repository root:

```bash
python Projects/Beta_Ncol_Weighting_Study/effect_of_weights.py
python Projects/Beta_Ncol_Weighting_Study/plot_weights_study.py
```

Generated tables and plots are written under `results/`, which is ignored by Git.
