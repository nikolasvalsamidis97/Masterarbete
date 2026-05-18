# Atmospheric Rp Exobase Beta1 Study

Checks whether beta reaches unity at the exobase radius for planet and species combinations.

## Scripts

- `R_p_exo_beta1.py`: main Rp/exobase beta=1 calculation.
- `make_R_p_exo_beta1_summary.py`: summarizes the raw Rp/exobase output tables into ranked CSV and PDF outputs.

## Run

From the repository root:

```bash
python Projects/Atmospheric_Rp_Exobase_Beta1_Study/R_p_exo_beta1.py
```

Run the summary helper after the raw table exists. Generated outputs are written under `results/`, which is ignored by Git.
