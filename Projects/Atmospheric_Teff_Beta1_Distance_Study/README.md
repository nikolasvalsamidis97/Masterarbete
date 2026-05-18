# Atmospheric Teff Beta1 Distance Study

Computes beta=1 heights over stellar effective temperature, orbital distance, planets, and species.

## Scripts

- `Teff_planet_beta1_dist.py`: main grid calculation.
- `plot_by_txt_file.py`: generic plotter for saved text tables.
- `plot_r_at_beta1.py`: batch plotter for beta=1 radius tables.
- `make_r_beta1_summary.py`: summary tables and plots for beta=1 radius outputs.
- `make_four_example_plot.py`: four-example planet figure helper.
- `make_four_strongest_responders_plot.py`: strongest-responder figure helper.
- `plot_Tatm_vs_beta1.py`: atmosphere-temperature plot helper.
- `plot_gravity_study.py`: surface-gravity plot helper.

## Run

From the repository root:

```bash
python Projects/Atmospheric_Teff_Beta1_Distance_Study/Teff_planet_beta1_dist.py
```

Run plotting and summary helpers after the relevant text tables exist. Generated outputs are written under `results/`, which is ignored by Git.
