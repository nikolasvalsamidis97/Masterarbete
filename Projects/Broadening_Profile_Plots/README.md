# Broadening Profile Plots

Creates reference plots for line profiles, optical depth, and rotational broadening.

## Scripts

- `BroadeningProfilesPlots.py`: Voigt/profile comparison figure.
- `OpticalDepth_Ncol.py`: optical-depth and column-density broadening figure.
- `RotationalBroadeningPlots.py`: rotational broadening figure.
- `animate_saturation_effect.py`: side-by-side GIF animation of Na I line saturation for two Doppler widths as column density increases.

## Run

From the repository root, run the figure you need, for example:

```bash
python Projects/Broadening_Profile_Plots/BroadeningProfilesPlots.py
```

To create the saturation animation:

```bash
python Projects/Broadening_Profile_Plots/animate_saturation_effect.py
```

Generated plots are written under `results/`, which is ignored by Git.
