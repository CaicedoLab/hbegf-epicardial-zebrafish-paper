# Data

Inputs needed to regenerate the paper figures with the notebooks in `notebooks/`.
The two `*_combined_dataframe.csv` files are too large for GitHub (>100 MB), so they
are gitignored and have to be obtained separately.

## Per-time-point summaries (outputs)

`movies_times.csv` (written by `notebooks/heatmaps.ipynb`) and `pulse_times_full.csv`
(written by `notebooks/pulse.ipynb`) have one row per sample (`vclass`, `vidx`) and frame:

- `time (h)`: time of the frame
- `n_green_in_frame`, `n_red_in_frame`: number of green / red cells in the frame
- `delta_n_green_in_frame`, `delta_n_red_in_frame`: change from the sample's first frame
- `frac_green`: g / (g + r); `delta_frac_green`: its change from the first frame

`pulse_times.csv` is an input (frame times), not one of these outputs.
