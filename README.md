# Hb-EGF Heart Explant

## Setup python environment
Prepare environment to run the automatic analysis pipeline. After cloning the repo,
```sh
cd /environments/
conda env create -f heart-env.yml
conda activate heart-env
```

## Running analysis

```txt
usage: python main.py [-h] [-c Video class] [-i Video index] [--all_videos] [-g GPU Index] [-d Path to directory]
                    [--full_pipeline] [--tracking] [--statistics] [--view]

options:
  -h, --help            show this help message and exit
  -c Video class, --vclass Video class
                        One of `vehicle`, `hb_egf_10`, `hb_egf_50`.
  -i Video index, --vidx Video index
                        Should be 1-6.
  --all_videos          Run selected pipeline on all videos (takes priority over -c and -i).
  -g GPU Index, --gpu GPU Index
                        Choose which GPU to use.
  -d Path to directory, --path2directory Path to directory
                        Path to directory.yaml 
  --full_pipeline       Run full pipeline (takes priority over all other mode flags)
  --tracking            Run tracking
  --statistics          Compute statistics
  --view                Write viewer
```

## Examples

### Run the full analysis on every video

```sh
python main.py --full_pipeline --all_videos
```

### Run the full analysis on Vehicle_1

```sh
python main.py -c vehicle -i 1 --full_pipeline
```

### Run just tracking on Vehicle_1

```sh
python main.py -c vehicle -i 1 --tracking
```


# Napari visualization
Install a new conda environment to visualize the analysis results on Napari
```sh
cd save_path/masks/
python3 view.py
```
# Paper figures

Provided the statistics computed from the above analysis, the jupyter notebooks in `notebooks` will reproduce the appropriate paper figures.