import argparse
import yaml
import numpy as np
import skimage
from pathlib import Path
from tifffile import imwrite as tiffwrite

def process_mov(path2mov):
    """
    Extract every fifth frame and crop x a bit to remove blank space.
    """
    frames = skimage.io.imread(path2mov)
    t_steps = np.arange(0, frames.shape[0], step=5)
    x_range = slice(512, 512+1024)
    return frames[t_steps, :, x_range, :]
# end process_mov

def main():
    # read path to directory and load associated yaml
    parser = argparse.ArgumentParser(prog='extract)frames')
    parser.add_argument('-d', '--path2directory', type=str, 
                        metavar='Path to directory', help='Path to directory.yaml')
    args = parser.parse_args()
    path2directory = args.path2directory
    with open(path2directory, "r") as file:
        directory = yaml.safe_load(file)
    # end with

    for vid_type in directory.values():
        raw_paths = vid_type['raw']
        tiff_paths = vid_type['tiff']

        for raw_path, tiff_path in zip(raw_paths.values(), tiff_paths.values()):
            raw_path = Path(raw_path)
            tiff_path = Path(tiff_path)

            frames = process_mov(raw_path)
            tiffwrite(tiff_path, frames, compression=True)
        # end for
    # end for

    return None
# end main

if __name__ == '__main__':
    main()
# end if
