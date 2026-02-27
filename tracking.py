import shutil
from time import time as tick
import pickle

import numpy as np
import torch

from tifffile import imread as tiffread
from tifffile import imwrite as tiffwrite

# cellpose
from cellpose import models

# trackastra
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks

from utils import paths, parse_args

##################################################################

    
def tracking(args):   
    ### parse arguments and load associated paths
    gpu = args.gpu
    diameter = args.cellpose_diameter
    video_class = args.vclass
    video_index = args.vidx
    path2directory = args.path2directory

    ps = paths(path2directory, video_class, video_index)
    path2frames_raw = ps['frames_raw']
    path2frames_results = ps['frames_results']
    path2cellpose_mask = ps['cellpose_mask']
    path2track_mask = ps['track_mask']
    path2napari_graph = ps['napari_graph']
    path2napari_tracks = ps['napari_tracks']
    path2ctc_tracks = ps['ctc_tracks']
    
    # load frames and make a copy in the results folder
    FRAMES_ORIGIN = tiffread(path2frames_raw) # TYXC (ome CTYX)
    # print(FRAMES_ORIGIN.shape)
    
    # reshape ome data
    FRAMES_ORIGIN = np.transpose(FRAMES_ORIGIN, axes=(1,2,3,0))
    FRAMES_ORIGIN = np.asarray(FRAMES_ORIGIN, dtype=np.uint8) # change from 16bit to 8bit
    FRAMES = np.zeros_like(FRAMES_ORIGIN)
    FRAMES[..., 0] = FRAMES_ORIGIN[..., 1]
    FRAMES[..., 1] = FRAMES_ORIGIN[... ,0]
    # print(FRAMES.shape)
    
    shutil.copyfile(path2frames_raw, path2frames_results)
    
    ### Cellpose Segmentation
    t0 = tick()

    # FRAMES = FRAMES ** (1/3)

    CELLPOSE_DIAMETER = diameter
    CELLPOSE_CHANNEL_AXIS = -1
    CELLPOSE_FLOW_THRESHOLD = 1.5
    # CELLPOSE_CELLPROB_THRESHOLD = -6 # default 0.0
    CELLPOSE_CELLPROB_THRESHOLD = 1
    
    torch.cuda.set_device(gpu)
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = models.CellposeModel(gpu=True, pretrained_model='cpsam', device=torch.device(device))
    masks, flows, styles = model.eval(
        [FRAMES[i, ...] for i in range(FRAMES.shape[0])], # cellpose needs a list of 2D images >:(, exclude blue channel!
            diameter=CELLPOSE_DIAMETER,
            channel_axis=CELLPOSE_CHANNEL_AXIS,
            # flow_threshold=CELLPOSE_FLOW_THRESHOLD,
            cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD
    )

    MASKS = np.stack(masks)
    tiffwrite(path2cellpose_mask, MASKS)
    
    print(f'Cellpose done in {((tick() - t0)/60):.3f} minutes')
    
    ### Trackastra Tracking
    t0 = tick()

    model = Trackastra.from_pretrained("general_2d", device=device)
    
    FRAMES_TRACK = np.sum(FRAMES, axis=-1)
    t0 = tick()
    track_graph = model.track(FRAMES_TRACK, MASKS, mode="greedy") 
    ctc_tracks, masks_tracked = graph_to_ctc(
      track_graph,
      MASKS,
      outdir=None)
    
    napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
    tiffwrite(path2track_mask, masks_tracked)
    ctc_tracks.to_csv(path2ctc_tracks , index=False)
    pickle.dump(napari_tracks, open(path2napari_tracks, "wb" ))
    pickle.dump(napari_tracks_graph, open(path2napari_graph, "wb" ))
    
    print(f'Trackastra done in {((tick() - t0)/60):.3f} minutes')

    return None
# end tracking


if __name__ == '__main__':
    args = parse_args()
    tracking(args)
# end if