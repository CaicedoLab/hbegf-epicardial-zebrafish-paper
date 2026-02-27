# Generate statistics of of tracked video

from IPython import embed # debugging

from time import time as tick
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

# classification
from sklearn.cluster import KMeans

# heart detection
import skimage
from scipy.ndimage import labeled_comprehension, center_of_mass
from tifffile import imread as tiffread
from tifffile import imwrite as tiffwrite

from utils import paths, parse_args
import os

##################################################################

def cell_statistics(args):
    ### parse arguments and load associated paths    
    video_class = args.vclass
    video_index = args.vidx
    path2directory = args.path2directory

    ps = paths(path2directory, video_class, video_index)
    path2frames_raw = ps['frames_raw']
    path2track_mask = ps['track_mask']
    path2ctc_tracks = ps['ctc_tracks']
    path2color_mask = ps['color_mask']
    path2dataframe = ps['dataframe']

    ### load video frames and tracked paths
    FRAMES_ORIGIN = tiffread(path2frames_raw)
    MASKS = tiffread(path2track_mask)
    
    # N_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, N_CHANNELS = FRAMES.shape
    
    FRAMES_ORIGIN = np.transpose(FRAMES_ORIGIN, axes=(1,2,3,0))
    FRAMES_ORIGIN = np.asarray(FRAMES_ORIGIN, dtype=np.uint8) # change from 16bit to 8bit
    FRAMES = np.zeros_like(FRAMES_ORIGIN)
    FRAMES[..., 0] = FRAMES_ORIGIN[..., 1]
    FRAMES[..., 1] = FRAMES_ORIGIN[... ,0]
    
    N_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, N_CHANNELS = FRAMES.shape
    
    LABELS = np.unique(MASKS)[1:] # 0 is background
    N_LABELS = len(LABELS) 
    
    print("CREATE DATAFRAME AND ADD LABEL/FRAME DATA")
    t0 = tick()
    DATA = pd.DataFrame()

    rows = []
    for label in LABELS:
        for frame_idx in range(N_FRAMES):
            rows.append({'label': label, 'frame': frame_idx})
        # end for
    # end for

    DATA = pd.DataFrame(rows)
    print(f'Done in {((tick() - t0)/60):.3f} minutes')


    print("CELL AREAS")
    t0 = tick()
    areas = np.zeros((N_FRAMES, N_LABELS))

    # the "input" needs to be mapped to a boolean array, otherwise func=np.sum will calculate the total
    # area weighted by the actual label value; this way it will add 1 for each appearance of a given label, i.e.
    # the total number of pixels with a given label, i.e. the area
    for f in range(N_FRAMES):
        areas[f, :] = labeled_comprehension(
            input=MASKS[f] > 0,
            labels=MASKS[f],                                   
            index=LABELS,                        
            func=np.sum,                                        
            out_dtype=np.float64,
            default=0
        )
    # end for

    DATA['area'] = areas.T.flatten()
    DATA['exists'] = DATA['area'] > 0
    print(f'Done in {((tick() - t0)/60):.3f} minutes')
    
    print("COLOR CLASSIFICATION")
    t0 = tick()
    colors = np.zeros((N_FRAMES, N_LABELS, N_CHANNELS))

    # we add up the total number of pixels in each channel for each label
    for f in range(N_FRAMES):
        for c in range(N_CHANNELS):
            colors[f, :, c] = labeled_comprehension(
                input=FRAMES[f, :, :, c].astype(np.float64), # need float64 since summing
                labels=MASKS[f],                                   
                index=LABELS,                          
                func=np.sum,                                        
                out_dtype=np.float64,
                default=0
            )
        # end for
    # end for

    # only need RED, GREEN channel
    colors = colors[:,:,0:2] 

    # normalize to color fraction, i.e. the cell is r% red and g% green
    colors_tot = colors.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        colors = colors / colors_tot # will divide by zero if cell doesn't exist, but okay
    
        # color entropy; how "uncertain" we are about the color
        # closer to 1 = better chance the cell is yellow
        # closer to 0 = better chance the cell is max(r, g)
        red = colors[:,:,0].T.flatten()
        green = colors[:,:,1].T.flatten()
        entropy = -(red*np.log(red) + green*np.log(green))/np.log(2)
    # end with

    DATA['intensity'] = colors_tot.squeeze().T.flatten()
    DATA['red'] = red
    DATA['green'] = green
    DATA['entropy'] = entropy
    
    # KMEANS CLASSIFICATION
    non_nan = np.nonzero(~np.isnan(red))
    color_data = np.vstack([red[non_nan], green[non_nan]]).T
    km = KMeans(3).fit(color_data) # 3 clusters for red, green, yellow
    # we don't know which index is which color, so we have to remap the results
    # so that [0, 1, 2] = [red, green, yellow]
    # assume that red/green are the cluster centers with maximum in that category
    # and therefore yellow is the remaining one
    ccs = km.cluster_centers_
    ccs = ccs / ccs.sum(axis=1, keepdims=True) # rescale manually in case kmean centers doesn't for some reason
    red_label = np.argmax(ccs[:,0])
    green_label = np.argmax(ccs[:,1])
    yellow_label = np.setdiff1d([0, 1, 2], [red_label, green_label]).item()
    # map kmeans labels to the corresponding colors
    _mapper = np.empty(3, dtype=str)
    _mapper[red_label] = 'r'
    _mapper[green_label] = 'g'
    _mapper[yellow_label] = 'y'
    km_labels = _mapper[km.labels_]
    
    colors_class = np.full(len(DATA), '')
    colors_class[non_nan] = km_labels
    DATA['color'] = colors_class
    
    # WRITE COLOR MASK
    MASKS_COLORS = deepcopy(MASKS)
    napari_red = 10
    napari_green = 37
    napari_yellow = 9
    napari_none = 0

    def color_to_napari(color):
        if color == '':
            return napari_none
        return {'r': napari_red, 'g': napari_green, 'y': napari_yellow}[color]
    # end color_to_napari

    # need N_LABELS + 1 since MASKS contains zeros (unlabeled)
    lookup = np.full((N_FRAMES, N_LABELS + 1), fill_value=napari_none, dtype=np.int32) 

    for row in DATA.itertuples():
        f, l, c = int(row.frame), int(row.label), row.color
        lookup[f, l] = color_to_napari(c)
    # end for

    for f in range(N_FRAMES):
        MASKS_COLORS[f] = lookup[f][MASKS[f]]
    # end for

    tiffwrite(path2color_mask, MASKS_COLORS)
    print(f'Done in {((tick() - t0)/60):.3f} minutes')
    
    
    print("PARENT CELL INFORMATION")
    t0 = tick()
    ctc_tracks = pd.read_csv(path2ctc_tracks)
    DATA['parent'] = DATA['label'].map(ctc_tracks['parent'])
    DATA['parent'] = DATA['parent'].replace(0, pd.NA)
    print(f'Done in {((tick() - t0)/60):.3f} minutes')
        
    
    print("CELL CENTROIDS")
    t0 = tick()
    centroid_rows = []

    for f in range(N_FRAMES):
        centroids = center_of_mass(
            input=MASKS[f], 
            labels=MASKS[f], 
            index=LABELS
        )

        for label, (y_center, x_center) in zip(LABELS, centroids):
            centroid_rows.append({
                'frame': f,
                'label': label,
                'x_cm': x_center,
                'y_cm': y_center
            })
        # end for
    # end for

    cm_df = pd.DataFrame(centroid_rows)
    DATA = DATA.merge(cm_df, on=['frame', 'label'], how='left')
    print(f'Done in {((tick() - t0)/60):.3f} minutes')
    

    print("CELL DISPLACEMENTS")
    t0 = tick()
    DATA['dx'] = DATA.groupby('label')['x_cm'].transform(lambda x: x.diff()).fillna(0)
    DATA['dy'] = DATA.groupby('label')['y_cm'].transform(lambda x: x.diff()).fillna(0)
    DATA['d_total'] = np.sqrt(DATA['dx']**2 + DATA['dy']**2)
    print(f'Done in {((tick() - t0)/60):.3f} minutes')

    print('HEART MASK CENTER')
    # load mask
    if video_class == 'vehicle':
        mask_class = 'V'
    elif video_class == 'hb_egf_10':
        mask_class = 'H10'
    elif video_class == 'hb_egf_50':
        mask_class = 'H50'
    heart_mask_filename = os.path.join(args.annotation_path, mask_class + f'_{video_index}-mask.tiff')
    HEART_MASK = skimage.io.imread(heart_mask_filename)
    HEART_MASK = np.asarray(HEART_MASK, dtype=np.uint8)
    
    for frame in range(HEART_MASK.shape[0]):
        y, x = center_of_mass(HEART_MASK[frame, ...])
        y, x = int(y), int(x)
        properties = skimage.measure.regionprops(HEART_MASK[frame, ...])
        radius = int(properties[0].equivalent_diameter / 2)
        
        DATA.loc[DATA.frame == frame, 'heart_y_cm'] = y
        DATA.loc[DATA.frame == frame, 'heart_x_cm'] = x
        DATA.loc[DATA.frame == frame, 'heart_radius'] = radius
    
    print("HEART POLAR COORDINATES")
    t0 = tick()
    # hh = pd.read_csv(args.path2heartcm) 
    # hh = hh[(hh['vclass'] == video_class) & (hh['vidx'] == video_index)]
    # heart_cm_y = hh['heart_y_cm'].values.item()
    # heart_cm_x = hh['heart_x_cm'].values.item()
    heart_cm_y = DATA['heart_y_cm'].values
    heart_cm_x = DATA['heart_x_cm'].values
    DATA['d_heart'] = np.sqrt((DATA['x_cm'] - heart_cm_x)**2 + (DATA['y_cm'] - heart_cm_y)**2)
    DATA['angle_heart_rad'] = np.atan2(DATA['y_cm'] - heart_cm_y, DATA['x_cm'] - heart_cm_x)
    print(f'Done in {((tick() - t0)/60):.3f} minutes')
    
    
    print('ADD FRAME TIMES')
    t0 = tick()
    with open(args.path2frames, "r") as file:
        frames = yaml.safe_load(file)['frames']
    # end with

    frame = frames[video_class][video_index]
    condition = (DATA['frame'] == 0)
    value_if_true = 0
    value_if_false = frame['dt12'] + (DATA['frame'] - 1) * frame['dt']
    DATA['time_minutes'] = np.where(condition, value_if_true, value_if_false)
    print(f'Done in {((tick() - t0)/60):.3f} minutes')
    
    ### EXPORT, FIX TYPES, REMOVE ROWS WHERE CELL DOESN'T EXIST
    DATA = DATA.loc[DATA.exists].reset_index(drop=True)
    DATA.drop('exists', axis=1, inplace=True)
    DATA = DATA.convert_dtypes()
    
    # print(path2dataframe)
    DATA.to_csv(path2dataframe, index=False)

    return None
# end cell_statistics


if __name__ == '__main__':
    args = parse_args()
    cell_statistics(args)
# end if
    