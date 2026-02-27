# image analysis
import skimage
from tifffile import imread as tiffread
import numpy as np
import pandas as pd
import torch

# cellpose
from cellpose import models

# classification
from sklearn.cluster import KMeans
from scipy.ndimage import labeled_comprehension

# utils
import argparse
import os
from time import time as tick
from copy import deepcopy
from tqdm import tqdm
from textwrap import dedent

# debug
from IPython import embed


def write_napari(path_raw, img_name, path_cellpose, path_color, file_save_path):
    if '12hr' in img_name:
        time = '12hr'
        another_time = '3hr'
        path_raw_2 = path_raw.replace('12hr', '3hr')
        path_cellpose_2 = path_cellpose.replace('12hr', '3hr')
        path_color_2 = path_color.replace('12hr', '3hr')
    elif '3hr' in img_name:
        time = '3hr'
        another_time = '12hr'
        path_raw_2 = path_raw.replace('3hr', '12hr')
        path_cellpose_2 = path_cellpose.replace('3hr', '12hr')
        path_color_2 = path_color.replace('3hr', '12hr')
        
    view_contents = dedent(f'''
    import napari
    from tifffile import imread as tiffread

    viewer = napari.Viewer()
        
    viewer.add_image(tiffread('{path_raw}'), name='{img_name.replace('.tif', '')}')
    viewer.add_labels(tiffread('{path_cellpose}'), name='CELLPOSE-{time}', visible=False)
    viewer.add_labels(tiffread('{path_color}'), name='COLORS-{time}', visible=False)

    viewer.add_image(tiffread('{path_raw_2}'), name='{img_name.replace('.tif', '').replace(time, another_time)}')
    viewer.add_labels(tiffread('{path_cellpose_2}'), name='CELLPOSE-{another_time}', visible=False)
    viewer.add_labels(tiffread('{path_color_2}'), name='COLORS-{another_time}', visible=False)

    viewer.dims.current_step = (0,)
    napari.run()
    ''')
    
    with open(file_save_path, 'w') as file:
        file.write(view_contents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Image Screens Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--path', type=str, help='Image Screen folder path', default='/hdd/jcaicedo/projects/de/Image_Screen/')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    
    args = parser.parse_args()
    PATH = args.path
    GPU = args.gpu # GPU index
    
    torch.cuda.set_device(GPU)
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    
    # initialize cellpose model
    diameter = 30
    model = models.CellposeModel(gpu=True, pretrained_model='cpsam', device=torch.device(device))
    
    # loop all subfolders and images
    subfolders = os.listdir(PATH)
    subfolders = [folder for folder in subfolders if not '.' in folder]
    
    df_list = []
    for subfolder in tqdm(subfolders):
        
        image_files = os.listdir(os.path.join(PATH, subfolder))
        image_files = [file for file in image_files if not file.startswith('.')]
        image_files = [file for file in image_files if file.endswith('.tif')]
        
        for image_file in image_files:
            im_path = os.path.join(PATH, subfolder, image_file)
            print(f'folder: {subfolder}, image: {image_file}')
            im = skimage.io.imread(im_path)
            H, W, N_CHANNELS = im.shape
            transofmed_img = np.mean(im, axis=-1) # average channels help segmenting green cells
            index = image_file.split('_')[1]
            time = image_file.split('_')[-1].split('.')[0]
    
            
            SAVE_PATH = os.path.join(PATH, subfolder, f'{subfolder}-{index}-results')
            if not os.path.exists(os.path.join(PATH, subfolder, f'{subfolder}-{index}-results')):
                os.makedirs(SAVE_PATH)
                
            # Cell segmentation
            s = tick()
            masks, flows, styles = model.eval(
                [transofmed_img],
                diameter=diameter
            )
            e = tick()
            print(f'Time used: {e - s:.3f}')
            
            MASKS = np.asarray(masks[0], dtype='uint16')
            N_LABELS = np.max(MASKS)
            
            rows = []
            for label in np.unique(MASKS)[1:]: # 0 is background
                rows.append({'label': label})
            DATA = pd.DataFrame(rows)
            
            # Save Mask in the folder
            MASK_SAVE_PATH = os.path.join(SAVE_PATH, f'cellpose-{image_file}')
            skimage.io.imsave(MASK_SAVE_PATH, MASKS)
            
            # K-means classification
            colors = np.zeros((N_LABELS, N_CHANNELS))
            for c in range(N_CHANNELS):
                colors[:,c] = labeled_comprehension(
                    input = im[..., c].astype(np.float64), # need float64 since summing
                    labels = MASKS,
                    index = np.unique(MASKS)[1:], # 0 is background,
                    func = np.sum,
                    out_dtype = np.float64,
                    default=0
                )
                
            # only need RED, GREEN channel
            colors = colors[:,0:2]
            colors_tot = colors.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                colors = colors / colors_tot
                red = colors[..., 0].T.flatten()
                green = colors[..., 1].T.flatten()
                entropy = -(red*np.log(red) + green*np.log(green))/np.log(2)
            
            non_nan = np.nonzero(~np.isnan(red))
            color_data = np.vstack([red[non_nan], green[non_nan]]).T
            km = KMeans(3).fit(color_data)

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

            ### for debugging
            # num_red_cells = sum(km_labels == 'r')
            # num_green_cells = sum(km_labels == 'g')
            # num_yellow_cells = sum(km_labels == 'y')

            # if (num_red_cells < num_green_cells) or (num_red_cells < num_yellow_cells):
            #     print(f'folder: {subfolder}, image: {image_file}')
            #     print(f'color_data means = {np.vstack([color_data[km.labels_ == i].mean(axis=0) for i in range(3)])}')
            #     print(f'ccs = {ccs}\n')
            #     print(f'rgy = {red_label}, {green_label}, {yellow_label}\n')
            #     print(f'km_labels original = {km.labels_[1:100]}')
            #     print(f'km_labels permuted = {km_labels[1:100]}')
            # # end if

            num_green_cells = sum(km_labels == 'g')
            
            DATA['intensity'] = colors_tot.squeeze().T.flatten()
            DATA['red'] = red
            DATA['green'] = green
            DATA['entropy'] = entropy
            DATA['color'] = colors_class
            
            # write color mask
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
            lookup = np.full((N_LABELS + 1), fill_value=napari_none, dtype=np.int32) 

            for row in DATA.itertuples():
                l, c = int(row.label), row.color
                lookup[l] = color_to_napari(c)
            
            # Save color mask
            MASKS_COLORS = lookup[MASKS]
            COLOR_SAVE_PATH = os.path.join(SAVE_PATH, f'color-{image_file}')
            skimage.io.imsave(COLOR_SAVE_PATH, MASKS_COLORS)
            
            # Write napari
            NAPARI_SAVE_PATH = os.path.join(SAVE_PATH, 'view.py')
            write_napari(
                path_raw=im_path,
                img_name=image_file,
                path_cellpose=MASK_SAVE_PATH,
                path_color=COLOR_SAVE_PATH,
                file_save_path=NAPARI_SAVE_PATH
            )
            
        
            info = {'image_screen_folder':subfolder, 
                    'index':index, 
                    'time':time, 
                    'num_green_cells':num_green_cells, 
                    'total_num_cells':N_LABELS}
            df_list.append(info)
            
            
            
    df_final = pd.DataFrame(df_list)
    df_final.to_csv(os.path.join(PATH, 'image_screen.csv'), index=False)
    