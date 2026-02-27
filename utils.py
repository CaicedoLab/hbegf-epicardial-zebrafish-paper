import argparse
import yaml
from pathlib import Path

##################################################################

def parse_args():
    parser = argparse.ArgumentParser(prog='python main.py')

    # picking subset to process
    parser.add_argument('-c', '--vclass', type=str, metavar='Video class', 
                        help='One of `vehicle`, `hb_egf_10`, `hb_egf_50`.')
    parser.add_argument('-i', '--vidx', type=int, metavar='Video index', 
                        help='Should be 1-6.')
    parser.add_argument("--all_videos", action="store_true", 
                        help="Run selected pipeline on all videos (takes priority over -c and -i).")
    
    # logistics
    parser.add_argument('-g', '--gpu', type=int, metavar='GPU Index.', 
                        default=3,
                        help='Choose which GPU to use.')
    parser.add_argument('-cpd', '--cellpose_diameter', type=int, metavar = 'Cellpose Diameter Parameter',
                        default=30, # 30 for ome tiff, 10 for the tiff extracted from movies
                        help='The diameter of cells that cellpose looks for.')
    parser.add_argument('-d', '--path2directory', type=str, metavar = 'Path to directory.',
                        default='/mnt/cephfs/mir/jcaicedo/projects/de/movies/analysis/directory/directory.yaml',
                        # default='/scr/yren/heart-explant/directory/directory.yaml',
                        help='Path to directory.yaml')
    parser.add_argument('-hcm', '--path2heartcm', type=str, metavar = 'Path to heart center of mass.',
                        default='/mnt/cephfs/mir/jcaicedo/projects/de/movies/analysis/directory/heart_centroid.csv',
                        help='Path to heart_centroid.csv')
    parser.add_argument('-fr', '--path2frames', type=str, metavar = 'Path to frame data.',
                    default='/mnt/cephfs/mir/jcaicedo/projects/de/movies/analysis/directory/frames.yaml',
                    help='Path to frames.yaml')
    
    # modes
    parser.add_argument("--full_pipeline", action="store_true", 
                        help="Run full pipeline (takes priority over all other mode flags)")
    parser.add_argument("--tracking", action="store_true", help="Run tracking")
    parser.add_argument("--statistics", action="store_true", help="Compute statistics")
    parser.add_argument("--view", action="store_true", help="Write viewer")

    return parser.parse_args() 
# end parse_args

def paths(path2directory, video_class, video_index):
    """
    Return a dictionary containing paths to all the relevant results.

    Also creates needed/missing folders.
    """
    with open(path2directory, "r") as file:
        directory = yaml.safe_load(file)
    # end with

    path2results = Path(directory[video_class]['results'][video_index])
    path2annotations = Path(directory[video_class]['annotations'][video_index])
    path2heart_circle = Path(directory[video_class]['heart-circle'][video_index])
    path2masks = path2results / 'masks'
    if not path2masks.is_dir():
        path2masks.mkdir(parents=True, exist_ok=True)
    # end if
    path2stats = path2results / 'stats'
    if not path2stats.is_dir():
        path2stats.mkdir(parents=True, exist_ok=True)
    # end if

    return {
        'frames_raw': directory[video_class]['tiff'][video_index],
        'frames_results': path2masks / f'{video_class}_{video_index}.tiff',
        'cellpose_mask': path2masks / 'cellpose_mask.tiff',
        'track_mask': path2masks / 'track_mask.tiff',
        'napari_graph': path2masks / 'napari_graph.pkl',
        'napari_tracks': path2masks / 'napari_tracks.pkl',
        'view': path2masks / 'view.py',
        'color_mask': path2masks / 'color_mask.tiff',
        'ctc_tracks': path2stats / 'ctc_tracks.csv',
        'dataframe': path2stats / 'dataframe.csv',
        'combined_dataframe': path2results.parent / 'combined_dataframe.csv',
        'heart_annotation': path2annotations,
        'heart_circle':path2heart_circle
    }
# end paths