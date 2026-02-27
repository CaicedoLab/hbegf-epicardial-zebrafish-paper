from textwrap import dedent

from utils import paths, parse_args

##################################################################

def write_view(args):    
    video_class = args.vclass
    video_index = args.vidx
    path2directory = args.path2directory

    ps = paths(path2directory, video_class, video_index)
    path2frames_results = ps['frames_results']
    path2cellpose_mask = ps['cellpose_mask']
    path2track_mask = ps['track_mask']
    path2color_mask = ps['color_mask']
    path2napari_graph = ps['napari_graph']
    path2napari_tracks = ps['napari_tracks']
    path2annotations = ps['heart_annotation']
    path2heart_circle = ps['heart_circle']

    view_contents = dedent(f'''
    import pickle
    import napari
    from tifffile import imread as tiffread

    napari_tracks_graph = pickle.load(open('{path2napari_graph}', "rb" ) )
    napari_tracks = pickle.load(open('{path2napari_tracks}', "rb" ) )

    viewer = napari.Viewer()
    viewer.add_image(tiffread('{path2frames_results}'), name='{video_class}_{video_index}')
    viewer.add_labels(tiffread('{path2cellpose_mask}'), name='CELLPOSE', visible=False)
    viewer.add_labels(tiffread('{path2color_mask}'), name='COLORS', visible=False)
    viewer.add_labels(tiffread('{path2track_mask}'), name='TRACKASTRA', visible=False)
    viewer.add_labels(tiffread('{path2annotations}'), name='HEART MASK', visible=False)
    viewer.add_labels(tiffread('{path2heart_circle}'), name='HEART CIRCLE', visible=False)
    viewer.add_tracks(data=napari_tracks, graph=napari_tracks_graph, visible=False)

    viewer.dims.current_step = (0,)
    napari.run()
    ''')

    path2view = ps['view']
    with open(path2view, "w") as file:
        file.write(view_contents)
    # end

    return None
# end

if __name__ == '__main__':
    args = parse_args()
    print(f'Path Directory: {args.path2directory}')
    write_view(args)
# end if