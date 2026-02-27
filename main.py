from IPython import embed # debug

import yaml
from time import time as tick

from tracking import tracking
from cell_statistics import cell_statistics
from write_view import write_view

from utils import parse_args

##################################################################

def _main(args):
    run_full_pipeline = args.full_pipeline
    run_tracking = args.tracking
    run_statistics = args.statistics
    run_view = args.view

    print(f'Processing {args.vclass}_{args.vidx}.')
    T0 = tick()
    
    if run_tracking or run_full_pipeline:
        t0 = tick()
        tracking(args)
        print(f'Tracking done in {((tick() - t0)/60):.3f} minutes')
    else:
        print("Skipping tracking.")
    # end

    if run_statistics or run_full_pipeline:
        t0 = tick()
        cell_statistics(args)
        print(f'Tracking done in {((tick() - t0)/60):.3f} minutes')
    else:
        print("Skipping statistics.")
    # end

    if run_view or run_full_pipeline:
        t0 = tick()
        write_view(args)
        print(f'Writing viewer done in {((tick() - t0)/60):.3f} minutes')
    else:
        print("Skipping write view.")
    # end


    print(f'Processed {args.vclass}_{args.vidx} in {((tick() - T0)/60):.3f} minutes')

    return None
# end _main

def main():
    args = parse_args()
    args.annotation_path = '/mnt/cephfs/mir/jcaicedo/projects/de/movies/movie_annotations/'
    all_videos = args.all_videos
    
    if args.vclass is None or args.vidx is None:
        if all_videos:
            print("PROCESSING ALL VIDEOS.")
        else:
            raise ValueError("One of --vclass or --vidx were not provided and --all_videos was not set. Aborting.")
        # end
    # end if

    if not all_videos:
        return _main(args)
    # end
 
    # if we're here, we need to process all videos
    T0 = tick()
    with open(args.path2directory, "r") as file:
        directory = yaml.safe_load(file)
    # end with
    for vclass in directory.keys():
        for vidx in directory[vclass]['raw'].keys():
            args.vclass = vclass
            args.vidx = vidx
            _main(args)
        # end for
    # end for
    print(f'All videos done in {((tick() - T0)/60):.3f} minutes')

    return None
# end main


if __name__ == '__main__':
    main()
# end