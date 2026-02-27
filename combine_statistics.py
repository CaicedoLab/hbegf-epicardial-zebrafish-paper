from IPython import embed # debug

import yaml
from time import time as tick

import pandas as pd

from utils import parse_args, paths

##################################################################

def combine_statistics():
    args = parse_args()

    T0 = tick()
    with open(args.path2directory, "r") as file:
        directory = yaml.safe_load(file)
    # end with

    rows = []
    for vclass in directory.keys():
        for vidx in directory[vclass]['raw'].keys():
            args.vclass = vclass
            args.vidx = vidx
            _paths = paths(args.path2directory, vclass, vidx)

            _df = pd.read_csv(_paths['dataframe'])
            _df['vclass'] = vclass
            _df['vidx'] = vidx
            rows.append(_df)
        # end for
    # end for
    df = pd.concat(rows)
    df.to_csv(_paths['combined_dataframe'], index=False)

    print(f'Statistics combined in {((tick() - T0)/60):.3f} minutes')

    return None
# end combine_statistics

if __name__ == '__main__':
    combine_statistics()
# end