import numpy as np
import yaml, os, sys

from lib.utils import read_FitsCat, add_clusters_unique_id
from lib.utils import concatenate_cl_tiles
from lib.wazp import cl_duplicates_filtering


# read config files as online arguments 
config = sys.argv[1]
dconfig = sys.argv[2]
# read config file
with open(config) as fstream:
    param_cfg = yaml.safe_load(fstream)
with open(dconfig) as fstream:
    param_data = yaml.safe_load(fstream)

# definition of paths and files
workdir = param_cfg['out_paths']['workdir']
out_paths = param_cfg['out_paths']
admin = param_cfg['admin']
wazp_cfg = param_cfg['wazp_cfg']
clcat = param_cfg['clcat']
cosmo_params = param_cfg['cosmo_params']
tiles_filename = os.path.join(
    workdir, admin['tiling_wazp']['rpath'], 
    admin['tiling_wazp']['tiles_filename']
)
all_tiles = read_FitsCat(tiles_filename)
zpslices_filename = os.path.join(
    workdir, 
    wazp_cfg['zpslices_filename']
)
zpslices = read_FitsCat(zpslices_filename)


# concatenate all tiles 
data_clusters0 = concatenate_cl_tiles(out_paths, all_tiles, 'wazp')

# final filtering of duplicates & zpmax
data_clusters0f = cl_duplicates_filtering(
    data_clusters0, wazp_cfg, clcat, zpslices, cosmo_params, 
    'survey'
)

# create unique index with decreasing SNR 
data_clusters = add_clusters_unique_id(
    data_clusters0f, clcat['wazp']['keys']
)
data_clusters.write(param_cfg['clcat']['wazp']['cat'], overwrite=True)

