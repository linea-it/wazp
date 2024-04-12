import numpy as np
import yaml, os, sys

from lib.utils import read_FitsCat, create_tile_specs
from lib.wazp import wazp_tile

# read config files as online arguments 
config = sys.argv[1]
dconfig = sys.argv[2]
tile_id = int(sys.argv[3])

# read config file
with open(config) as fstream:
    param_cfg = yaml.safe_load(fstream)
with open(dconfig) as fstream:
    param_data = yaml.safe_load(fstream)

# load config info
survey, ref_filter  = param_cfg['survey'], param_cfg['ref_filter']
maglim = param_cfg['maglim_det']
galcat = param_data['galcat'][survey]
clcat = param_cfg['clcat']
out_paths = param_cfg['out_paths']
admin = param_cfg['admin']
footprint = param_data['footprint'][survey]
zp_metrics = param_data['zp_metrics'][survey][ref_filter]
magstar_file = param_data['magstar_file'][survey][ref_filter]
wazp_cfg = param_cfg['wazp_cfg']

# load tiles info
workdir = out_paths['workdir']
all_tiles = read_FitsCat(
    os.path.join(
        workdir, admin['tiling_wazp']['rpath'],
        admin['tiling_wazp']['tiles_filename'])
)
hpix_tile_lists = np.load(
    os.path.join(
        workdir, admin['tiling_wazp']['rpath'],
        admin['tiling_wazp']['tiles_npy']
    ), 
    allow_pickle=True
)
hpix_core_lists = np.load(
    os.path.join(
        workdir, admin['tiling_wazp']['rpath'],
        admin['tiling_wazp']['sky_partition_npy']
    ), 
    allow_pickle=True
)

# generate tile specs and run detection
tile_specs = create_tile_specs(
    admin['target_mode'], admin['tiling_wazp'],
    all_tiles[tile_id],  
    hpix_core_lists[tile_id], hpix_tile_lists[tile_id]
)
wazp_tile(
    admin, tile_specs, 
    galcat, footprint, 
    magstar_file, maglim,
    wazp_cfg, clcat, param_cfg['cosmo_params'], 
    out_paths, param_cfg['verbose'] 
) 

