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
        workdir, 'sky_partition',
        admin['tiling']['tiles_filename'])
)
idt = np.argwhere(all_tiles['thread_id']==int(thread_id)).T[0]    
tiles_specs = all_tiles[idt]
hpix_tile_lists = np.load(
    os.path.join(
        workdir, 'sky_partition', 
        admin['tiling']['tiles_npy']
    ), 
    allow_pickle=True
)[idt]
hpix_core_lists = np.load(
    os.path.join(
        workdir, 'sky_partition', 
        admin['tiling']['sky_partition_npy']
    ), 
    allow_pickle=True
)[idt]

# generate tile specs and run detection
tile_specs = create_tile_specs(
    tiles_specs[tile_id], admin, 
    hpix_core_lists[tile_id], hpix_tile_lists[tile_id],
    None, None
)
wazp_tile(
    admin, tile_specs, 
    galcat, footprint, 
    zp_metrics, magstar_file, maglim,
    wazp_cfg, clcat, param_cfg['cosmo_params'], 
    out_paths, param_cfg['verbose'] 
) 

