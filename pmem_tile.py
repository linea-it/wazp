import numpy as np
import yaml, os, sys

from lib.utils import read_FitsCat, create_tile_specs
from lib.pmem import pmem_tile

# read config files as online arguments 
config = sys.argv[1]
dconfig = sys.argv[2]
tile_id = int(sys.argv[3])

# read config file
with open(config) as fstream:
    param_cfg = yaml.safe_load(fstream)
with open(dconfig) as fstream:
    param_data = yaml.safe_load(fstream)

survey, ref_filter  = param_cfg['survey'], param_cfg['ref_filter']
maglim = param_cfg['maglim_pmem']
galcat = param_data['galcat'][survey]
clcat = param_data['clcat'][param_cfg['clusters']]
out_paths = param_cfg['out_paths']
admin = param_cfg['admin']
footprint = param_data['footprint'][survey]
zp_metrics = param_data['zp_metrics'][survey][ref_filter]
magstar_file = param_data['magstar_file'][survey][ref_filter]
workdir = out_paths['workdir']
data_cls = read_FitsCat(param_data['clcat'][param_cfg['clusters']]['cat'])

# load tiles info
all_tiles = read_FitsCat(
    os.path.join(
        workdir, admin['tiling_pmem']['rpath'],
        admin['tiling_pmem']['tiles_filename'])
)
hpix_tile_lists = np.load(
    os.path.join(
        workdir, admin['tiling_pmem']['rpath'],
        admin['tiling_pmem']['tiles_npy']
    ), 
    allow_pickle=True
)
hpix_core_lists = np.load(
    os.path.join(
        workdir, admin['tiling_pmem']['rpath'],
        admin['tiling_pmem']['sky_partition_npy']
    ), 
    allow_pickle=True
)

# generate tile specs and run detection
tile_specs = create_tile_specs(
    admin['target_mode'], admin['tiling_pmem'],
    all_tiles[tile_id],  
    hpix_core_lists[tile_id], hpix_tile_lists[tile_id],
)

pmem_tile(
    admin, tile_specs,
    param_cfg['pmem_cfg'], 
    data_cls, data_cls, clcat, 
    footprint, galcat, maglim, 
    zp_metrics['sig_dz0'], param_cfg['cosmo_params'],
    magstar_file, out_paths, param_cfg['verbose']
)


