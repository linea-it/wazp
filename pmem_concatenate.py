import numpy as np
import yaml, os, sys
from astropy.table import join

from lib.utils import read_FitsCat
from lib.wazp import tiles_with_clusters, official_wazp_cat

from lib.pmem import pmem_concatenate_tiles
from lib.pmem import concatenate_calib_dz, eff_tiles_for_pmem


# read config files as online arguments 
config = sys.argv[1]
dconfig = sys.argv[2]
# read config file
with open(config) as fstream:
    param_cfg = yaml.safe_load(fstream)
with open(dconfig) as fstream:
    param_data = yaml.safe_load(fstream)
 
workdir = param_cfg['out_paths']['workdir']
out_paths = param_cfg['out_paths']
admin = param_cfg['admin']
wazp_cfg = param_cfg['wazp_cfg']
pmem_cfg = param_cfg['pmem_cfg']
tiles_filename = os.path.join(
    workdir, admin['tiling_pmem']['rpath'], 
    admin['tiling_pmem']['tiles_filename']
)
all_tiles = read_FitsCat(tiles_filename)
data_clusters = read_FitsCat(param_cfg['clcat']['wazp']['cat'])
clusters = param_cfg['clusters']

# concatenate all tiles
if pmem_cfg['calib_dz']['mode']:
    data_calib = concatenate_calib_dz(
        all_tiles, pmem_cfg, workdir, 
        os.path.join(
            workdir, 'calib', pmem_cfg['calib_dz']['filename']
        )
    )

data_richness, data_members = pmem_concatenate_tiles(
    all_tiles, param_cfg['out_paths'], 
    os.path.join(workdir, 'tmp', 'pmem_richness.fits'),
    os.path.join(workdir, 'wazp_members.fits')
)

# merge clusters + richness  
data_clusters_with_rich = join(data_clusters, data_richness)

#produce wazp cat for distribution
official_wazp_cat(
    data_clusters_with_rich, param_cfg['clcat'][clusters]['keys'], 
    pmem_cfg['richness_specs'], wazp_cfg['rich_min'],
    os.path.join(workdir, 'wazp_clusters.fits')
)

print ('all done folks !')




