import numpy as np
import yaml, os, sys

from lib.utils import read_FitsCat, tile_dir_name
from lib.wazp import concatenate_clusters
from lib.wazp import cl_duplicates_filtering
from lib.wazp import add_clusters_unique_id
from lib.wazp import tiles_with_clusters


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
clcat = param_cfg['clcat']
cosmo_params = param_cfg['cosmo_params']
tiles_filename = os.path.join(
    workdir, 'sky_partition', 
    admin['tiling']['tiles_filename']
)
all_tiles = read_FitsCat(tiles_filename)

zpslices_filename = os.path.join(
    workdir, 
    wazp_cfg['zpslices_filename']
)
zpslices = read_FitsCat(zpslices_filename)


# tiles with clusters 
eff_tiles = tiles_with_clusters(out_paths, all_tiles)



# concatenate all tiles 
print ('Concatenate clusters')
list_clusters = []
for it in range(0, len(eff_tiles)):
    tile_dir = tile_dir_name(
        workdir, int(eff_tiles['id'][it]) 
    )
    list_clusters.append(
        os.path.join(tile_dir, out_paths['wazp']['results'])
    )
data_clusters0 = concatenate_clusters(
    list_clusters, 'clusters.fits', 
    os.path.join(workdir, 'tmp', 'clusters0.fits')
)   
 
# final filtering 
print ('........wazp final filtering') 
    
# .... zpmax 
condzmax = (data_clusters0[clcat['wazp']['keys']['key_zp']] <= \
            zpslices['zsl_max'][::-1][0])
condzmin = (data_clusters0[clcat['wazp']['keys']['key_zp']] >= \
            zpslices['zsl'][0])

# .... duplicates 
data_clusters0f = cl_duplicates_filtering(
    data_clusters0[condzmin & condzmax], 
    wazp_cfg, clcat, zpslices, cosmo_params, 
    'survey'
)

# create unique index with decreasing SNR 
data_clusters = add_clusters_unique_id(
    data_clusters0f, clcat['wazp']['keys']
)
data_clusters.write(param_cfg['clcat']['wazp']['cat'], overwrite=True)

