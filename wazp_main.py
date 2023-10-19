import numpy as np
import yaml, os, sys, json

from lib.utils import sky_partition, read_FitsCat
from lib.utils import create_mosaic_footprint
from lib.utils import create_directory, add_key_to_fits
from lib.utils import update_data_structure, get_footprint
from lib.utils import slurm_submit

from lib.wazp import compute_zpslices, bkg_global_survey
from lib.wazp import wazp_concatenate
from lib.wazp import update_config, create_wazp_directories
from lib.wazp import tiles_with_clusters, official_wazp_cat
from lib.wazp import store_wazp_confs
from lib.pmem import run_pmem_tile, pmem_concatenate_tiles
from lib.pmem import concatenate_calib_dz, eff_tiles_for_pmem

# read config files as online arguments 
config = sys.argv[1]
dconfig = sys.argv[2]

# open config files
with open(config) as fstream:
    param_cfg = yaml.safe_load(fstream)
with open(dconfig) as fstream:
    param_data = yaml.safe_load(fstream)
globals().update(param_data)

# create directory structure 
survey = param_cfg['survey']
workdir = param_cfg['out_paths']['workdir']
create_wazp_directories(workdir)

# log message 
print ('WaZP run on survey = ', survey)
print ('....... ref filter = ', param_cfg['ref_filter'])
print ('Workdir = ', workdir)

# update param_data & config (ref_filter, etc.)
param_cfg, param_data = update_config(param_cfg, param_data)

# create required data structure if not exist and update params
if not input_data_structure[survey]['footprint_hpx_mosaic']:
    print ('create footprint mosaic')
    create_mosaic_footprint(
        footprint[survey], os.path.join(workdir, 'footprint')
    )
    param_data['footprint'][survey]['mosaic']['dir'] = os.path.join(
        workdir, 'footprint'
    )

# store config files in workdir
config, dconfig = store_wazp_confs(workdir, param_cfg, param_data)

# useful keys 
admin = param_cfg['admin']
wazp_cfg = param_cfg['wazp_cfg']
pmem_cfg = param_cfg['pmem_cfg']
cosmo_params = param_cfg['cosmo_params']
ref_filter = param_cfg['ref_filter']
clusters = param_cfg['clusters']
tiles_filename = os.path.join(
    workdir, 'sky_partition', 
    admin['tiling']['tiles_filename']
)

#
sky_partition(
    admin['tiling'], 
    param_data['galcat'][survey]['mosaic']['dir'],
    param_data['footprint'][survey],
    os.path.join(workdir, 'sky_partition')
)

# compute zp slicing 
compute_zpslices(
    param_data['zp_metrics'][survey][ref_filter], 
    wazp_cfg, -1., workdir
)

# compute global bkg ppties 
print ('Global bkg computation')
bkg_global_survey(
    param_data['galcat'][survey], param_data['footprint'][survey], 
    admin['tiling'], cosmo_params, 
    param_data['magstar_file'][survey][ref_filter], 
    wazp_cfg, workdir)


# run detection on all tiles 
all_tiles = read_FitsCat(tiles_filename)
job_id1 = slurm_submit(
    'wazp_tile', config, dconfig, slurm_cfg, len(all_tiles)
)
# concatenate 
job_id2 = slurm_submit(
    'wazp_concatenate', config, dconfig, dep=job_id1
)


'''

# tiles with clusters 
eff_tiles = tiles_with_clusters(param_cfg['out_paths'], all_tiles)

# concatenate clusters + sort by decreasing SNR + final filtering  
data_clusters = wazp_concatenate(
    eff_tiles, zpslices_filename, wazp_cfg, param_cfg['clcat'], 
    cosmo_params, param_cfg['out_paths']
)
data_clusters.write(param_cfg['clcat']['wazp']['cat'], overwrite=True)

# eff tiles for Pmems (not necessarily = as for wazp because of overlap)
eff_tiles_pmem = eff_tiles_for_pmem(
    data_clusters, param_cfg['clcat']['wazp'], all_tiles, admin
)

# Run pmem on each tile 
print ('Pmem starts')
for ith in np.unique(all_tiles['thread_id']):
    run_pmem_tile(config, dconfig, ith)

# concatenate calib_dz file 
if pmem_cfg['calib_dz']['mode']:
    data_calib = concatenate_calib_dz(
        eff_tiles_pmem, pmem_cfg, workdir, 
        os.path.join(
            workdir, 'calib', pmem_cfg['calib_dz']['filename']
        )
    )

# concatenate pmems
data_richness, data_members = pmem_concatenate_tiles(
    eff_tiles_pmem, param_cfg['out_paths'], 
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

print ('results in ', workdir)
print ('all done folks !')

'''

