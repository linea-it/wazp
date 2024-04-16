import numpy as np
import yaml, os, sys, json

from wazp.utils import sky_partition, read_FitsCat
from wazp.utils import create_mosaic_footprint
from wazp.utils import create_directory, update_data_structure
from wazp.utils import slurm_submit

from wazp.detection import compute_zpslices, bkg_global_survey
from wazp.detection import update_config, create_wazp_directories
from wazp.detection import store_wazp_confs

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
detection_cfg = param_cfg['detection_cfg']
pmem_cfg = param_cfg['pmem_cfg']
cosmo_params = param_cfg['cosmo_params']
ref_filter = param_cfg['ref_filter']
clusters = param_cfg['clusters']

# compute zp slicing 
compute_zpslices(
    param_data['zp_metrics'][survey][ref_filter], 
    detection_cfg, -1., workdir
)

# Partition for bkg & detection 
ntiles_detection = sky_partition(
    admin['tiling_detection'], 
    param_data['galcat'][survey]['mosaic']['dir'],
    param_data['footprint'][survey], workdir
)

# Partition for pmem
ntiles_pmem = sky_partition(
    admin['tiling_pmem'], 
    param_data['galcat'][survey]['mosaic']['dir'],
    param_data['footprint'][survey], workdir
)

# compute global bkg ppties 
bkg_global_survey(
    param_data['galcat'][survey], param_data['footprint'][survey], 
    admin['tiling_detection'], cosmo_params, 
    param_data['magstar_file'][survey][ref_filter], 
    detection_cfg, workdir
)

# run detection 
print ('Run detection_tile / slurm ')
job_id1 = slurm_submit(
    'detection_tile', config, dconfig, narray=ntiles_detection
)
# concatenate 
job_id2 = slurm_submit(
    'detection_concatenate', config, dconfig, dep=job_id1
)

# run Pmem 
print ('Run pmem_tile / slurm ')
job_id3 = slurm_submit(
    'pmem_tile', config, dconfig, narray=ntiles_pmem,
    dep=job_id2
)

# concatenate Pmems 
job_id4 = slurm_submit(
    'pmem_concatenate', config, dconfig, dep=job_id3
)

print ('results in ', workdir)
