import parsl
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.launchers import SrunLauncher
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

import cProfile
import pstats
import io
import pandas as pd

import numpy as np
import yaml, os, sys, json
from astropy.table import join
import time

from lib.multithread import split_survey
from lib.utils import create_directory
from lib.utils import update_data_structure, get_footprint
from lib.wazp import compute_zpslices, bkg_global_survey
from lib.wazp import run_wazp_tile, wazp_concatenate
from lib.wazp import update_config, create_wazp_directories, get_in_memory_directory
from lib.wazp import tiles_with_clusters, official_wazp_cat
from lib.pmem import run_pmem_tile, pmem_concatenate_tiles
from lib.pmem import concatenate_calib_dz, eff_tiles_for_pmem

#start Cprofile
pr = cProfile.Profile()
pr.enable()

wazp_init_time = time.time()
# read config files as online arguments 
config = sys.argv[1]
dconfig = sys.argv[2]
parsl_max_workers = int(sys.argv[3])
parsl_nodes_per_block = int(sys.argv[4])

def load_parsl_with_slurm():
    config = Config(
    executors=[
        HighThroughputExecutor(
            label='WaZP_SD',
            # Optional: The network interface on node 0 which compute nodes can communicate with.
            #address=address_by_interface('ipogif0')
            # one worker per manager / node
            max_workers=parsl_max_workers,
            provider=LocalProvider(
                channel=LocalChannel(script_dir='.'),
                # make sure the nodes_per_block matches the nodes requested in the submit script in the next step
                nodes_per_block=parsl_nodes_per_block,
                #launcher=SrunLauncher(overrides='-c 32'),
                launcher=SrunLauncher(),
                cmd_timeout=120,
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy=None,
    )

    parsl.load(config)

load_parsl_with_slurm()


# open config files
with open(config) as fstream:
    param_cfg = yaml.load(fstream)
with open(dconfig) as fstream:
    param_data = yaml.load(fstream)


# log message 
print ('WaZP run on survey = ', param_cfg['survey'])
print ('....... ref filter = ', param_cfg['ref_filter'])
print ('Workdir = ', param_cfg['out_paths']['workdir'])

# create directory structure 
workdir = param_cfg['out_paths']['workdir']
create_wazp_directories(workdir)

# create in memory directory, depends on $UID
in_mem_dir = get_in_memory_directory()
create_directory(in_mem_dir)

# create required data structure if not exist and update params
param_data = update_data_structure(param_cfg, param_data)

# update configs (ref_filter, etc.)
param_cfg, param_data = update_config(param_cfg, param_data)

# store config file in workdir
with open(
        os.path.join(workdir, 'config', 'wazp.cfg'), 'w'
) as outfile:
    json.dump(param_cfg, outfile)
config = os.path.join(workdir, 'config', 'wazp.cfg')    
with open(
        os.path.join(workdir, 'config', 'data.cfg'), 'w'
) as outfile:
    json.dump(param_data, outfile)
dconfig = os.path.join(workdir, 'config', 'data.cfg')    

# useful keys 
admin = param_cfg['admin']
wazp_cfg = param_cfg['wazp_cfg']
pmem_cfg = param_cfg['pmem_cfg']
tiles_filename = os.path.join(
    workdir, admin['tiling']['tiles_filename']
)
zpslices_filename = os.path.join(
    workdir, wazp_cfg['zpslices_filename']
)
gbkg_filename = os.path.join(workdir, 'gbkg', wazp_cfg['gbkg_filename'])
cosmo_params = param_cfg['cosmo_params']
survey = param_cfg['survey']
ref_filter = param_cfg['ref_filter']
clusters = param_cfg['clusters']


# read or create global footprint & split survey 
survey_footprint = get_footprint(
    param_data['input_data_structure'][survey], 
    param_data['footprint'][survey], workdir
)
all_tiles = split_survey(
    survey_footprint, param_data['footprint'][survey], 
    admin, tiles_filename
)

# compute zp slicing 
compute_zpslices(
    param_data['zp_metrics'][survey][ref_filter], 
    wazp_cfg, -1., zpslices_filename
)

# compute global bkg ppties 
if not os.path.isfile(gbkg_filename):
    print ('Global bkg computation')
    bkg_global_survey(
        param_data['galcat'][survey], param_data['footprint'][survey], 
        tiles_filename, zpslices_filename, 
        admin['tiling'], cosmo_params, 
        param_data['magstar_file'][survey][ref_filter], 
        wazp_cfg, gbkg_filename)

# detect clusters on all tiles 
print ('Run wazp in tiles')

# Measure time for run_waz_tile
run_wazp_all_tile_start = time.time()

for ith in np.unique(all_tiles['thread_id']): 
    each_tile_start = time.time()
    run_wazp_tile(config, dconfig, ith)
    each_tile_end = time.time()
    print('Tile ', ith, 'time processing is:', (each_tile_end - each_tile_start), flush=True)
# Call python app in a for loop
# tile_results = []
# for ith in np.unique(all_tiles['thread_id']): 

    # tile_results.append(run_wazp_tile(config, dconfig, ith))

# outputs_tiles = [r.result() for r in tile_results]
# print(outputs_tiles)

run_wazp_all_tile_end = time.time()
print('Total Time all Tiles: ', (run_wazp_all_tile_end - run_wazp_all_tile_start), flush=True)

# Stop Cprofile and generate csv
pr.disable()
result = io.StringIO()
pstats.Stats(pr,stream=result).print_stats()
result=result.getvalue()
# chop the string into a csv-like buffer
result='ncalls'+result.split('ncalls')[-1]
result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
# save it to disk
filename="teste.csv"
with open(filename, 'w+') as f:
    f.write(result)
    f.close()


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
pmem_start_time = time.time()
for ith in np.unique(all_tiles['thread_id']):
    pmem_tile_start_time = time.time()
    run_pmem_tile(config, dconfig, ith)
    pmem_tile_end_time = time.time()
    print('Pmem Time for Tile ', ith, 'is:', (pmem_tile_end_time - pmem_tile_start_time), flush=True)

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
pmem_end_time = time.time()
print('Pmem and concatenate Time is:', (pmem_end_time-pmem_start_time), flush=True)
wazp_end_time = time.time()
print('WaZP Time Execution :', (wazp_end_time - wazp_init_time))
print ('results in ', workdir)
print ('all done folks !')
