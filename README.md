# WaZP version 2.0

## WaZP short description

This is a code intended to identify and qualify galaxy clusters in large multi-band galaxy photometric surveys. 


## WaZP workflow 

1. 2 tilings of the sky - including overlaps
   * wtiles for detection
   * ptiles for characterization
2. generate redshift slices 
3. computation of global survey quantities
4. for each wtile 
   * loop over redshift slices for detection
   * merge detections over slices 
   * refine redshift of detections
5. concatenate detections over all tiles 
6. for each ptile
   * loop over clusters:estimate richness and membership
7. concatenate richnesses and membership
8. merge detections and richnesses 


## WaZP with SLURM

The division of the sky in tiles offers an easy way to parallelize the code, which is now orchestrated by SLURM.

wazp_main generates and launches 4 sbatch scripts in array mode with dependencie:
   * wazp_tile (step 4 in the workflow)
   * wazp_concatenate (step 5)
   * pmem_tile (step 6)
   * pmem_concatenate (steps 7 and 8)


## Installation 

Clone the repository and create an environment with Conda:
```bash
git clone https://github.com/linea-it/wazp && cd wazp 
conda create -n wazp python=3.11
conda activate wazp
conda install -c conda-forge cfitsio=3.430
conda install -c cta-observatory sparse2d
pip install scikit-image
pip install -U scikit-learn
pip install astropy
pip install healpy
ipython kernel install --user --name=wazp
```


