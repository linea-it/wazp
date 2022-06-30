# WaZP version 1.0

## WaZP short description

This is a code intended to identify and qualify galaxy clusters in large multi-band galaxy photometric surveys. 


## WaZP workflow 

1. tiling of the sky - including overlaps 
2. generate redshift slices 
3. computation of global survey quantities
4. for each tile 
   * loop over redshift slices for detection
   * merge detections over slices 
   * refine redshift of detections
5. concatenate detections over all tiles 
6. for each tile
   * loop over clusters:estimate richness and membership
7. concatenate richnesses and membership
8. merge detections and richnesses 

## Installation 

Clone the repository and create an environment with Conda:
```bash
git clone https://github.com/linea-it/wazp && cd wazp 
conda create -n wazp python=3.8
conda activate wazp
conda install -c conda-forge cfitsio=3.430
conda install -c cta-observatory sparse2d
pip install scikit-image
pip install astropy
pip install healpy
ipython kernel install --user --name=wazp
```


