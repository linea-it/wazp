
README for WaZP - 27/06/2022

WaZP content :

- call script :    wazp_main.py 
- parameter file : wazp.cfg 
- data config:     data.cfg
- wazp/ : detection.py  pmem.py utils.py  multithread.py 
- aux_data/ : auxiliary files 
       	 + mag_star files for each survey/filter 
- input_data/ : example of input galcat + footprint 
- output_ref/ : example of output with verbose=1


Launch wazp in one line : 
> python wazp_main.py wazp.cfg data.cfg
 default configuration to allow the generation of a 
 folder output/ to be compared to output_ref/


Requirements :
- Python 3.6
- installation of SPARSE2D / mr_filter C++ code 


Parallelization : 
- The natural multithreading comes in 2 different places of 
  wazp_main.py 

    for ith in np.unique(all_tiles['thread_id']): 
    	run_wazp_tile(config, dconfig, ith)

and 

    for ith in np.unique(all_tiles['thread_id']):
    	run_pmem_tile(config, dconfig, ith)

Here it comes as a loop that should be treated with your 
favourite multithreading fct or batch scheduler. 

Note that the number of threads can be updated in the wazp.cfg
file under :  'admin / nthread_max'

Note on the data.cfg file : 
- this file describes various implemented surveys
- each survey contains 4 types of products 
  + a galaxy catalog
  + a footprint
  + a magstar file 
  + a photo z metric 
- each product is described by 
  + its location on disc 
  + key names of the quantities used by wazp
  + data structure (presented in tiles or single files)

Note on the wazp.cfg file : 
- The header of this file specifies 
  + the survey being used 
  + the selected reference filter 
  + the level of verbosity 

Scientific use of WaZP: 
- Preparation of the wazp.cfg and data.cfg files
  + install survey files and update data.cfg
  + select survey and filter in wazp.cfg


Some basic properties of WaZP  :
- the tiling is done in healpix pixels so that 
  it can operate at any RA-Dec
  Technically, detection is performed on a spherical 
  cap that has an angular radius large enough to enclose 
  the healpix tile + an overlap region between the tiles. 

- the divisision of the N tiles in P cores is 
  done to optimize the distribution of the area 
  to be analyzed (as equal area as possible). 

- there are 3 levels of verbose. With verbose = 0 no intermediate 
  file is written on disc except those necessary for the code.

- there are several re-entry points with the generation of 
  numpy files (.npx). But this can be switched off if necessary


Main steps of wazp_main.py : 

- read the wazp.cfg 
- from the selected survey and filter update the 
  wazp.cfg and data.cfg to match the survey properties (z range..).
- compute global mean bkg densities in a subset of the survey 
  in an area specified in wazp.cfg
- build the list of tiles and zp slices 
- Exectute detection in each tile (parallel)
- combine all outputs, filter and rank by decreasing SNR 
- Execute cluster membership (pmem) and richness computation in 
  each tile
- combine all outputs and merge detection + pmem


