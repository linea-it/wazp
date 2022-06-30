import numpy as np 
import astropy.io.fits as fits
import os

from .utils import read_FitsCat, hpx_split_survey
from .utils import add_key_to_fits


def split_equal_nr_of_tiles_in_threads(n_threads, ntiles):
    if ntiles <= n_threads:
        n_threads = ntiles
        thread_ids = np.arange(0, ntiles)

    if n_threads == 0:
        n_threads = ntiles
        thread_ids = np.arange(0, ntiles)

    if ntiles > n_threads:
        thread_ids = np.zeros(ntiles)

        p = float(ntiles)/float(n_threads)

        if ntiles % n_threads == 0:
            ng = int(p)
            k=0
            for i in range(0, n_threads):
                for j in range(0, ng):
                    thread_ids[k] = i
                    k+=1    

        if ntiles % n_threads !=0:
            ng = int(p)
            nr = ntiles - ng * n_threads

            k=0
            for i in range(0, n_threads):
                if i < nr:
                    for j in range(0, ng+1):
                        thread_ids[k] = i
                        k+=1    
                else:
                    for j in range(0, ng):
                        thread_ids[k] = i
                        k+=1    

    thread_idsf = thread_ids + 1

    return n_threads, thread_idsf.astype(int)



def split_equal_area_in_threads(n_threads, tiles_filename):

    tiles = read_FitsCat(tiles_filename)
    ntiles = len(tiles)

    if ntiles <= n_threads:
        n_threads = ntiles
        thread_ids = np.arange(0, ntiles)

    if n_threads == 0:
        n_threads = ntiles
        thread_ids = np.arange(0, ntiles)

    if ntiles > n_threads:

        eff_area = tiles['eff_area_deg2']
        area_thread = np.sum(eff_area)/float(n_threads)        
        thread_ids = np.zeros(ntiles)
        area_per_thread = np.zeros(n_threads)
        for j in np.argsort(-eff_area):
            i = np.argmin(area_per_thread)
            thread_ids[j]=i
            area_per_thread[i] += eff_area[j]

        print ('.....mean, min, max area / thread (deg2) = ', 
               np.round(area_thread, 2), 
               np.round(np.amin(area_per_thread), 2), 
               np.round(np.amax(area_per_thread), 2))

    thread_idsf = thread_ids + 1

    return n_threads, thread_idsf.astype(int)


def split_survey(survey_footprint, footprint, admin, tiles_filename):

    if not os.path.isfile(tiles_filename):
        ntiles = hpx_split_survey(
            survey_footprint, footprint, admin['tiling'], tiles_filename
        )
        n_threads, thread_ids = split_equal_area_in_threads(
            admin['nthreads_max'], tiles_filename
        )
        add_key_to_fits(tiles_filename, thread_ids, 'thread_id', 'int')
        all_tiles = read_FitsCat(tiles_filename)
    else:
        all_tiles = read_FitsCat(tiles_filename)
        ntiles, n_threads = len(all_tiles), np.amax(all_tiles['thread_id']) 

    print ('Ntiles / Nthreads = ', ntiles, ' / ', n_threads)
    return all_tiles
