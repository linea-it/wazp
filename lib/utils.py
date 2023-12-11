import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import yaml, os, sys, subprocess, time
from astropy.cosmology import FlatLambdaCDM as flat
from astropy import units as u
import healpy as hp
from math import log, exp, atan, atanh
from astropy.table import Table
from sklearn import cluster


def create_slurm_script(task, config, dconfig, narray, script):
    with open(config) as fstream:
        param_cfg = yaml.safe_load(fstream)
    slurm_cfg = param_cfg['admin']['slurm']

    print ('...task = ', task)
    print (
        '      ntiles / max nthreads = ',
        narray, slurm_cfg['max_parallel']
    )

    scr = __file__.replace('lib/utils.py', '')

    f = open(f"{script}", "w")
    f.write("#!/bin/sh\n")
    f.write(f"#SBATCH --job-name={task}\n")
    if narray > 1:
        f.write(
            f"#SBATCH --cpus-per-task={slurm_cfg['cpus-per-task']}\n"
        )
        f.write(
            f"#SBATCH --array=0-{narray-1}%{slurm_cfg['max_parallel']}\n"
        )
    f.write(f"#SBATCH --mem={slurm_cfg['memory'][task]}G\n")
    if narray > 1:
        f.write(f"python {scr}{task}.py {config} {dconfig} $SLURM_ARRAY_TASK_ID\n")
    else:
        f.write(f"python {scr}{task}.py  {config} {dconfig} 0\n")
    f.close()
    return 
    

def slurm_submit(task, config, dconfig, narray=1, dep=None):

    if dep is not None:
        time.sleep(3)

    create_slurm_script(task, config, dconfig, narray, f"job_{task}.sh")    
    if dep is not None:
        cmd = f"sbatch --depend=afterany:{dep} job_{task}.sh"
    else:
        cmd = f"sbatch job_{task}.sh"

    res = subprocess.run(cmd, shell=True, capture_output=True)
    job_id = str(res.stdout).split("batch job ")[1].split("\\")[0]
    return job_id


def make_tile_cat(tile_specs, sourcecat, footprint, maglim, wcfg):
    
    data_gal_tile = read_hpix_mosaicFitsCat(
        tile_specs['hpix_tile'], 
        sourcecat['mosaic']['dir']
    )
    data_gal_tile = data_gal_tile\
                    [data_gal_tile[sourcecat['keys']['key_mag']]<=\
                     np.float64(maglim)]

    data_fp_tile = read_hpix_mosaicFootprint(
        tile_specs['hpix_tile'], 
        footprint['mosaic']['dir']
    )

    # add hpx to galcat to speed up condition_in_disc around all detections
    data_gal_tile = add_hpx_to_cat(
        data_gal_tile,
        data_gal_tile[sourcecat['keys']['key_ra']], 
        data_gal_tile[sourcecat['keys']['key_dec']],
        wcfg['Nside_tmp'], wcfg['nest_tmp'], 'hpix_tmp'
    )
    return data_gal_tile, data_fp_tile


def create_directory(dir):
    """_summary_

    Args:
        dir (_type_): _description_
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    return


def read_FitsCat(cat):
    """_summary_

    Args:
        cat (_type_): _description_

    Returns:
        _type_: _description_
    """
    hdulist=fits.open(cat)
    dat=hdulist[1].data
    hdulist.close()
    return dat


def read_FitsFootprint(hpx_footprint, hpx_meta):
    """_summary_

    Args:
        hpx_footprint (_type_): _description_
        hpx_meta (_type_): _description_

    Returns:
        _type_: _description_
    """

    dat = read_FitsCat(hpx_footprint)
    hpix_map = dat[hpx_meta['key_pixel']].astype(int)
    if hpx_meta['key_frac'] is None:
        frac_map = np.ones(len(hpix_map)).astype(float)
    else:
        frac_map = dat[hpx_meta['key_frac']]
    return  hpix_map, frac_map


def read_sources_in_hpixs(dat, keys, hpix_arr, Nside, nest):
    hpix = hp.ang2pix(
        Nside, dat[keys['key_ra']], dat[keys['key_dec']],
        nest, lonlat=True)
    return dat[np.isin(hpix, hpix_arr)]
    

def read_hpix_mosaicFitsCat(hpix_arr, datadir):

    i = 0
    for hh in hpix_arr:
        datas = read_FitsCat(
            os.path.join(datadir, str(hh)+'.fits')
        )
        if i == 0:
            data = np.copy(datas)
        else:
            data = np.append(data, datas)
        i+=1
    return data


def read_hpix_mosaicFootprint(hpix_arr, datadir):

    i = 0
    for hh in hpix_arr:
        datas = read_FitsCat(
            os.path.join(datadir, str(hh)+'_footprint.fits')
        )
        if i == 0:
            data = np.copy(datas)
        else:
            data = np.append(data, datas)
        i+=1
    return data


def read_mosaicFitsCat_in_disc (galcat, tile, radius_deg):
    """From a list of galcat files, selects objects in a cone centered 
    on racen, deccen Output is a structured array

    Args:
        galcat (_type_): _description_
        tile (_type_): _description_
        radius_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    # tile 
    racen, deccen = tile['ra'], tile['dec']
    # list of available galcats => healpix pixels 
    gdir = galcat['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0] for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]

    # find list of fits intersection cluster field
    Nside_fits, nest_fits = galcat['mosaic']['Nside'],\
                            galcat['mosaic']['nest']
    fits_pixels_in_disc = hp.query_disc(
        nside=Nside_fits, nest=nest_fits, 
        vec=hp.ang2vec(racen, deccen, lonlat=True),
        radius = np.radians(radius_deg), inclusive=True
    )
    relevant_fits_pixels = fits_pixels_in_disc\
                           [np.isin(
                               fits_pixels_in_disc, 
                               hpix_fits, 
                               assume_unique=True
                           )]

    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat_disc = read_FitsCat(
                os.path.join(gdir, str(relevant_fits_pixels[i])+extension)
            )
            dcen = np.degrees( 
                dist_ang(
                    dat_disc[galcat['keys']['key_ra']], 
                    dat_disc[galcat['keys']['key_dec']],
                    racen, deccen
                )
            )
            if i == 0:
                data_gal_disc = np.copy(dat_disc[dcen<radius_deg])
            else:
                data_gal_disc = np.append(
                    data_gal_disc, 
                    dat_disc[dcen<radius_deg]
                )
    else:
        data_gal_disc = None
    return data_gal_disc


def read_mosaicFootprint_in_disc (footprint, tile, radius_deg):
    """From a list of galcat files, selects objects in a cone 
    centered on racen, deccen
    Output is a structured array

    Args:
        footprint (_type_): _description_
        tile (_type_): _description_
        radius_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    # tile 
    racen, deccen = tile['ra'], tile['dec']
    # list of available galcats => healpix pixels 
    gdir = footprint['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0].replace('_footprint','') for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    # find list of fits intersection cluster field
    Nside_fits, nest_fits = footprint['mosaic']['Nside'],\
                            footprint['mosaic']['nest']
    fits_pixels_in_disc = hp.query_disc(
        nside=Nside_fits, nest=nest_fits, 
        vec=hp.ang2vec(racen, deccen, lonlat=True),
        radius = np.radians(radius_deg), 
        inclusive=True
    )
    relevant_fits_pixels = fits_pixels_in_disc\
                           [np.isin(
                               fits_pixels_in_disc, 
                               hpix_fits, 
                               assume_unique=True
                           )]
    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat_disc = read_FitsCat(
                os.path.join(
                    gdir, 
                    str(relevant_fits_pixels[i])+'_footprint'+extension
                )
            )
            ra, dec = hp.pix2ang(
                footprint['Nside'],
                dat_disc[footprint['key_pixel']],
                footprint['nest'], 
                lonlat=True
            )
            dcen = np.degrees(dist_ang(ra, dec, racen, deccen))
            if i == 0:
                data_fp_disc = np.copy(dat_disc[dcen<radius_deg])
            else:
                data_fp_disc = np.append(
                    data_fp_disc, 
                    dat_disc[dcen<radius_deg]
                )
    else:
        data_fp_disc = None

    return data_fp_disc


def read_mosaicFitsCat_in_hpix (galcat, hpix_tile, Nside_tile, nest_tile):
    """_summary_

    Args:
        footprint (_type_): _description_
        hpix_tile (_type_): _description_
        Nside_tile (_type_): _description_
        nest_tile (_type_): _description_

    Returns:
        _type_: _description_
    """
    """
    From a list of galcat files, selects objects in a cone 
    centered on racen, deccen
    Output is a structured array
    """
    # list of available galcats => healpix pixels 
    gdir = galcat['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0] for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    Nside_fits, nest_fits = galcat['mosaic']['Nside'], galcat['mosaic']['nest']

    # warning we assume Nside_tile > Nside_fits !!
    ra_fits, dec_fits = hp.pix2ang(
        Nside_fits, hpix_fits, nest_fits, lonlat=True 
    )
    hpix_fits_tile = hp.ang2pix(
        Nside_tile, ra_fits, dec_fits, nest_tile, lonlat=True
    )
    relevant_fits_pixels = np.unique(
        hpix_fits[np.isin(hpix_fits_tile, hpix_tile)]
    )
    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat = read_FitsCat(
                os.path.join(gdir, str(relevant_fits_pixels[i])+extension)
            )
            if i == 0:
                data_gal_hpix = np.copy(dat)
            else:
                data_gal_hpix = np.append(data_gal_hpix, dat)
    else:
        data_gal_hpix = None
    return data_gal_hpix


def read_mosaicFootprint_in_hpix (footprint, hpix_tile, Nside_tile, nest_tile):
    """_summary_

    Args:
        footprint (_type_): _description_
        hpix_tile (_type_): _description_
        Nside_tile (_type_): _description_
        nest_tile (_type_): _description_

    Returns:
        _type_: _description_
    """

    # list of available footprints => healpix pixels 
    gdir = footprint['mosaic']['dir']
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0].replace('_footprint','') for x in raw_list]
    ).astype(int)
    extension = os.path.splitext(raw_list[0])[1]
    Nside_fits, nest_fits = footprint['mosaic']['Nside'],\
                            footprint['mosaic']['nest']

    # warning we assume Nside_tile > Nside_fits !!
    ra_fits, dec_fits = hp.pix2ang(
        Nside_fits, hpix_fits, nest_fits, lonlat=True 
    )
    hpix_fits_tile = hp.ang2pix(
        Nside_tile, ra_fits, dec_fits, nest_tile, lonlat=True
    )

    relevant_fits_pixels = np.unique(
        hpix_fits[np.isin(hpix_fits_tile, hpix_tile)]
    )

    if len(relevant_fits_pixels) > 0:
        # merge intersecting fits 
        for i in range (0, len(relevant_fits_pixels)):
            dat = read_FitsCat(
                os.path.join(
                    gdir, 
                    str(relevant_fits_pixels[i])+'_footprint'+extension
                )
            )
            if i == 0:
                data_fp_hpix = np.copy(dat)
            else:
                data_fp_hpix = np.append(data_fp_hpix, dat)
    else:
        data_fp_hpix = None
    return data_fp_hpix


def create_survey_footprint_from_mosaic(footprint, survey_footprint):
    """_summary_

    Args:
        footprint (_type_): _description_
        fpath (_type_): _description_
    """
    all_files = np.array(os.listdir(footprint['mosaic']['dir']))
    flist = [os.path.join(footprint['mosaic']['dir'], f) for f in all_files]
    concatenate_fits(flist, survey_footprint)
    return


def create_mosaic_footprint(footprint, fpath):
    """_summary_

    Args:
        footprint (_type_): _description_
        fpath (_type_): _description_
    """
    # from a survey footprint create a mosaic of footprints at lower resol.
    if os.path.exists(fpath):
        return

    print ('Create footprint mosaic')
    create_directory(fpath)
    hpix0, frac0 = read_FitsFootprint(
        footprint['survey_footprint'], footprint
    )
    ra0, dec0 = hp.pix2ang(
        footprint['Nside'], hpix0, footprint['nest'], lonlat=True 
    )
    hpix = hp.ang2pix(
        footprint['mosaic']['Nside'], ra0, dec0, footprint['mosaic']['nest'], 
        lonlat=True
    )
    for hpu in np.unique(hpix):
        all_cols = fits.ColDefs([
            fits.Column(
                name = footprint['key_pixel'],  
                format = 'K',
                array = hpix0[np.isin(hpix, hpu)]
            ),
            fits.Column(
                name = 'ra',       
                format = 'D',
                array = ra0[np.isin(hpix, hpu)]
            ),
            fits.Column(
                name = 'dec',      
                format = 'D',
                array = dec0[np.isin(hpix, hpu)]
            ),
            fits.Column(
                name = footprint['key_frac'],   
                format = 'E',
                array = frac0[np.isin(hpix, hpu)]
            )
        ])
        hdu = fits.BinTableHDU.from_columns(all_cols)
        hdu.writeto(
            os.path.join(fpath, str(hpu)+'_footprint.fits'),
            overwrite=True
        )
    return


def concatenate_fits(flist, output):
    """_summary_

    Args:
        flist (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range (0, len(flist)):
        dat = read_FitsCat(flist[i])
        if i == 0:
            cdat = np.copy(dat)
        else:
            cdat = np.append(cdat, dat)
    t = Table(cdat)
    t.write(output, overwrite=True)
    return cdat


def concatenate_fits_with_label(flist, label_name, label, output):
    """_summary_

    Args:
        flist (_type_): _description_
        label_name (_type_): _description_
        label (_type_): _description_
        output (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in range (0, len(flist)):
        dat = read_FitsCat(flist[i])
        datL = Table(dat)
        datL[label_name] = int(label[i])*np.ones(len(dat)).astype(int)
        if i == 0:
            cdat = np.copy(datL)
        else:
            cdat = np.append(cdat, datL)
    t = Table(cdat)#, names=names)
    t.write(output, overwrite=True)
    return cdat


def add_key_to_fits(fitsfile, key_val, key_name, key_type):
    """_summary_

    Args:
        fitsfile (_type_): _description_
        key_val (_type_): _description_
        key_name (_type_): _description_
        key_type (_type_): _description_
    """
    dat = read_FitsCat(fitsfile)
    orig_cols = dat.columns
    if key_type == 'float':
        new_col = fits.ColDefs([
            fits.Column(name=key_name, format='E',array=key_val)])
    if key_type == 'int':
        new_col = fits.ColDefs([
            fits.Column(name=key_name, format='J',array=key_val)])

    hdu = fits.BinTableHDU.from_columns(orig_cols + new_col)    
    hdu.writeto(fitsfile, overwrite = True)
    return

        
def filter_hpx_tile(data, cat, tile_specs):
    """_summary_

    Args:
        data (_type_): _description_
        cat (_type_): _description_
        tile_specs (_type_): _description_

    Returns:
        _type_: _description_
    """
    ra, dec = data[cat['keys']['key_ra']],\
              data[cat['keys']['key_dec']]
    Nside, nest = tile_specs['Nside'], tile_specs['nest']
    pixel_tile = tile_specs['hpix']
    hpx = hp.ang2pix(Nside, ra, dec, nest, lonlat=True)
    return data[np.argwhere(hpx == pixel_tile).T[0]]


def filter_disc_tile(data, cat, tile_specs):
    """_summary_

    Args:
        data (_type_): _description_
        cat (_type_): _description_
        tile_specs (_type_): _description_

    Returns:
        _type_: _description_
    """
    ra, dec = data[cat['keys']['key_ra']],\
              data[cat['keys']['key_dec']]
    ra_tile, dec_tile = tile_specs['ra'], tile_specs['ra']
    radius_tile_deg = tile_specs['radius_tile_deg']
    dcen_deg = np.degrees(dist_ang(ra, dec, ra_tile, dec_tile))
    return data[dcen_deg<=radius_tile_deg]


def add_hpx_to_cat(data_gal, ra, dec, Nside_tmp, nest_tmp, keyname):
    """_summary_

    Args:
        data_gal (_type_): _description_
        ra (_type_): _description_
        dec (_type_): _description_
        Nside_tmp (_type_): _description_
        nest_tmp (_type_): _description_
        keyname (_type_): _description_

    Returns:
        _type_: _description_
    """
    ghpx = hp.ang2pix(Nside_tmp, ra, dec, nest_tmp, lonlat=True)
    t = Table (data_gal)
    t[keyname] = ghpx
    return t


def mad(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1.4826*np.median(abs(x))


def gaussian(x, mu, sig):
    """_summary_

    Args:
        x (_type_): _description_
        mu (_type_): _description_
        sig (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.exp(-(x - mu)**2 / (2.*sig**2) ) / (sig * np.sqrt(2.*np.pi))


def dist_ang(ra1, dec1, ra_ref, dec_ref):
    """_summary_

    Args:
        ra1 (_type_): _description_
        dec1 (_type_): _description_
        ra_ref (_type_): _description_
        dec_ref (_type_): _description_

    Returns:
        _type_: _description_
    """
    """
    angular distance between (ra1, dec1) and (ra_ref, dec_ref)
    ra-dec in degrees
    ra1-dec1 can be arrays 
    ra_ref-dec_ref are scalars
    output is in radian
    """
    costheta = np.sin(np.radians(dec_ref)) * np.sin(np.radians(dec1)) +\
               np.cos(np.radians(dec_ref)) * np.cos(np.radians(dec1)) *\
               np.cos(np.radians(ra1-ra_ref))
    costheta[costheta>1.] = 1.
    costheta[costheta<-1.] = -1.
    dist_ang = np.arccos(costheta)
    return dist_ang 


def area_ann_deg2(theta_1, theta_2):
    """_summary_

    Args:
        theta_1 (_type_): _description_
        theta_2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    area = 2. * np.pi * (np.cos(np.radians(theta_1)) -\
                         np.cos(np.radians(theta_2))) *\
        (180./np.pi)**2
    return area


def _mstar_ (mstar_filename, zin):
    """
    from a given (z, mstar) ascii file
    interpolate to provide the mstar at a given z_in
    """
    zst, mst = np.loadtxt(mstar_filename, usecols=(0, 1), unpack=True)
    return np.interp (zin,zst,mst)


def join_struct_arrays(arrays):
    """_summary_

    Args:
        arrays (_type_): _description_

    Returns:
        _type_: _description_
    """
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:,offset:offset+size] = a.view(np.uint8).reshape(n,size)
        #print ('desc ', a.dtype.descr)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)

def radec_window_area (ramin, ramax, decmin, decmax):
    """_summary_

    Args:
        ramin (_type_): _description_
        ramax (_type_): _description_
        decmin (_type_): _description_
        decmax (_type_): _description_

    Returns:
        _type_: _description_
    """
    nstep = int((decmax-decmin)/0.1)+1
    step = (decmax-decmin)/float(nstep)
    decmini = np.arange(decmin, decmax, step)
    decmaxi = decmini+step
    decceni = (decmini + decmaxi)/2.
    darea = (ramax-ramin)*np.cos(np.pi*decceni/180.)*(decmaxi-decmini)
    return np.sum(darea)


# healpix functions
def sub_hpix(hpix, Nside, nest):
    """_summary_

    Args:
        hpix (_type_): _description_
        Nside (_type_): _description_
        nest (_type_): _description_

    Returns:
        _type_: _description_
    """
    # from a list of pixels at resolution Nside 
    # get the corresponding list at resolution Nside*2
    rac, decc = np.zeros(4*len(hpix)), np.zeros(4*len(hpix))
    i=0
    for p in hpix:
        ra, dec = hp.vec2ang(hp.boundaries(Nside, p, 1, nest).T, lonlat=True)
        racen, deccen = hp.pix2ang(Nside, p, nest, lonlat=True)
        for j in range(0,4):
            rac[i], decc[i] = (ra[j]+racen)/2., (dec[j]+deccen)/2. 
            i+=1
    return hp.ang2pix(Nside*2, rac, decc, nest, lonlat=True)


def makeHealpixMap(ra, dec, weights=None, nside=1024, nest=False):
    """_summary_

    Args:
        ra (_type_): _description_
        dec (_type_): _description_
        weights (_type_, optional): _description_. Defaults to None.
        nside (int, optional): _description_. Defaults to 1024.
        nest (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # convert a ra/dec catalog into healpix map with counts pe_r cell
    ipix = hp.ang2pix(nside, (90-dec)/180*np.pi, ra/180*np.pi, nest=nest)
    return np.bincount(ipix, weights = weights, minlength=hp.nside2npix(nside))


def all_hpx_in_annulus (ra, dec, radius_in_deg, radius_out_deg, 
                        hpx_meta, inclusive):
    """
    Get the list of all healpix pixels falling in an annulus around 
    ra-dec (deg) 
    the radii that define the annulus are in degrees    
    pixels are inclusive on radius_out but not radius_in
    """
    Nside, nest = hpx_meta['Nside'], hpx_meta['nest']
    pixels_in_disc = hp.query_disc(
        nside=Nside, nest=nest, 
        vec=hp.ang2vec(ra, dec, lonlat=True),
        radius = np.radians(radius_out_deg), 
        inclusive=inclusive
    )
    if radius_in_deg>0.:
        pixels_in_disc_in = hp.query_disc(
            nside=Nside, nest=nest, 
            vec=hp.ang2vec(ra, dec, lonlat=True),
            radius = np.radians(radius_in_deg), 
            inclusive=inclusive
        )
        id_annulus = np.isin(
            pixels_in_disc, 
            pixels_in_disc_in, 
            assume_unique=True, 
            invert=True
        )
        pixels_in_ann = pixels_in_disc[id_annulus]
    else:
        pixels_in_ann = np.copy(pixels_in_disc)

    return pixels_in_ann

def hpx_in_annulus (ra, dec, radius_in_deg, radius_out_deg, 
                    data_fp, hpx_meta, inclusive):
    """
    Given an array of healpix pixels (hpix, frac) where frac is the 
    covered fraction of each hpix pixel,
    computes the sub list of these pixels falling in an annulus around position 
    ra-dec (deg)
    the radii that define the annulus are in degrees
    hpx pixels are inclusive on radius_out but not radius_in
    """
    Nside, nest = hpx_meta['Nside'], hpx_meta['nest']
    hpix, frac = data_fp[hpx_meta['key_pixel']], data_fp[hpx_meta['key_frac']]

    area_pix = hp.nside2pixarea(Nside, degrees=True)
    pixels_in_ann = all_hpx_in_annulus (
        ra, dec, radius_in_deg, radius_out_deg, hpx_meta, inclusive
    )
    npix_all = len(pixels_in_ann)
    area_deg2 = 0.
    coverfrac = 0.
    hpx_in_ann, frac_in_ann = [], []

    if npix_all > 0:
        idx = np.isin(hpix, pixels_in_ann)
        hpx_in_ann = hpix[idx]  # visible pixels
        frac_in_ann = frac[idx] 
        npix = len(hpx_in_ann)
        if npix > 0:
            area_deg2 = np.sum(frac_in_ann) * area_pix
            coverfrac = np.sum(frac_in_ann)/float(npix_all)
    return hpx_in_ann, frac_in_ann, area_deg2, coverfrac


# FCT to split surveys 
def survey_ra_minmax(ra):
    """_summary_

    Args:
        ra (_type_): _description_

    Returns:
        _type_: _description_
    """
    ramin, ramax = np.amin(ra), np.amax(ra)
    if ramin<0.5 and ramax>359.5:
        nbins = 360
        hist, bin_edges = np.histogram(ra, bins=nbins, range=(0., 360))
        ramin_empty = bin_edges[np.amin ( np.argwhere(hist==0 ))]
        ramax_empty = bin_edges[np.amax ( np.argwhere(hist==0 ))]
        
        ra1 = ra[(ra<ramin_empty+1.)]
        ra2 = ra[(ra>ramax_empty-1.)]-360.
        ra_new = np.hstack((ra1, ra2))
        ramin, ramax = np.amin(ra_new), np.amax(ra_new)
    return ramin, ramax


def hpx_degrade(pix_in, frac_in, nside_in, nest_in, nside_out, nest_out):
    """_summary_

    Args:
        pix_in (_type_): _description_
        nside_in (_type_): _description_
        nest_in (_type_): _description_
        nside_out (_type_): _description_
        nest_out (_type_): _description_

    Returns:
        _type_: _description_
    """
    ra, dec = hp.pix2ang(nside_in, pix_in, nest_in, lonlat=True)
    pix_out0 = hp.ang2pix(nside_out, ra, dec, nest_out, lonlat=True)
    pix_out, counts = np.unique(pix_out0, return_counts=True)
    nsamp = (float(nside_in)/float(nside_out))**2
    return pix_out, counts.astype(float)/nsamp


def hpix_list_bary(hpix, Nside, nest):
        ra, dec = hp.pix2ang(Nside, hpix, nest, lonlat=True)
        dec_bary = np.mean(dec)
        if (np.amax(ra)-np.amin(ra))>180.:
            ra[ra>180.] = ra[ra>180.]-360.
            ra_bary = np.mean(ra)
            if ra_bary<0.:
                ra_bary = ra_bary+360.
        else:
            ra_bary = np.mean(ra)
        return ra_bary, dec_bary


def hpx_split_survey_equal_tiles (footprint_files, footprint, tiling):
    """_summary_

    Args:
        footprint_file (_type_): _description_
        footprint (_type_): _description_
        tiling (_type_): _description_

    Returns:
        _type_: _description_
    """
    Nside_fp  , nest_fp   = footprint['Nside'], footprint['nest']
    Nside_tile, nest_tile = tiling['Nside'], tiling['nest']

    hpix_tiles = np.array([]).astype('int')
    frac_tiles = np.array([])
    i=0
    for footprint_file in footprint_files:
        dat = read_FitsCat(footprint_file)
        hpix_map = dat[footprint['key_pixel']]
        frac_map = dat[footprint['key_frac']]
        hpix_tile, frac_tile = hpx_degrade(
            hpix_map, frac_map, Nside_fp, nest_fp, Nside_tile, nest_tile
        )
        hpix_tiles = np.hstack((hpix_tiles, hpix_tile))
        frac_tiles = np.hstack((frac_tiles, frac_tile))
        i+=1

    frac_unique = np.zeros(len(np.unique(hpix_tiles)))
    ra_bary = np.zeros(len(np.unique(hpix_tiles)))
    dec_bary = np.zeros(len(np.unique(hpix_tiles)))
    hpix_unique = np.unique(hpix_tiles)
    for i in range(0, len(hpix_unique)):
        frac_unique[i] = np.sum(
            frac_tiles[np.argwhere(hpix_tiles==hpix_unique[i])]
        )
        dat_fp_hpix = read_mosaicFootprint_in_hpix(
            footprint, hpix_unique[i], Nside_tile, nest_tile
        )

        ra_bary[i], dec_bary[i] = hpix_list_bary(
            dat_fp_hpix[footprint['key_pixel']], Nside_fp, nest_fp
        )
    return hpix_unique, frac_unique, ra_bary, dec_bary


def max_dist_to_parent_hpix(
        hpix_parent, hpix, Nside_tile, nest_tile,
        footprint, Nside_fp, nest_fp):

    dat_fp_hpix = read_mosaicFootprint_in_hpix(
        footprint, hpix, Nside_tile, nest_tile
    )
    ra, dec = hp.pix2ang(
        Nside_fp, 
        dat_fp_hpix[footprint['key_pixel']], 
        nest_fp, lonlat=True
    )
    rac_parent, decc_parent = hp.pix2ang(
        Nside_tile, hpix_parent, nest_tile, lonlat=True
    )
    dist2parent_hpix = np.degrees(
        dist_ang(ra, dec, rac_parent, decc_parent)
        )
    return np.amax(dist2parent_hpix)


def scan_survey_ra(hpix, Nside, nest):

    ra0_crossing = False
    ra_split = 0.

    ra, dec = hp.pix2ang(
        Nside, hpix, nest, lonlat=True 
    )
    size = 1.5*hp.nside2pixarea(Nside, degrees=True)**0.5
    ramin, ramax = np.amin(ra), np.amax(ra)
    if ramin < size and 360.-ramax < size:
        ra0_crossing = True
        coords = np.concatenate([[ra], [dec]]).T
        kmeans = cluster.KMeans(n_clusters=2, n_init=10)
        kmeans.fit(coords)
        labels = kmeans.labels_
        ra_split = (np.amax(ra[np.argwhere(labels==0).T])+\
                   np.amin(ra[np.argwhere(labels==1).T]))/2.
    return ra0_crossing, ra_split


def tile_center(hplist, Nside, nest):
    ra0_crossing, ra_split = scan_survey_ra(hplist, Nside, nest)
    ra, dec = hp.pix2ang(
        Nside, hplist, nest, lonlat=True 
    )
    deccen = np.mean(dec)

    if ra0_crossing:
        ra[ra>ra_split] = ra[ra>ra_split] - 360.
    racen = np.mean(ra)
    if racen < 0.:
        racen += 360.
    return racen, deccen


def tile_radius(hplist, Nside, nest):
    racen, deccen = tile_center(hplist, Nside, nest)
    ra, dec = hp.pix2ang(
        Nside, hplist, nest, lonlat=True 
    )
    dcen = np.degrees( 
        dist_ang(
            ra, dec, 
            racen, deccen
        )
    )
    half_pix_size = 0.5*1.45*hp.nside2pixarea(
            Nside, degrees=True
        )**0.5
    radius_deg = np.amax(dcen) + half_pix_size

    return radius_deg


def sky_partition(tiling, gdir, footprint, workdir):

    if tiling['ntiles'] > 0:
        print ('Sky partition for characterization')
    else:
        print ('Sky partition for detection')

    if os.path.isfile(
            os.path.join(
                workdir, tiling['rpath'],
                tiling['tiles_filename'])):
        ntiles = len(read_FitsCat(
            os.path.join(
                workdir, tiling['rpath'],
                tiling['tiles_filename'])))
        print ('.....Nr. of Tiles = ', ntiles)
        return ntiles
        
    overlap_deg = tiling['overlap_deg']
    Nside = tiling['Nside']
    nest = tiling['nest']
    mdir = footprint['mosaic']['dir']

    # read all pixels in survey
    raw_list = np.array(os.listdir(gdir))
    hpix_fits = np.array(
        [os.path.splitext(x)[0] for x in raw_list]
    ).astype(int)
    ra0_crossing, ra_split = scan_survey_ra(hpix_fits, Nside, nest)
    if tiling['ntiles'] > 0:
        ntiles = tiling['ntiles']
    else:
        # estimate the number of tiles from the desired tile area 
        tile_area = tiling['mean_area_deg2']
        area_pix = hp.nside2pixarea(Nside, degrees=True)
        npix = len(hpix_fits)
        ntiles = int(npix*area_pix/tile_area)
    print ('.....Nr. of Tiles = ', ntiles)

    
    # partition
    if not os.path.isfile(
            os.path.join(
                workdir, tiling['rpath'],
                tiling['sky_partition_npy'])):

        # partitionning with KMeans algorithm and save hpix's in npy
        hp_ra0, hp_dec0 = hp.pix2ang(
            Nside,
            hpix_fits,
            nest, 
            lonlat=True
        )
        if ra0_crossing:
            hp_ra0[hp_ra0>ra_split] = hp_ra0[hp_ra0>ra_split] - 360.
        hp_ra = hp_ra0*np.cos(np.radians(hp_dec0))
        hp_dec = hp_dec0
        coords = np.concatenate([[hp_ra], [hp_dec]]).T
        kmeans = cluster.KMeans(n_clusters=ntiles, n_init=10)
        kmeans.fit(coords)
        labels = kmeans.labels_
        partition=[]
        for i in range(0, len(np.unique(labels))):
            label = np.unique(labels)[i]
            hp_lab = hpix_fits[np.argwhere(labels==label).T[0]]
            partition.append(hp_lab)
        np.save(
            os.path.join(
                workdir, tiling['rpath'],
                tiling['sky_partition_npy']
            ), np.array(partition, dtype=object)
        )
        
        # ra-dec plot of the partition
        plt.clf()
        plt.figure(figsize=(8, 8))
        plt.scatter(
            hp_ra0, hp_dec0, c=labels, s=5, cmap=plt.cm.nipy_spectral
        )
        plt.xlabel('R.A. [deg]')
        plt.ylabel('Dec. [deg]')
        plt.title('Sky partinioning : '+str(ntiles)+' tiles')
        plt.savefig(os.path.join(
            workdir,tiling['rpath'], 'partition.png'
        ))


    # compute the tile overlaps and save lists of hpix's in npy
    if not os.path.isfile(os.path.join(
            workdir, tiling['rpath'], tiling['tiles_npy'])):

        # compute the number of layers for the overlap
        N_layers = 1+int(
            overlap_deg / hp.nside2pixarea(Nside, degrees=True)**0.5
        )
        overlap_eff_size = N_layers*1.42*hp.nside2pixarea(
            Nside, degrees=True
        )**0.5

        partition = np.load(
            os.path.join(
                workdir,tiling['rpath'],
                tiling['sky_partition_npy']), allow_pickle=True
        )
        hpix_tiles = []
        
        ntiles = len(partition)
        print ('......Nr. of Tiles = ', ntiles)

        for i in range(0, ntiles):

            if (ntiles>10):
                if (i % 10) == 0:
                    print ('......Tile ',i, ' / ', ntiles)
            else:
                print ('......Tile ',i, ' / ', ntiles)    
            hp_lab = np.array(partition[i]).astype(int)
            all_hp_neigh = np.array([]).astype(int)
            for j in range(0, N_layers):
                hp_tile = np.hstack((hp_lab, all_hp_neigh))
                all_hp = np.unique(
                    np.concatenate(
                        hp.get_all_neighbours(
                            Nside, hp_tile, nest=True
                        )
                    ).ravel()
                )
                hp_neigh_in_survey = all_hp[np.isin(all_hp, hpix_fits)]
                hp_neigh = hp_neigh_in_survey[np.isin(
                    hp_neigh_in_survey, hp_tile, invert=True
                )]
                all_hp_neigh = np.hstack((all_hp_neigh, hp_neigh))
                if tiling['plot_tiles']:
                    plot_tile(i, hp_lab, all_hp_neigh, hpix_fits, \
                              Nside, nest, overlap_eff_size,
                              os.path.join(workdir, tiling['rpath']))
            hpix_tiles.append(np.hstack((hp_lab, all_hp_neigh)))
        
        np.save(
            os.path.join(
                workdir,tiling['rpath'], tiling['tiles_npy']
            ), np.array(hpix_tiles, dtype=object)
        )

    # build tile fits file with effective areas racen, deccen
    partition = np.load(
        os.path.join(
            workdir, tiling['rpath'],
            tiling['sky_partition_npy']), allow_pickle=True
    )
    tiles = np.load(
        os.path.join(
            workdir,tiling['rpath'],
            tiling['tiles_npy']), allow_pickle=True
    )
    ntiles = len(partition)
    npix_core = np.zeros(ntiles).astype('int')
    npix_tile = np.zeros(ntiles).astype('int')
    area_core = np.zeros(ntiles)
    area_tile = np.zeros(ntiles)
    racen , deccen = np.zeros(len(partition)), np.zeros(len(partition))
    radius_deg = np.zeros(len(partition))

    for i in range(0, ntiles):
        hp_core = np.array(partition[i]).astype(int)
        hp_tile = np.array(tiles[i]).astype(int)
        npix_core[i] = len(hp_core)
        npix_tile[i] = len(hp_tile)
        racen[i], deccen[i] = tile_center(
            hp_tile, Nside, nest
        )
        radius_deg[i] = tile_radius(
            hp_tile, Nside, nest
        )
            
        for hh in hp_core:
            hpix, hfrac = read_FitsFootprint(
                os.path.join(mdir, str(hh)+'_footprint.fits'), footprint
            )
            area_core[i] += np.sum(hfrac)
        area_core[i] = area_core[i]*\
                       hp.nside2pixarea(
                           footprint['Nside'], degrees=True)
        for hh in hp_tile:
            hpix, hfrac = read_FitsFootprint(
                os.path.join(mdir, str(hh)+'_footprint.fits'), footprint
            )
            area_tile[i] += np.sum(hfrac)
        area_tile[i] = area_tile[i]*\
                       hp.nside2pixarea(
                           footprint['Nside'], degrees=True)

    # plot area distributions
    ##
    # write file
    data_tiles = np.zeros( ntiles, 
                           dtype={
                               'names':(
                                   'id', 
                                   'npix_core', 'npix_tile', 
                                   'area_core_deg2', 'area_tile_deg2',
                                   'ra', 'dec', 'radius_deg'
                               ),
                               'formats':(
                                   'i8', 
                                   'i8', 'i8', 
                                   'f8', 'f8',
                                   'f8', 'f8', 'f8'
                               )
                           }
    )
    data_tiles['id'] = np.arange(ntiles)
    data_tiles['npix_core'] = npix_core
    data_tiles['npix_tile'] = npix_tile
    data_tiles['area_core_deg2'] = area_core
    data_tiles['area_tile_deg2'] = area_tile
    data_tiles['ra'], data_tiles['dec'] = racen, deccen
    data_tiles['radius_deg'] = radius_deg
    t = Table(data_tiles)
    t.write(
        os.path.join(
            workdir, tiling['rpath'],
            tiling['tiles_filename']), 
        overwrite=True
    )
    return ntiles


def ra_shift(ra, ra0_crossing, ra360_crossing):
    if ra360_crossing:
        ra[ra<180.] = ra[ra<180.]+360.        
    if ra0_crossing:
        ra[ra>180.] = ra[ra>180.]-360.        
    return ra

def radec_window(racen, deccen, radius):
    ramin, ramax = racen - radius/np.cos(np.radians(deccen)),\
                   racen + radius/np.cos(np.radians(deccen)), 
    decmin, decmax = deccen - radius,\
                     deccen + radius 
    return ramin, ramax, decmin, decmax


def get_boundaries(pixel, Nside, nest, ra0_crossing, ra360_crossing):
    """
    get the ra-dec boundaries of a healpix pixel for graphic representation 
    """
    ra, dec = hp.vec2ang(hp.boundaries(Nside, pixel, 1, nest).T, lonlat=True)
    ra = ra_shift(ra, ra0_crossing, ra360_crossing)       
    return ra, dec


def draw_hpixels(pixels, Nside, nest, ra0_crossing, ra360_crossing, color, label):
    i=0
    for p in pixels[:]:
        if i==0:
            plt.fill(
                *get_boundaries(p, Nside, nest, 
                                ra0_crossing, ra360_crossing), 
                color=color, alpha=.1,zorder=2,
                label = label
            )
        else:
            plt.fill(
                *get_boundaries(p, Nside, nest, 
                                ra0_crossing, ra360_crossing), 
                color=color, alpha=.1,zorder=2
            )
        i+=1
    return


def plot_tile(
        tile_id, hp_lab, all_hp_neigh, hpix_fits, Nside, nest, osize, outdir
):

    create_directory(os.path.join(outdir, 'tile_plots'))

    hpix_fits_out = hpix_fits[np.isin(
        hpix_fits, np.hstack((hp_lab, all_hp_neigh)), invert=True
    )]
    ra_core, dec_core = hp.pix2ang(
        Nside, hp_lab, nest, lonlat=True 
    )

    racen, deccen = np.median(ra_core), np.median(dec_core)
    if (np.std(ra_core)>100.):
        ra_core[ra_core>=180.] = ra_core[ra_core>=180.]-360.
        racen = np.median(ra_core)

    radius = 1.+osize + 1.42*0.5*(len(hp_lab)*\
                 hp.nside2pixarea(Nside, degrees=True))**0.5
    ramin, ramax, decmin, decmax = radec_window(racen, deccen, radius)
    ramin_ext, ramax_ext, decmin_ext, decmax_ext = radec_window(
        racen, deccen, radius+1.
    )
    
    ra360_crossing = False
    ra0_crossing = False
    if ramax_ext > 360.:
        ra360_crossing = True
    if ramin_ext < 0.:
        ra0_crossing = True

    rai, deci = hp.pix2ang(Nside, hpix_fits_out, nest, lonlat=True)
    rai = ra_shift(rai, ra0_crossing, ra360_crossing)       
    mask = (rai<ramax) & (rai>ramin) & (deci<decmax) & (deci>decmin)
    hpix_all = hp.ang2pix(Nside, rai[mask], deci[mask], nest, lonlat=True)

    plt.clf()
    fig, ax = plt.subplots()
    ax.set_aspect(1./np.cos(np.radians(deccen)))

    # plot pixels of tile core
    draw_hpixels(
        hp_lab, Nside, nest, ra0_crossing, ra360_crossing, 'r', 'Core'
    )
    # plot pixels of tile overlap
    draw_hpixels(
        all_hp_neigh, Nside, nest, ra0_crossing, ra360_crossing, 'g', 'Overlap'
    )
    # plot pixels of surrounding survey pixels
    draw_hpixels(
        hpix_all, Nside, nest, ra0_crossing, ra360_crossing, 'b', 'Survey'
    )
    
    plt.title('Tile '+str(tile_id))
    plt.xlabel('R.A. [deg]', fontsize=20)
    plt.ylabel('Dec. [deg]', fontsize=20)
    plt.axis((ramin, ramax, decmin, decmax))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'tile_plots','tile_'+str(tile_id)+'.png'))
    plt.close()
    return


def disc_coverfrac(ra, dec, radius_deg, dat_footprint, footprint):
    pixels_in_disc = hp.query_disc(
        nside = footprint['Nside'], 
        nest = footprint['nest'],  
        vec = hp.ang2vec(ra, dec, lonlat=True),
        radius = np.radians(radius_deg), 
        inclusive=True
    )
    fhpx = dat_footprint[footprint['key_pixel']]
    ind_in = np.isin (pixels_in_disc, fhpx)
    return float(len(pixels_in_disc[ind_in])) / float(len(pixels_in_disc))


def create_tile_specs(target_mode, tiling, tile,
                      hpix_core_list, hpix_tile_list):
    """_summary_

    Args:
        tile (_type_): _description_
        tile_radius_deg (_type_): _description_
        admin (_type_): _description_

    Returns:
        _type_: _description_
    """

    hpix_core = hpix_core_list 
    hpix_tile = hpix_tile_list 
    Nside, nest = tiling['Nside'], tiling['nest']

    area_deg2 = float(tile['npix_core'])*\
        hp.nside2pixarea(Nside, degrees=True)
    eff_area_deg2 = tile['area_core_deg2']
    framed_eff_area_deg2 = tile['area_tile_deg2']
    tile_radius_deg = tile['radius_deg']

    tile_specs = {'id':tile['id'],
                  'ra': tile['ra'], 'dec': tile['dec'],
                  'hpix_tile': hpix_tile,
                  'hpix_core': hpix_core,
                  'Nside': Nside,
                  'nest': nest,
                  'area_deg2': area_deg2,
                  'eff_area_deg2': eff_area_deg2,
                  'framed_eff_area_deg2': framed_eff_area_deg2,
                  'radius_tile_deg': np.round(tile_radius_deg, 3)} 

    return tile_specs 


def create_ptile_specs(target, radius, 
                       search_radius, radius_unit,
                       dat_footprint, footprint):
    """_summary_
        Create specs of pointed tile towrds given target

    Args:
        tile (_type_): _description_
        tile_radius_deg (_type_): _description_
        admin (_type_): _description_

    Returns:
        _type_: _description_
    """

    # coverfrac in  30 and 5 arcmin
    coverfrac_30 = disc_coverfrac(
        target['ra'], target['dec'], 0.5, dat_footprint, footprint
    )
    coverfrac_5 = disc_coverfrac(
        target['ra'], target['dec'], 1./12., dat_footprint, footprint
    )
    area_deg2 = np.round(area_ann_deg2(0., radius), 3)
    eff_area_deg2 = disc_coverfrac(
        target['ra'], target['dec'], radius, dat_footprint, footprint
    )
    if radius_unit == 'arcmin':
        radius_filter_deg = search_radius/60.
    if radius_unit == 'mpc':
        radius_filter_deg = np.degrees(
            (search_radius / target['conv_factor'])
        )

    ptile_specs = {'id': 0,
                   'name': target['name'],
                   'ra': target['ra'], 'dec': target['dec'],
                   'area_deg2': area_deg2,
                   'eff_area_deg2': eff_area_deg2,
                   'framed_eff_area_deg2': eff_area_deg2,
                   'radius_tile_deg': np.round(radius, 3), 
                   'radius_filter_deg': np.round(radius_filter_deg, 3), 
                   'coverfrac_30arcmin': np.round(coverfrac_30, 2),
                   'coverfrac_5arcmin': np.round(coverfrac_5, 2)}

    return ptile_specs 


def cond_in_disc(rag, decg, hpxg, Nside, nest, racen, deccen, rad_deg):
    """_summary_

    Args:
        rag (_type_): _description_
        decg (_type_): _description_
        hpxg (_type_): _description_
        Nside (_type_): _description_
        nest (_type_): _description_
        racen (_type_): _description_
        deccen (_type_): _description_
        rad_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    pix_size = (hp.nside2pixarea(Nside, degrees=True))**0.5
    dist2cl = np.ones(len(rag))*2.*rad_deg

    pixels_in_disc = hp.query_disc(
        nside = Nside, nest=nest, 
        vec = hp.ang2vec(racen, deccen, lonlat=True),
        radius = np.radians(rad_deg), 
        inclusive=True
    )
    if pix_size > 0.1*rad_deg:
        cond = np.isin(hpxg, pixels_in_disc)
        dist2cl[cond] = np.degrees(
            dist_ang(
                rag[cond], decg[cond], racen, deccen
            )
        )
    else: 
        pixels_in_disc_strict = hp.query_disc(
            nside = Nside, nest=nest, 
            vec = hp.ang2vec(racen, deccen, lonlat=True),
            radius = np.radians(0.8*rad_deg), 
            inclusive = False
        )
        pixels_edge = pixels_in_disc[np.isin(
            pixels_in_disc, pixels_in_disc_strict, 
            invert=True, assume_unique=True
        )]
        cond_strict = np.isin(hpxg, pixels_in_disc_strict)
        cond_edge  =  np.isin(hpxg, pixels_edge)

        dist2cl[cond_strict] = 0.
        dist2cl[cond_edge] = np.degrees(
            dist_ang(
                rag[cond_edge], decg[cond_edge], racen, deccen
            )
        )
    return (dist2cl<rad_deg)


def cond_in_hpx_disc(hpxg, Nside, nest, racen, deccen, rad_deg):
    """_summary_

    Args:
        hpxg (_type_): _description_
        Nside (_type_): _description_
        nest (_type_): _description_
        racen (_type_): _description_
        deccen (_type_): _description_
        rad_deg (_type_): _description_

    Returns:
        _type_: _description_
    """

    pixels_in_disc_strict = hp.query_disc(
        nside=Nside, nest=nest, 
        vec=hp.ang2vec(racen, deccen, lonlat=True),
        radius = np.radians(rad_deg), 
        inclusive=False
    )
    cond_strict = np.isin(hpxg, pixels_in_disc_strict)
    return cond_strict


def normal_distribution_function(x):
    value = scipy.stats.norm.pdf(x,mean,std)
    return value


def compute_gaussian_kernel_1d(kernel):
# kernel is an integer >0
    mean = 0.0 
    kk = []
    for n in range (0,3*kernel+1):
        x1 = mean - 1./2. + float(n)
        x2 = mean + 1./2. + float(n)
        res, err = quad(normal_distribution_function, x1, x2)
        kk = np.append(kk, 100.*res)
    return np.array(np.concatenate((np.sort(kk)[0:len(kk)-1], kk)))


def get_gaussian_kernel_1d(kernel):

    if kernel == 1:
        gkernel = 0.01*np.array([0.60, 6.06, 24.17, 38.29, 24.17, 6.06, 0.60])
    if kernel == 2:
        gkernel = 0.01*np.array([ 0.24, 0.924, 2.783, 6.559, 12.098, \
                                  17.467, 19.741, 17.467, 12.098, 6.559, \
                                  2.783, 0.924, 0.24])
    if kernel == 3:
        gkernel = 0.01*np.array([ 0.153, 0.391, 0.892, 1.825, 3.343, 5.487, \
                                  8.066, 10.621, 12.528, 13.237, 12.528, \
                                  10.621, 8.066, 5.487, 3.343, 1.825, 0.892,\
                                  0.391, 0.153])
    if kernel > 3:
        gkernel = compute_gaussian_kernel_1d(kernel)
    return gkernel


def concatenate_clusters(tiles_dir, infilename, clusters_outfile): 
    """_summary_

    Args:
        tiles_dir (_type_): _description_
        clusters_outfile (_type_): _description_
    """
    # assumes that clusters are called 'clusters.fits'
    # and the existence of 'tile_info.fits'
    clist = []
    for tile_dir in tiles_dir: 
        clist.append(os.path.join(tile_dir, infilename))
    clcat = concatenate_fits(clist, clusters_outfile)
    return clcat


def concatenate_members(all_tiles, list_path_members, 
                        infilename, data_clusters, members_outfile):
    # data_clusters = clusters over the whole survey
    for it in range(0, len(all_tiles)):
        tile_id = int(all_tiles['id'][it])
        clusters_tile = data_clusters[data_clusters['tile'] == tile_id]
        clusters_id_in_tile = clusters_tile['index_cl_tile']
        members = read_FitsCat(
            os.path.join(list_path_members[it], infilename)
        )
        members_kept = members[np.isin(
            members['index_cl_tile'], clusters_id_in_tile
        )]
        # sort clusters by id_in_tile 
        ids = clusters_tile[np.argsort(clusters_id_in_tile)]['id']
        nmems = clusters_tile[np.argsort(clusters_id_in_tile)]['nmem']
        idd = clusters_tile[np.argsort(clusters_id_in_tile)]['index_cl_tile']
        # sort members by id_in_tile 
        members_kept_sorted = members_kept[np.argsort(
            members_kept['index_cl_tile']
        )]
        if it == 0:
            final_members = np.copy(members_kept_sorted)
        else:
            final_members = np.hstack((final_members, members_kept_sorted))
        for i in range(0, len(clusters_id_in_tile)):
            if it == 0 and i == 0:
                ids_for_members = ids[i]*np.ones(nmems[i]).astype(int)
                tile_for_members = tile_id*np.ones(nmems[i]).astype(int)
            else:
                ids_for_members = np.hstack(
                    (ids_for_members, ids[i]*np.ones(nmems[i]).astype(int))
                )
                tile_for_members = np.hstack(
                    (tile_for_members, tile_id*np.ones(nmems[i]).astype(int))
                )
    t = Table (final_members)
    t['id_cl'] = ids_for_members
    t['tile'] = tile_for_members
    t.write(members_outfile, overwrite=True)
    return


def tiles_with_clusters(out_paths, all_tiles, code):
    """_summary_

    Args:
        out_paths (_type_): _description_
        all_tiles (_type_): _description_

    Returns:
        _type_: _description_
    """
    workdir = out_paths['workdir']
    flag = np.zeros(len(all_tiles))
    for it in range(0, len(all_tiles)):
        tile_dir = os.path.join(
            workdir, 'tiles', 'tile_'+str(int(all_tiles['id'][it])).zfill(5)
        )
        if os.path.isfile(os.path.join(
                tile_dir,
                out_paths[code]['results'], "tile_info.fits"
        )):
            if read_FitsCat(os.path.join(
                    tile_dir, out_paths[code]['results'], "tile_info.fits"
            ))[0]['Nclusters'] > 0:
                flag[it] = 1
            else:
                print ('warning : no detection in tile ', tile_dir)
        else:
            print ('warning : missing tile ', tile_dir)
    return all_tiles[flag==1]


def concatenate_cl_tiles(out_paths, all_tiles, code):

    print ('Concatenate clusters')
    workdir = out_paths['workdir']

    eff_tiles = tiles_with_clusters(out_paths, all_tiles, code)

    list_clusters = []
    for it in range(0, len(eff_tiles)):
        tile_dir = os.path.join(
            workdir, 'tiles', 'tile_'+str(int(eff_tiles['id'][it])).zfill(5)
        )
        list_clusters.append(
            os.path.join(tile_dir, out_paths[code]['results'])
        )
    data_clusters0 = concatenate_clusters(
        list_clusters, 'clusters.fits', 
        os.path.join(workdir, 'tmp', 'clusters0.fits')
    )   
    return data_clusters0


def add_clusters_unique_id(data_clusters, clkeys):
    """_summary_

    Args:
        data_clusters (_type_): _description_
        clkeys (_type_): _description_

    Returns:
        _type_: _description_
    """
    # create a unique id 
    id_in_survey = np.arange(len(data_clusters))
    snr = data_clusters[clkeys['key_snr']]
    t = Table(data_clusters[np.argsort(-snr)])
    t['id'] = id_in_survey.astype('str')
    return t


def get_footprint(input_data_structure, footprint, workdir):

    if input_data_structure['footprint_hpx_mosaic']: 
        survey_footprint = os.path.join(
            workdir, 'footprint', 'survey_footprint.fits'
        )
        if not os.path.isfile(survey_footprint):
            create_survey_footprint_from_mosaic(
                footprint, survey_footprint
            )
    else:
        survey_footprint = footprint['survey_footprint']
    return survey_footprint


def update_data_structure(param_cfg, param_data):

    workdir = param_cfg['out_paths']['workdir']
    footprint = param_data['footprint']
    survey = param_cfg['survey']
    input_data_structure = param_data['input_data_structure']

    # create required data structure if not exist and update config 
    if not input_data_structure[survey]['footprint_hpx_mosaic']:
        create_mosaic_footprint(
            footprint[survey], os.path.join(workdir, 'footprint_mosaic')
        )
        param_data['footprint'][survey]['mosaic']['dir'] = os.path.join(
            workdir, 'footprint_mosaic'
        )

    return param_data

