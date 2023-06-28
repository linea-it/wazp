import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os, yaml
from astropy.cosmology.core import FlatLambdaCDM as flat
from astropy import units as u
from astropy.convolution import convolve,Gaussian1DKernel
import healpy as hp
from math import log, exp, atan, atanh
from astropy.table import Table
from scipy.optimize import least_squares
from scipy import interpolate
import math
import logging 
import time 

from .utils import mad, gaussian, dist_ang, _mstar_, join_struct_arrays
from .utils import all_hpx_in_annulus, hpx_in_annulus
from .utils import cond_in_disc, cond_in_hpx_disc
from .utils import get_gaussian_kernel_1d, create_directory
from .utils import read_mosaicFitsCat_in_disc, read_FitsCat, add_hpx_to_cat
from .utils import create_tile_specs, concatenate_clusters
from .utils import concatenate_members
from .utils import read_mosaicFootprint_in_disc, filter_hpx_tile
from .utils import filter_disc_tile, area_ann_deg2


def tile_dir_name(workdir, tile_nr):
    return os.path.join(workdir, 'tiles', 'tile_'+str(tile_nr).zfill(3))

def thread_dir_name(workdir, tile_nr):
    return os.path.join(workdir, 'threads', 'tile_'+str(tile_nr).zfill(3))


def create_pmem_directories(workdir, path):  # only called from pmem_main
    """
    creates the relevant directories for writing results/plots
    """
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    if not os.path.exists(os.path.join(workdir, path['results'])):
        os.mkdir(os.path.join(workdir, path['results']))
    if not os.path.exists(os.path.join(workdir, path['plots'])):
        os.mkdir(os.path.join(workdir, path['plots']))
    if not os.path.exists(os.path.join(workdir, path['files'])):
        os.mkdir(os.path.join(workdir, path['files']))

    return

def nfw_2D (R, Rc, Rs, h):
    """
     2D NFW density profile 
     taken from Rykoff 2012 - section 3.1 & Bartelmann 1996 A&A
     kNFW valid only for    Rs = 0.15/h # 0.214Mpc and Rcore = 0.1/h # 0.142Mpc
    """
    x = R / Rs
    if x > 1.:
        f = 1. - (2./(x**2-1.)**0.5) * atan( ((x-1.)/(x+1.))**0.5 )
    if x < 1.:
        f = 1. - (2./(1.-x**2)**0.5) * atanh( ((1-x)/(1.+x))**0.5 )
    if x == 1.:
        f = 0.

    rho = log(Rc)        
    kNFW = exp(
        1.6517-0.5479*rho+0.1382*rho**2-0.0719*rho**3-0.01582*rho**4-\
        0.00085499*rho**5
    ) # valid for given values of Rs and Rcore  / and 0.001<Rc<3.

    if x != 1.:
        Sigma = kNFW * f / (x**2-1.)
    else:
        Sigma = kNFW * 1./12.

    return Sigma


def nfw_2D_core (R, Rc, Rs, Rcore, h):
    """
     2D NFW density profile with a core 
     taken from Rykoff 2012 - section 3.1 & Bartelmann 1996 A&A
     kNFW valid only for    Rs = 0.15/h # 0.214Mpc and Rcore = 0.1/h # 0.142Mpc
    """
    
    Sigma = nfw_2D (R, Rc, Rs, h)
    Sigma_core = nfw_2D (Rcore, Rc, Rs, h)

    if R < Rcore:
        Sigma = Sigma_core

    return Sigma


def nfw_2D_core_array (R, Rc, Rs, Rcore, h):
    """
     2D NFW density profile computed for an array of radii "R" 
     taken from Rykoff 2012 - section 3.1 & Bartelmann 1996 A&A
     kNFW valid only for    Rs = 0.15/h # 0.214Mpc and Rcore = 0.1/h # 0.142Mpc
    """

    Sigma = np.zeros(len(R))

    i=0
    for rr in R:

        Sigma[i] = nfw_2D (rr, Rc, Rs, h)
        Sigma_core = nfw_2D (Rcore, Rc, Rs, h)

        if rr < Rcore:
            Sigma[i] = Sigma_core
        i+=1
    return Sigma


def get_boundaries(pixel, Nside, nest, racl, ramin, ramax):
    """
    get the ra-dec boundaries of a healpix pixel for graphic representation 
    """
    ra, dec = hp.vec2ang(hp.boundaries(Nside, pixel, 1, nest).T, lonlat=True)
    # deal with 0. and 360. crossings
    if racl>=0. and ramin<0.:
        ra1 = ra-360.
        ra[ra>180.] = ra1[ra>180.]
    if racl<=360. and ramax>360.:
        ra1 = ra+360.
        ra[ra<=180.] = ra1[ra<=180.]        
    return ra, dec
    
        
def get_circ(ra, dec, rad, npts):
    """
    get an array of ra-dec's to draw a circle
    rad is in arcmin
    the output ra-dec's are in degrees
    """
    t = np.linspace(0, 2*np.pi, npts)
    r = (rad/60.)

    return [r*np.cos(t)/np.cos(np.radians(dec))+ra, r*np.sin(t)+dec]


def hpix_in_discs(ra, dec, radius_deg, hpx_meta):
    """
    get the list of healpix pixels falling in discs centered on an 
    array of centers ra-deg (deg)
    """
    Nside, nest = hpx_meta['Nside'], hpx_meta['nest']
    ndisks = len(ra)
    all_pixels=np.zeros(0)
    for i in range(0, ndisks):
        pixels_in_disc = hp.query_disc(
            nside=Nside, nest=nest, 
            vec=hp.ang2vec(ra[i], dec[i], lonlat=True),
            radius = np.radians(radius_deg[i]), 
            inclusive=False
        )        
        all_pixels = np.concatenate((all_pixels, pixels_in_disc))
    pixels = np.unique(all_pixels)
    return pixels


def hpx_in_radec_window(hpx_meta, ramin0, ramax0, decmin, decmax):
    """
    computes the list of healpix pixels in a ra-dec (deg) window
    """
    Nside, nest = hpx_meta['Nside'], hpx_meta['nest']
    racen, deccen = (ramin0 + ramax0)/2., (decmin + decmax)/2.

    if racen < 0.:
        racen = 360. + racen
    if ramin0 < 0.:
        ramin = 360. + ramin0
    else:
        ramin = ramin0
    if ramax0 < 0.:
        ramax = 360. + ramax0
    else:
        ramax = ramax0

    theta_deg = np.degrees(dist_ang(racen, deccen, ramax, decmax))
    radius_deg = 1.1 * theta_deg
    pixels_in_disc = hp.query_disc(
        nside=Nside, 
        vec=hp.ang2vec(racen, deccen, lonlat=True),
        radius = np.radians(radius_deg), 
        inclusive=True, 
        nest=nest
    )
    ra0, dec0 = hp.pix2ang(Nside, pixels_in_disc, nest=nest, lonlat=True)
    ra1 = np.copy(ra0)
    if ramin0 < 0.:
        idx = np.argwhere(ra0>ramin).T[0]
        # idx = ra0>ramin  REMPLACER np.argwhere
        ra1[idx] = ra0[idx] - 360.
        ramin = ramin0
        ramax = ramax0
        
    hpx = pixels_in_disc\
          [(ra1<=ramax) & (ra1>=ramin) & (dec0<=decmax) & (dec0>=decmin)]
    return hpx 


def local_footprint(my_cluster, data_fp, hpx_meta, bkg_specs):
    """
    selects the pixels around the cluster in radius_max * 1.6 (to fill 
    radec window)
    Limits computation time foor each cluster 
    """
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    radius_deg = 1.6 * np.degrees(bkg_specs['radius_max_mpc'] / conv_factor)
    hpix_local, frac_local, area_deg2, coverfac = hpx_in_annulus (
        racl, deccl, 
        0., radius_deg, data_fp, hpx_meta, inclusive=True
    )
    data_lfp = np.zeros((len(hpix_local)), 
                        dtype={
                            'names':(
                                hpx_meta['key_pixel'], 
                                hpx_meta['key_frac']
                            ),
                            'formats':(
                                'i8', 'f8'
                            )
                        }
    )
    data_lfp[hpx_meta['key_pixel']] = hpix_local
    data_lfp[hpx_meta['key_frac']] = frac_local
    return data_lfp


def footprint_tile (hpx_meta, hpx_footprint):
    """
    Builds a healpix map for the used galcat tile
    It starts from a given footprint or not.
    ramin should be < ramax => can be <0 if needs be to solve 0-crossing
    """

    if hpx_footprint is not None: # read the hp map - may be larger than tile 
        hdulist=fits.open(hpx_footprint)
        dat = hdulist[1].data
        hdulist.close()
        hpix_map = dat[hpx_meta['key_pixel']].astype(int)
        if hpx_meta['key_frac'] is None:
            frac_map = np.ones(len(hpix_map0)).astype(float)
        else:
            frac_map = dat[hpx_meta['key_frac']]

    else: # build a map from corners of the tile 
        ramin, ramax, decmin, decmax = hpx_meta['ramin'],\
                                       hpx_meta['ramax'],\
                                       hpx_meta['decmin'],\
                                       hpx_meta['decmax'] 
        hpix_map = hpx_in_radec_window (hpx_meta, ramin, ramax, decmin, decmax)
        frac_map = np.ones(len(hpix_map))

    data_fp = np.zeros( (len(hpix_map)), 
                        dtype={
                            'names':(
                                hpx_meta['key_pixel'], 
                                hpx_meta['key_frac']
                            ),
                            'formats':(
                                'i8', 'f8'
                            )
                        }
    )
    data_fp[hpx_meta['key_pixel']] = hpix_map
    data_fp[hpx_meta['key_frac']] = frac_map

    return data_fp


def weighted_coverfrac_in_disk (my_cluster, data_fp, hpx_meta, 
                                weighted_coverfrac_specs, cosmo_params):
    """
    For a cluster defined by (racl, deccl, zcl), and given the 
    healpix visibility map (hpix, frac)
    where frac is the covered fraction of hpix, 
    compute the covered fraction of the cluster weighted by a 2D NFW 
    profile with parameters defined 
    in weighted_coverfrac_specs 
    """
    Nside, nest = hpx_meta['Nside'], hpx_meta['nest']
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    radius_test_deg = np.degrees(
        weighted_coverfrac_specs['radius_mpc'] / conv_factor
    )
    hpx_in_disc, frac_in_disc, area_deg2, coverfrac = hpx_in_annulus (
        racl, deccl, 0., radius_test_deg, 
        data_fp, hpx_meta, inclusive=False
    )
    pixels_in_fdisc = all_hpx_in_annulus (
        racl, deccl, 0., radius_test_deg, hpx_meta, inclusive=False
    )

    # full disc
    ra_in_fdisc, dec_in_fdisc = hp.pix2ang(
        Nside, pixels_in_fdisc, nest, lonlat=True
    )
    dist_fdisc_mpc = conv_factor * \
                     dist_ang(ra_in_fdisc, dec_in_fdisc, racl, deccl)

    # hpx disc
    ra_in_disc, dec_in_disc = hp.pix2ang(
        Nside, hpx_in_disc, nest, lonlat=True
    )
    dist_mpc = conv_factor * \
               dist_ang(ra_in_disc, dec_in_disc, racl, deccl)

    Rc = weighted_coverfrac_specs['radius_mpc'] 
    h = cosmo_params['H']/100.
    Rs = weighted_coverfrac_specs['Rs'] / h 
    Rcore = weighted_coverfrac_specs['Rcore'] / h 
    fweight = nfw_2D_core_array(dist_fdisc_mpc, Rc, Rs, Rcore, h)
    weight  = nfw_2D_core_array(dist_mpc      , Rc, Rs, Rcore, h)
    weighted_coverfrac = np.sum(np.multiply(frac_in_disc, weight))/\
                         np.sum(fweight)

    return weighted_coverfrac


def compute_cl_coverfracs(my_cluster, weighted_coverfrac_specs, 
                          data_fp, hpx_meta, cosmo_params):
    """
    compute the effective coverage of the cluster by the galcat
    First without weight and then weighted by NFW 2D profiles
    """
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    radius_testcl_deg = np.degrees(
        weighted_coverfrac_specs['radius_mpc'] / conv_factor
    )
    hpix, frac, testcl_area_deg2, testcl_coverfrac = hpx_in_annulus (
        racl, deccl, 0., radius_testcl_deg, 
        data_fp, hpx_meta, inclusive=False
    )
    weighted_coverfrac = weighted_coverfrac_in_disk (
        my_cluster, data_fp, hpx_meta, weighted_coverfrac_specs, cosmo_params
    )
    return testcl_coverfrac, weighted_coverfrac


def compute_bkg_coverfracs(bkg_specs, my_cluster, data_fp, data_fp_mask, 
                           hpx_meta):
    """
    compute the effective coverage of the bkg by the galcat
    without and with consideration of nearby clusters
    """
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    radius_bkg_max_deg = np.degrees(
        bkg_specs['radius_max_mpc'] / conv_factor
    )
    radius_bkg_min_deg = np.degrees(
        bkg_specs['radius_min_mpc'] / conv_factor
    )
    hpix, frac, bkg_area_deg2, bkg_coverfrac = hpx_in_annulus (
        racl, deccl, radius_bkg_min_deg, radius_bkg_max_deg, 
        data_fp, hpx_meta, inclusive=False
    )
    hpix, frac, bkg_wmask_area_deg2, bkg_wmask_coverfrac = hpx_in_annulus (
        racl, deccl, radius_bkg_min_deg, radius_bkg_max_deg, 
        data_fp_mask, hpx_meta, inclusive=False
    )
    return bkg_coverfrac, bkg_wmask_coverfrac, bkg_wmask_area_deg2


def plot_footprint(my_cluster, data_fp, hpx_meta, radius_bkg_min_mpc,
                   radius_bkg_max_mpc,radius_testcl_mpc, 
                   bkg_cfc, cl_cfc, cl_wcfc, output):
    """
    Plots the footprint at the location of the cluster including the local 
    background region
    """
    
    idcl = my_cluster['idcl']
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    radius_bkgmax_deg = np.degrees(radius_bkg_max_mpc / conv_factor)
    radius_bkgmin_deg = np.degrees(radius_bkg_min_mpc / conv_factor)
    radius_testcl_deg = np.degrees(radius_testcl_mpc / conv_factor)
    radius_deg = radius_bkgmax_deg

    ramin = racl - (radius_deg*1.1)/np.cos(np.radians(deccl))
    ramax = racl + (radius_deg*1.1)/np.cos(np.radians(deccl))
    decmin = deccl - (radius_deg*1.1)
    decmax = deccl + (radius_deg*1.1)

    nstep = 10 
    Nside_plot = compute_Nside_plot(hpx_meta, radius_deg, nstep)
    nest = hpx_meta['nest']
    data_fp_plot = degraded_footprint(data_fp, hpx_meta, Nside_plot)

    hpix, frac = data_fp_plot[hpx_meta['key_pixel']],\
                 data_fp_plot[hpx_meta['key_frac']]

    plt.clf()
    hpix10 = hpix[(frac>=0.9) ]
    hpix09 = hpix[(frac>=0.7) & (frac<0.9) ]
    hpix08 = hpix[(frac>=0.5) & (frac<0.7) ]

    ra10, dec10 = hp.pix2ang(Nside_plot, hpix10, nest, lonlat=True) 
    ra09, dec09 = hp.pix2ang(Nside_plot, hpix09, nest, lonlat=True) 
    ra08, dec08 = hp.pix2ang(Nside_plot, hpix08, nest, lonlat=True) 

    dra10, ddec10 = np.degrees(dist_ang(ra10, dec10, racl, dec10)),\
                    np.degrees(dist_ang(racl, dec10, racl, deccl))
    dra09, ddec09 = np.degrees(dist_ang(ra09, dec09, racl, dec09)),\
                    np.degrees(dist_ang(racl, dec09, racl, deccl))
    dra08, ddec08 = np.degrees(dist_ang(ra08, dec08, racl, dec08)),\
                    np.degrees(dist_ang(racl, dec08, racl, deccl))

    circle_bkg_max = get_circ(racl, deccl, 60.*radius_bkgmax_deg, 50)
    circle_bkg_min = get_circ(racl, deccl, 60.*radius_bkgmin_deg, 50)
    circle         = get_circ(racl, deccl, 60.*radius_testcl_deg, 50)
    
    # get pix inside region of interest
    pix10 = hpix10[(dra10<=(radius_deg*1.1))*(ddec10<=(radius_deg*1.1))]
    pix09 = hpix09[(dra09<=(radius_deg*1.1))*(ddec09<=(radius_deg*1.1))]
    pix08 = hpix08[(dra08<=(radius_deg*1.1))*(ddec08<=(radius_deg*1.1))]

    fig, ax = plt.subplots()
    ax.set_aspect(1./np.cos(np.radians(deccl)))

    for p in pix10[:]:
        plt.fill(
            *get_boundaries(p, Nside_plot, nest, racl, ramin, ramax), 
            color='r', alpha=0.6,zorder=1
        )
    for p in pix09[:]:
        plt.fill(
            *get_boundaries(p, Nside_plot, nest, racl, ramin, ramax), 
            color='r', alpha=.4,zorder=1
        )
    for p in pix08[:]:
        plt.fill(
            *get_boundaries(p, Nside_plot, nest, racl, ramin, ramax), 
            color='r', alpha=.2,zorder=1
        )

    plt.plot(*circle_bkg_max,color='b',zorder=8, label='bkg region')
    plt.plot(*circle_bkg_min,color='b',zorder=8)
    plt.plot(*circle,
             color='g',zorder=8,label=str(round(radius_testcl_mpc,1))+' Mpc'
    )
    plt.axis([ramax, ramin, decmin, decmax])
    plt.title(
        'cl. id = '+str(idcl).zfill(6)+'    coverfrac ( bkg / cl / cl+w ) = '+\
        str(round(bkg_cfc,2))+' / '+str(round(cl_cfc,2))+' / '+\
        str(round(cl_wcfc,2)), fontsize=8
    )
    plt.xlabel('R.A. [deg]')
    plt.ylabel('Dec. [deg]')
    plt.legend(loc=2)
    plt.savefig(output,bbox_inches='tight')

    return


def  plot_richness_vec (radius_vec_mpc, richness_vec, richness_err_vec, 
                        rfit, r200, idcl, workdir, out_paths):
    """
    Diagnostic plot to check the vectorial richness
    """
    workdir, path = out_paths['workdir_loc'], out_paths['pmem']
    slope = rfit[1]
    n200_pmem_cor = 10**(rfit(np.log10(r200)))
    plt.clf()
    plt.errorbar(radius_vec_mpc, richness_vec, richness_err_vec)
    plt.plot(
        radius_vec_mpc, 10**(rfit(np.log10(radius_vec_mpc))), 
        color='red', label = "linear fit / slope = "+str(round(slope,3))
    )
    plt.axvline(x=r200, 
                color='red', linestyle='--', 
                label = "$R_{200} =$"+str(round(r200, 2))+' Mpc'
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('R (Mpc)')
    plt.ylabel('Richness = $\sum{P_{mem}}$ ($\leq$ R)')
    plt.legend(loc=2)
    plt.title(
        'cluster id = '+str(idcl).zfill(6)+\
        '    N$_{200}=$'+str(round(n200_pmem_cor,2)), fontsize=10
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            workdir, path['plots'],
            'richness_vec_cl'+str(idcl)+'.png'
        ), dpi=300
    )
    return


def plot_pmems(my_cluster, data_gal, galcat_keys, pmem, r200, n200, out_paths):

    workdir, path = out_paths['workdir_loc'], out_paths['pmem']
    idcl = my_cluster['idcl']
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    size_deg = 1.5 * np.degrees(r200 / conv_factor)
    
    ramax = racl + size_deg/np.cos(np.radians(deccl))
    ramin = racl - size_deg/np.cos(np.radians(deccl))
    decmax = deccl + size_deg
    decmin = deccl - size_deg

    plt.clf()
    fig = plt.figure()    
    r1_deg = np.degrees(0.5 / conv_factor)
    r2_deg = np.degrees(1.0 / conv_factor)
    r3_deg = np.degrees(1.5 / conv_factor)
    r4_deg = np.degrees(2.0 / conv_factor)
    r200_deg = np.degrees(r200 / conv_factor)

    circle1         = get_circ(racl, deccl, 60.*r1_deg, 100)
    circle2         = get_circ(racl, deccl, 60.*r2_deg, 100)
    circle3         = get_circ(racl, deccl, 60.*r3_deg, 100)
    circle4         = get_circ(racl, deccl, 60.*r4_deg, 100)
    circle_r200     = get_circ(racl, deccl, 60.*r200_deg, 100)

    magp = data_gal[galcat_keys['key_mag']]

    plt.axvline(racl, color='black', linestyle='--', linewidth=0.5, zorder=1)
    plt.axhline(deccl, color='black', linestyle='--', linewidth=0.5, zorder=1)
    plt.plot(*circle1,
             color='black', linewidth=0.5, zorder=1, label='0.5 Mpc separation'
    )
    plt.plot(*circle2,color='black', linewidth=0.5, zorder=1)
    plt.plot(*circle3,color='black', linewidth=0.5, zorder=1)
    plt.plot(*circle4,color='black', linewidth=0.5, zorder=1)

    plt.plot(*circle_r200,
             color='red',  linestyle='--', linewidth=2, zorder=1, 
             label = 'R$_{200}=$'+str(round(r200, 2))+' Mpc'
    )
    plt.scatter(
        data_gal[galcat_keys['key_ra']], data_gal[galcat_keys['key_dec']], 
        c=pmem, s=20., alpha=0.5, cmap='jet', zorder=2
    )
    clb = plt.colorbar()
    clb.set_label('       P$_{mem}$', labelpad=-40, y=1.05, rotation=0)
    plt.xlabel('R.A. (deg)')
    plt.ylabel('Dec. (deg)')
    plt.title(
        'cluster id = '+str(idcl).zfill(6)+\
        '    N$_{200}=$'+str(round(n200,2)), fontsize=10, color='b'
    )
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis([ramax, ramin, decmin, decmax])

    ax = fig.add_subplot(111)
    ax.set_aspect(abs((ramax-ramin)/(decmax-decmin)), adjustable='box') 
    plt.legend(loc=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            workdir, path['plots'], 
            'pmems_cl'+str(idcl)+'.png'
        ), dpi=300
    )
    return


def sigma_dz0(sig_dz0, z):
    return (sig_dz0[0] + z*sig_dz0[1])


def filter_galcat(data_gal, galcat_keys, data_cluster, clcat_analysis_keys, 
                  photoz_support, sig_dz0, mstar_filename, pmem_specs):
    zclmax = max(data_cluster[clcat_analysis_keys['key_zp']])
    sig_dz0max = sigma_dz0(sig_dz0, zclmax)
    zpmax = zclmax +\
            photoz_support['nsig'] * sigma_dz0(sig_dz0, zclmax) * (1.+zclmax)
    mag = data_gal[galcat_keys['key_mag']]
    zp  = data_gal[galcat_keys['key_zp']]
    zpm = (zp + photoz_support['nsig']*sigma_dz0(sig_dz0, zclmax)) /\
          (1.-photoz_support['nsig']*sigma_dz0(sig_dz0, zclmax))
    magmax = _mstar_ (mstar_filename, zpm) + pmem_specs['dmagmax']
    cond = ((zp <= zpm) & (mag<=magmax)) 
    return data_gal[cond] 


def local_pdzcat(pdzcat, pdzcat_keys, pdz_specs, idg_list):
    """
    read external galcat and add hpx information 
    """
    zmin, zmax = pdz_specs['zmin'], pdz_specs['zmax']
    dz = pdz_specs['zstep']

    nzbin=int((zmax-zmin)/dz)+1
    zb_pdz = np.linspace(zmin,zmax,nzbin)

    hdulist=fits.open(pdzcat)
    dat=hdulist[1].data
    hdulist.close()
    
    idg = dat[pdzcat_keys["key_id"]]
    pdz = dat[pdzcat_keys["key_pdz"]]

    data_gpdz = pdz[np.isin(idg, idg_list)]
    return zb_pdz, data_gpdz


def init_richness_output(data_cls_analysis, richness_specs, 
                         clcat_analysis_keys):
    """
    initialize the table of richnesses for the whole list of analyzed clusters
    """
    #idform = clcat_analysis_keys["key_id_format"]
    npts = str(richness_specs['npts'])

    data_richness = np.zeros( (len(data_cls_analysis)), 
          dtype={
              'names':(
                  clcat_analysis_keys['key_id'], 
                  clcat_analysis_keys['key_ra'], 
                  clcat_analysis_keys['key_dec'], 
                  clcat_analysis_keys['key_zp'], 
                  clcat_analysis_keys['key_rank'], 
                  'nmem', 'mstar', 
                  'raw_coverfrac', 'weighted_coverfrac', 
                  'bkg_raw_coverfrac', 'bkg_coverfrac', 'ncl_los', 
                  'md_bkg_mpc2', 'md_bkg_arcmin2', 'r200_mpc', 
                  'r200_arcmin', 'slope_dprofile', 'slope_rad_over_rich', 
                  'n200', 'n200_pmem_raw', 'n200_pmem', 'n200_pmem_err', 
                  'radius_vec_mpc', 'radius_vec_arcmin', 
                  'richness', 'richness_err', 'flag_r200', 
                  'n500kpc_pmem', 'n500kpc_pmem_err', 'flag_pmem'
              ),
              'formats':(
                  'a30', 'f8', 'f8', 'f8', 'f8', 'i8', 'f8', 'f8', 
                  'f8', 'f8', 'f8', 'i8', 
                  'f8', 'f8', 'f8', 'f8', 
                  'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 
                  npts+'f8', npts+'f8', npts+'f8', 
                  npts+'f8', 'i8' , 'f8', 'f8', 'i8'
              ) 
          }
    )
    data_richness[clcat_analysis_keys['key_id']] = data_cls_analysis\
                                                   [clcat_analysis_keys\
                                                    ['key_id']].T
    data_richness[clcat_analysis_keys['key_ra']] = data_cls_analysis\
                                                   [clcat_analysis_keys\
                                                    ['key_ra']].T
    data_richness[clcat_analysis_keys['key_dec']] = data_cls_analysis\
                                                    [clcat_analysis_keys\
                                                     ['key_dec']].T
    data_richness[clcat_analysis_keys['key_zp']] = data_cls_analysis\
                                                   [clcat_analysis_keys\
                                                    ['key_zp']].T
    data_richness[clcat_analysis_keys['key_rank']] = data_cls_analysis\
                                                     [clcat_analysis_keys\
                                                      ['key_rank']].T
    data_richness['flag_pmem']  = np.zeros(len(data_cls_analysis)).astype(int)

    return data_richness


def init_members_output(nsat):
    """
    initialize the table of members for a given cluster
    """

    data_members = np.zeros( (nsat), 
                             dtype={
                                 'names':(
                                     'id_gal', 'id_cl', 
                                     'zcl', 
                                     'pmem', 'pmem_err', 
                                     'dist2cl_mpc', 'dist2cl_over_r200', 
                                     'ra', 'dec', 'zp', 'zp-zcl_over_sig_dz', 
                                     'mag', 'mag-mstar'),   
                                 'formats':(
                                     'a30', 'a30', 
                                     'f8', 
                                     'f8', 'f8', 
                                     'f8', 'f8', 'f8', 'f8', 
                                     'f8', 'f8', 'f8', 'f8'
                                 ) 
                             }
    )
    return data_members


def Gaussian_range_pos(x,eux,edx,dx,allx,sigma_unit=3.):
    """
    min/max index in the array allx for the variable x
    Inputs:
     - x = array of measurements
     - eux, edx = 1-sigma uncertainty
     - dx = increment in allx array
     - sigma_unit: intervals within [-sigma_unit;+sigma_unit] sigma 
    are considered
    """
    xpos = np.searchsorted(allx,x)-1
    deltad=sigma_unit*np.sqrt(dx**2.+edx**2.)
    deltau=sigma_unit*np.sqrt(dx**2.+eux**2.)

    dxd, dxu = np.int_((deltad/dx))+1, np.int_((deltau/dx))+1
    xstart, xend = xpos-dxd, xpos+dxu
    xstart[(xstart<0)]=0
    xend[(xend>=len(allx))]=len(allx)-1
    result = {'pos':xpos,'start pos':xstart,'end pos':xend}
    result = np.core.records.fromarrays(list(result.values()),\
                                        names=list(result.keys()))

    return result


def Gaussian_pdf(x,eux,edx,allx,dx,startpos,endpos):
    """ 
    estimates a Gaussian pdf for the observable x
      - x = observed value
      - eux, edx: 1sigma uncertainties
      - dx = increment in allx array
      - allx = xmin+ np.array(range(Nxbin+1))*dx -> x centered at 
    the LEFT edge of each bin (xmax included)
      - Nxbin = (xmax-xmin)/dx  
      - startpos, endpos: starting and ending positions in allx
    """
    xpos = list(range(startpos,endpos+1)) # pos where x should be considered 
    _x = allx[xpos]+dx/2.0     #+-3\sig centroid z considered 
    pdf = np.array([0.]*len(_x))    #aux pdf for x-value of AUXx 
    wd = (_x < x)             
    pdf[wd] = np.exp(-(_x[wd]-x)**2./(2.*(edx**2.+dx**2.)))# difference between bins above and below the obs value x
    pdf[~wd] = np.exp(-(_x[~wd]-x)**2./(2.*(eux**2.+dx**2.)))
    pdf = pdf*dx   #Integration over the bin
    if sum(pdf)!=0. : pdf = pdf/sum(pdf) # normalization equal to 1. Integration is truncated at 3\sigma
    result = {'pos':xpos,'pdf':pdf}
    result = np.core.records.fromarrays(
        list(result.values()), names=list(result.keys())
    )# dictionary => structured array
    return result


def Kernel_pdf(errx, dx, pos, vecmin, vecmax):
    """ 
    estimates a Gaussian pdf for the observable x
      - x = observed value
      - eux, edx: 1sigma uncertainties
      - dx = increment in allx array
      - allx = xmin+ np.array(range(Nxbin+1))*dx -> x centered at 
    the LEFT edge of each bin (xmax included)
      - Nxbin = (xmax-xmin)/dx  
      - startpos, endpos: starting and ending positions in allx
    """
    nbin = int((vecmax-vecmin)/dx)+1
    sig_in_pix = int( (errx**2 + dx**2)**0.5 / dx)
    gkernel = get_gaussian_kernel_1d(sig_in_pix)
    dim = int((len(gkernel)-1)/2)
    xpos = np.array(range(pos-dim,pos+dim+1)) # pos where x-vals are considered     
    result = {'pos':xpos[(xpos>=0) &\
                         (xpos<=nbin-1)],'pdf':gkernel[(xpos>=0) &\
                                                       (xpos<=nbin-1)]}
    result = np.core.records.fromarrays(list(result.values()), \
                                        names=list(result.keys()))
    return result


def mag_binning(vecbin_specs):
    """
    Generates the magnitude bins given the magbin_specs 
    """
    vecmin, vecmax = vecbin_specs['min'], vecbin_specs['max']
    step = vecbin_specs['step']
    nbin = int((vecmax-vecmin)/step)+1
    allvec = np.linspace(vecmin,vecmax,nbin)
    return allvec


def binned_counts_org(vec0, vecbin_specs, pmem_specs):
    """
    Sums up Gaussian pdfs for all vec0 quantities over a grid 
    defined by vecbin_specs
    Final counts are smoothed 
    It returns the bin values 
    """
    vecmin, vecmax = vecbin_specs['min'], vecbin_specs['max']
    step = vecbin_specs['step']
    #gkernel = pmem_specs['mag_counts_kernel']
    vec = vec0[((vec0>=vecmin-step) & (vec0<=vecmax+step))]
    evec = np.full(len(vec),step)
    nbin = int((vecmax-vecmin)/step)+1
    allvec = np.linspace(vecmin,vecmax,nbin)
    vrange_pos = Gaussian_range_pos(vec,evec,evec,step,allvec)
    mfd = np.zeros(nbin) 

    for l in range(len(vec)):
        pdf_vec = Gaussian_pdf(vec[l], evec[l], evec[l], allvec, step, 
                               vrange_pos['start pos'][l], 
                               vrange_pos['end pos'][l])
        imag_min=vrange_pos['start pos'][l]
        imag_max=vrange_pos['end pos'][l]
        mfd[ imag_min:imag_max+1 ] += pdf_vec['pdf']

    return allvec, mfd


def binned_counts(vec0, vecbin_specs, pmem_specs):
    """
    Sums up Gaussian pdfs for all vec0 quantities over a grid 
    defined by vecbin_specs
    Final counts are smoothed 
    It returns the bin values 
    Same as org but using a default pre computed kernel tp speed up
    """
    vecmin, vecmax = vecbin_specs['min'], vecbin_specs['max']
    step = vecbin_specs['step']
    vec= vec0[((vec0>=vecmin-step) & (vec0<=vecmax+step))]
    evec=np.full(len(vec),step)
    nbin = int((vecmax-vecmin)/step)+1
    allvec = np.linspace(vecmin,vecmax,nbin)
    mfd = np.zeros(nbin) 
    xpos = np.searchsorted(allvec, vec)-1

    hxpos, counts = np.unique(xpos, return_counts=True)
    for l in range(len(hxpos)):
        pdf_vec = Kernel_pdf(step, step, hxpos[l], vecmin, vecmax)
        imag_min = min(pdf_vec['pos'])
        imag_max = max(pdf_vec['pos'])
        mfd[ imag_min:imag_max+1 ] += float(counts[l])*pdf_vec['pdf']
        
    return allvec, mfd


def vec_bin_pos(vec, vecbin_specs):
    """
    From bin specs and given values "vec", returns the associated index's
    """
    vecmin, vecmax = vecbin_specs['min'], vecbin_specs['max']
    step = vecbin_specs['step']

    if isinstance(vec, np.ndarray):
        indf = (vec-vecmin)/step
        index = indf.astype(int)+1
    else:
        index = int((vec-vecmin)/step)+1
    return index


def rad_bin_pos(dist2cl, radial_bin_specs):
    """
    Given a list of distances from the cluster center, returns a list 
    of index's from 
    the radial binning described by radial_bin_specs
    """
    index = np.zeros(len(dist2cl))
    lradb = np.linspace(np.log10(radial_bin_specs['radius_min_mpc']),\
                        np.log10(radial_bin_specs['radius_max_mpc']),\
                        radial_bin_specs['nstep']) 

    lrad_step = lradb[1] - lradb[0]
    indf = (np.log10(dist2cl) - lradb[0] - lrad_step/2.)/lrad_step
    index = indf.astype(int)+1

    index[index<0] = 0
    index[index>(len(lradb)-1)] = len(lradb)-1
    return index


def bkg_mag_zp_counts(my_cluster, data_gal, galcat_keys, bkg_specs, 
                      mag_bin_specs, pmem_specs, richness_specs,
                      data_fp_mask, hpx_meta): 
    """
    Computes the galaxy mag counts in the bkg region 
    Returns the counts in /Mpc2  and /arcmin2
    magb are the mag bins 
    """
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    zpmin, zpmax = my_cluster['zpmin'], my_cluster['zpmax']
    mstar = my_cluster['mstar']
    magmax = mstar + richness_specs['dmag_faint'] 
    hpix_gal = hp.ang2pix(
        hpx_meta['Nside'],
        data_gal[galcat_keys['key_ra']],
        data_gal[galcat_keys['key_dec']],
        hpx_meta['nest'],
        lonlat=True
    )
    radius_bkg_max_deg = np.degrees(bkg_specs['radius_max_mpc'] / conv_factor)
    radius_bkg_min_deg = np.degrees(bkg_specs['radius_min_mpc'] / conv_factor)
    hpx_bkg_map, frac_bkg_map, area_bkg_deg2, coverfrac = hpx_in_annulus (
        racl, deccl, radius_bkg_min_deg, 
        radius_bkg_max_deg, data_fp_mask, 
        hpx_meta, 
        inclusive=False
    )
    in_bkg = np.isin(hpix_gal, hpx_bkg_map)     # galaxies in bkg region 

    magb, counts_bkg = binned_counts(
        data_gal[galcat_keys['key_mag']][in_bkg], mag_bin_specs, pmem_specs
    )
    counts_bkg_arcmin2 = counts_bkg / (3600.*area_bkg_deg2)
    area_bkg_mpc2 = area_bkg_deg2 * np.radians(conv_factor)**2
    counts_bkg_mpc2 = counts_bkg / area_bkg_mpc2

    # density f=(zp) along the zp support of the studied cluster
    mag = data_gal[galcat_keys['key_mag']][in_bkg]
    zps0 = data_gal[galcat_keys['key_zp']][in_bkg]
    zps = zps0[(mag < magmax) & (zps0<zpmax) & (zps0>zpmin)]
    npts = 32
    xx, density = density_fct(zps, zpmin, zpmax, npts) 

    """
    # plot to check density fct 
    plt.clf()
    nz, bins, patches = plt.hist(zps, npts,range=(zpmin,zpmax), density=False,facecolor='g', alpha=0.6)
    smoo = convolve(nz, gauss_kernel, boundary='extend')
    plt.plot(xx,smoo,'-',color='red')
    plt.axis([zpmin,zpmax,0.,max(nz)*1.1])
    plt.xlabel('redshift')
    plt.ylabel('counts')
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig('zphist_'+str(my_cluster['idcl'])+'.png',dpi=200)
    print ('sum2 ', np.sum(nz))
    print (len(zps))
    """
    return magb, counts_bkg_mpc2, counts_bkg_arcmin2, xx, density


def density_fct(data, datamin, datamax, npts): 
    histo, bins = np.histogram(
        data, bins=npts, 
        range=(datamin,datamax), 
        weights=None, density=True
    )
    xx = (bins[0:len(bins)-1] + bins[1:len(bins)])/2. 
    gauss_kernel = Gaussian1DKernel(2)
    density = convolve(histo, gauss_kernel, boundary='extend')
    return xx, density


def mean_bkg(my_cluster, counts_bkg_mpc2, counts_bkg_arcmin2, 
            mag_bin_specs, richness_specs): 
    """
    computes the mean density in /mpc2 and /arcmin2 by integrating 
    counts_bkg in the mag range defined for richness computation 
    """
    mstar = my_cluster['mstar']
    magmin = mstar - richness_specs['dmag_bright']
    magmax = mstar + richness_specs['dmag_faint']
    if magmin < mag_bin_specs['min']:
        magmin = mag_bin_specs['min']
    if magmax > mag_bin_specs['max']:
        magmax = mag_bin_specs['max']

    md_bkg_mpc2 = np.sum(counts_bkg_mpc2[vec_bin_pos(magmin, mag_bin_specs): \
                                         vec_bin_pos(magmax, mag_bin_specs)])
    md_bkg_arcmin2 = np.sum(
        counts_bkg_arcmin2[vec_bin_pos(magmin, mag_bin_specs):  \
                           vec_bin_pos(magmax, mag_bin_specs)]
    )
    return md_bkg_mpc2, md_bkg_arcmin2 


def build_my_cluster (index_cl, data_cluster, clcat_analysis_keys, 
                      photoz_support, sig_dz0, mag_bin_specs, pmem_specs, 
                      richness_specs, mstar_filename, cosmo_params):
    """
    """
    idcl = data_cluster[clcat_analysis_keys['key_id']]
    racl, deccl = data_cluster[clcat_analysis_keys['key_ra']],\
                  data_cluster[clcat_analysis_keys['key_dec']]
    zcl = data_cluster[clcat_analysis_keys['key_zp']]

    ext_radius = None
    if richness_specs['external_radius']:
        ext_radius = data_cluster[clcat_analysis_keys['key_radius']]

    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
    conv_factor = cosmo.angular_diameter_distance(zcl)# radian*conv=mpc    

    mstar = _mstar_ (mstar_filename, zcl)
    imag_star = vec_bin_pos(mstar, mag_bin_specs)

    zpmin = zcl - photoz_support['nsig']*sigma_dz0(sig_dz0, zcl)*(1.+zcl)
    zpmax = zcl + photoz_support['nsig']*sigma_dz0(sig_dz0, zcl)*(1.+zcl)
    sig_dz = sigma_dz0(sig_dz0, zcl)*(1.+zcl)
    magmin = mstar - pmem_specs['dmagmin']
    magmax = mstar + pmem_specs['dmagmax']

    if magmin < mag_bin_specs['min']:
        magmin = mag_bin_specs['min']
    if magmax > mag_bin_specs['max']:
        magmax = mag_bin_specs['max']
        
    slope_dprofile = pmem_specs['slope_dprofile_poly'][0] +\
                     pmem_specs['slope_dprofile_poly'][1]*zcl       

    my_cluster = {'idcl':str(idcl),
                  'index_cl':index_cl,
                  'racl': np.float64(racl), 'deccl': np.float64(deccl),
                  'zcl': zcl,
                  'snr_cl': data_cluster[clcat_analysis_keys['key_rank']],
                  'mstar': mstar, 'imag_star': imag_star,
                  'conv_factor': conv_factor.value,
                  'zpmin': zpmin, 'zpmax': zpmax,
                  'sig_dz': sig_dz,
                  'magmin': magmin, 'magmax': magmax,
                  'slope_dprofile': slope_dprofile,
                  'external_radius': ext_radius }
                  #'slope_dprofile': pmem_specs['slope_dprofile']}

    return my_cluster


def profile_mpc(my_cluster, data_gal, galcat_keys, magmin, magmax, 
                mag_bin_specs, pmem_specs, md_bkg_2d, radius, 
                data_fp, hpx_meta):
    """
    Computes the 2d (in discs of radius R) and 3d (in cylinders of 
    radius R and length 2R) 
    cumulative galaxy density profiles around the cluster center 
    racl-deccl (deg) for an array of radii 
    "radius" in Mpc.
    The computation takes into account missing data or masked 
    neighbouring clusters in the effective 
    areas through healpix maps (hpix_map, frac_map).   
    """
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    mag = data_gal[galcat_keys['key_mag']]
    dist2cl_mpc = data_gal['dist2cl_mpc']
    hpix = hp.ang2pix(
        hpx_meta['Nside'],
        data_gal[galcat_keys['key_ra']],
        data_gal[galcat_keys['key_dec']], 
        hpx_meta['nest'],
        lonlat=True
    )

    md_cl_2d = np.zeros(len(radius))
    md_cl_3d = np.zeros(len(radius))

    for i in range(len(radius)):
        radius_cl_max_mpc = radius[i]
        if i == 0:
            radius_previous = 0.
        else:
            radius_previous = radius[i-1]
        radius_cl_max_deg = np.degrees(radius_cl_max_mpc / conv_factor)
        hpix, frac, area, annulus_area_fraction = hpx_in_annulus (
            racl, deccl, 
            0., radius_cl_max_deg, data_fp, 
            hpx_meta,inclusive=False
        )
        area_cl_mpc2 = area_ann_deg2(0., radius_cl_max_mpc)
        ind_cl = ((dist2cl_mpc< radius_cl_max_mpc) &\
                  (dist2cl_mpc>=radius_previous))
        magb, cld_ = binned_counts(mag[ind_cl], mag_bin_specs, pmem_specs)
        if annulus_area_fraction>0. and pmem_specs['area_correction']:
            cld_ = cld_/annulus_area_fraction
        if i == 0:
            cl_ = cld_
        else:
            cl_ += cld_
        cl_mpc2 = cl_/area_cl_mpc2
        cl_dens = np.sum(
            cl_mpc2[vec_bin_pos(magmin, mag_bin_specs):\
                    vec_bin_pos(magmax, mag_bin_specs)])
        md_cl_2d[i] = (cl_dens-md_bkg_2d)
        md_cl_3d[i] =  md_cl_2d[i] /(2.*radius_cl_max_mpc)

    return md_cl_2d, md_cl_3d # densities in /mpc2 and /mpc3


def clusters_periphery(my_cluster, data_cls, clcat_keys, periphery_specs):

    racl, deccl, zcl = my_cluster['racl'],\
                       my_cluster['deccl'],\
                       my_cluster['zcl']
    conv_factor = my_cluster['conv_factor']
    sig_dz = my_cluster['sig_dz']

    nsig_msk, nsig_los = periphery_specs['nsig_msk'],\
                         periphery_specs['nsig_los']
    radius_msk_mpc, rad_los_mpc = periphery_specs['radius_msk_mpc'],\
                                  periphery_specs['rad_los_mpc']
    radmin_msk_mpc, radmax_msk_mpc = periphery_specs['radmin_msk_mpc'],\
                                     periphery_specs['radmax_msk_mpc']

    ra, dec, z = data_cls[clcat_keys['key_ra']],\
                 data_cls[clcat_keys['key_dec']],\
                 data_cls[clcat_keys['key_zp']]
    dist2cl_mpc = conv_factor * dist_ang(ra, dec, racl, deccl)
    cond_z_msk = (np.absolute(z-zcl)<= nsig_msk*sig_dz)
    cond_z_los = (np.absolute(z-zcl)<= nsig_los*sig_dz)
    cond_rad_msk = ( (dist2cl_mpc <= (radius_msk_mpc+radmax_msk_mpc)) &\
                     (dist2cl_mpc >= (radius_msk_mpc+radmin_msk_mpc)))
    cond_rad_los = (dist2cl_mpc <= rad_los_mpc)
    data_cls_peri = data_cls[(cond_z_msk & cond_rad_msk)]
    data_cls_los =  data_cls[(cond_z_los & cond_rad_los)]
    return data_cls_peri, data_cls_los


def footprint_with_cl_masks (my_cluster, data_cls, clcat_keys, 
                             periphery_specs, data_fp, hpx_meta):
    """
    build a healpix map for computing galaxy density profiles.
    The healpix map is derived from the original visibility map with additional 
    holes at the position of neighbouring clusters 
    """
    hpix_map, frac_map = data_fp[hpx_meta['key_pixel']],\
                         data_fp[hpx_meta['key_frac']]
    conv_factor = my_cluster['conv_factor']

    cl_mask_radius_mpc = periphery_specs['radius_msk_mpc']
    snr = data_cls[clcat_keys[periphery_specs['key_select']]]
    cond_snr = (snr>periphery_specs['select_min'])

    if len(data_cls[cond_snr]) >0:
        # select clusters falling in cl outer region 
        data_cls_msk, data_cls_los = clusters_periphery(
            my_cluster, data_cls[cond_snr], clcat_keys, periphery_specs
        )
        racl_out, deccl_out = data_cls_msk[clcat_keys['key_ra']],\
                              data_cls_msk[clcat_keys['key_dec']]
        ncl_masked = len(racl_out)
        if(len(racl_out))>0:
            radius_clmask_deg = np.full(
                len(racl_out), 
                np.degrees(cl_mask_radius_mpc / conv_factor)
            )
            hpix_clmask = hpix_in_discs(
                racl_out, deccl_out, radius_clmask_deg, hpx_meta
            )
            ind_out = (np.isin(
                hpix_map, hpix_clmask, invert=True
            ))     # galaxies in bkg region and not in outside clusters 
            hpix_map_with_clmask = hpix_map[ind_out]
            frac_map_with_clmask = frac_map[ind_out]
        else:
            hpix_map_with_clmask = np.copy(hpix_map)
            frac_map_with_clmask = np.copy(frac_map)

    else:
        hpix_map_with_clmask = np.copy(hpix_map)
        frac_map_with_clmask = np.copy(frac_map)
        ncl_masked = 0

    data_lfp_mask = np.zeros( (len(hpix_map_with_clmask)), 
                              dtype = {
                                  'names': (
                                      hpx_meta['key_pixel'], 
                                      hpx_meta['key_frac']
                                  ),
                                  'formats':(
                                      'i8', 'f8'
                                  )
                              }
    )
    data_lfp_mask[hpx_meta['key_pixel']] = hpix_map_with_clmask
    data_lfp_mask[hpx_meta['key_frac']] = frac_map_with_clmask

    return data_lfp_mask, ncl_masked 


def compute_r200_n200(my_cluster, data_gal, galcat_keys, bkg_mpc2,  
                      data_fp_mask, hpx_meta, pmem_cfg, cosmo_params, 
                      verbose, out_paths):
    """
    Compute the radius and richness corresponding to 
    (Ngal_in_cl / Ngal_in_bkg)_3D = Delta_critical - equivalent to R200-N200 
    but with Delta_c evolving with z as in LambdaCDM (from Eke et al. paper)

    returns 
    - R200 
    - N200 : corrected for holes 
    - the slope of the relation 
    log10 (Ngal_in_cl / Ngal_in_bkg)_3D = slope_3d * log10 (Radius) 
    slope_3d : slope of the 3D cumulative density profile 
    slope_dprofile : slope of the 2D density profile == slope_3d 
    - a boolean to indicate a failure - which will stop the analysis of the cluster
    - a flag if r200 is found smaller than minimum r200
    """
    workdir, path = out_paths['workdir_loc'], out_paths['pmem']
    r200_specs = pmem_cfg['r200_specs']
    radial_bin_specs = pmem_cfg['radial_bin_specs']
    mag_bin_specs = pmem_cfg['mag_bin_specs']
    pmem_specs = pmem_cfg['pmem_specs']
    richness_specs = pmem_cfg['richness_specs']

    flag_r200 = 0
    gauss_kernel = Gaussian1DKernel(1)

    idcl = my_cluster['idcl']
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    zcl = my_cluster['zcl']
    conv_factor = my_cluster['conv_factor']
    mstar = my_cluster['mstar']
    zpmin, zpmax = my_cluster['zpmin'], my_cluster['zpmax']
    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])

    lradb = np.linspace(np.log10(radial_bin_specs['radius_min_mpc']),\
                        np.log10(radial_bin_specs['radius_max_mpc']),\
                        radial_bin_specs['nstep']) 
    radius_test = 10**lradb

    om = cosmo.Om(zcl)
    Delta = (18.*np.pi**2 + 82.*(om-1.) - 39.*(om-1.)**2) / om 
    # see Eke et al. 1996   + Bryan & Norman 1998 / Mo et al CUP

    magmin, magmax = mstar - richness_specs['dmag_bright'],\
                     mstar + richness_specs['dmag_faint']
    if magmin < mag_bin_specs['min']:
        magmin = mag_bin_specs['min']
    if magmax > mag_bin_specs['max']:
        magmax = mag_bin_specs['max']

    md_bkg_2d = np.sum(
        bkg_mpc2[vec_bin_pos(magmin, mag_bin_specs):\
                 vec_bin_pos(magmax, mag_bin_specs)]
    )
    md_bkg_3d = md_bkg_2d/\
                cosmo.angular_diameter_distance_z1z2(zpmin, zpmax).value
    r200, n200, flag_r200 = 0., 0., 0
    slope_3d_mean = my_cluster['slope_dprofile']-1.
    no_failure = True

    if md_bkg_2d > 0.:
        md_cl_2d, md_cl_3d = profile_mpc(
            my_cluster, data_gal, galcat_keys, 
            magmin, magmax, mag_bin_specs, pmem_specs, 
            md_bkg_2d, radius_test, data_fp_mask, hpx_meta
        )
        ratio = md_cl_3d / md_bkg_3d

        # compute r200 without fit
        cond_init = ((ratio>0.))
        if len(ratio[cond_init]) > 2:
            xtrain, ytrain = np.log10(radius_test[cond_init]),\
                             np.log10(ratio[cond_init])
            smoo = convolve(ytrain, gauss_kernel, boundary='extend')
            fradius = interpolate.interp1d(
                smoo, xtrain, 
                kind = 'linear', 
                bounds_error=False, fill_value='extrapolate'
            )
            r200_init = 10**(fradius(np.log10(Delta)))
            cond_fit      = ((ratio>0.) & (radius_test <= r200_init))

            if len(radius_test[cond_fit]) > 1:
                xtrain, ytrain = np.log10(radius_test[cond_fit]),\
                                 np.log10(ratio[cond_fit])
                rfit = np.poly1d(np.polyfit(xtrain, ytrain, 1))
                y0 = rfit[0] 
                slope_3d = rfit[1]

                if pmem_specs['dprofile_forced']:
                    slope_3d = slope_3d_mean
                    y0 = np.mean(ytrain - slope_3d*xtrain)
                    rfit[0] = y0
                    rfit[1] = slope_3d 
                else:
                    if (slope_3d < pmem_specs['slope_free_min']-1.) or\
                       (slope_3d > pmem_specs['slope_free_max']-1.): 
                        slope_3d = slope_3d_mean
                        y0 = np.mean(ytrain - slope_3d*xtrain)
                        rfit[0] = y0
                        rfit[1] = slope_3d 

                r200_fit = 10**((np.log10(Delta)-y0)/slope_3d)
                r200 = min(r200_init, r200_fit)

                if r200 < r200_specs['min_mpc']:
                    r200 = r200_specs['min_mpc']
                    flag_r200 = 1

                if verbose >= 2:
                    plot_galdens_3d (
                        radius_test, ratio, rfit, 
                        os.path.join(
                            workdir, path['plots'], 
                            'dens_ratio_cl'+str(idcl)+'.png'
                        )
                    )
            else:
                r200 = r200_specs['min_mpc']
                flag_r200 = 1
                slope_3d = slope_3d_mean
        else:
            r200 = r200_specs['min_mpc']
            flag_r200 = 1
            slope_3d = slope_3d_mean
            no_failure = False # no overdensity at any radius_test 
   
        cl_dens_r200 = np.interp(r200, radius_test, md_cl_2d)
        r200_deg = np.degrees(r200 / conv_factor)
        n200 = cl_dens_r200 * np.pi * r200**2
        my_cluster['slope_dprofile'] = slope_3d + 1.

    return r200, n200, no_failure, flag_r200


def plot_mag_counts(my_cluster, magb, bkg_arcmin2, 
                    pmem_specs, 
                    data_gal, galcat_keys, radial_bin_specs, mag_bin_specs,
                    data_fp_mask, hpx_meta,
                    output):
    """
    Creates a plot displaying 
       - the galaxy density in the cluster in the annulus (rad1, rad2) as a fct of mag
       - the galaxy density in the bkg as a fct of mag
    """
    cl_local_overdensity, cl_local_cor, cl_local = cl_density_ratio_drad_dmag( 
        my_cluster, data_gal, galcat_keys,  
        radial_bin_specs, mag_bin_specs, 
        pmem_specs, data_fp_mask, hpx_meta, bkg_arcmin2
    )

    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    magmin, magmax = my_cluster['magmin'], my_cluster['magmax']
    mstar = my_cluster['mstar']

    imag_star = my_cluster['imag_star']

    rad1, rad2 = 0., pmem_specs['radius_densnorm']
    radius_cl_min_deg = np.degrees(rad1 / conv_factor)
    radius_cl_max_deg = np.degrees(rad2 / conv_factor)
    cond1 = ((magb > magmin) & (magb < magmax) )

    plt.clf()
    plt.plot(
        magb[cond1],bkg_arcmin2[cond1],
        linestyle='-',  color='blue' , label = 'background'
    )
    plt.plot(
        magb[cond1],cl_local_cor[cond1],
        linestyle='--',  color='green' , label = 'cluster - effective'
    )
    plt.scatter(
        magb[cond1],cl_local[cond1],  
        color='green' , 
        label = 'cluster (R$\leq$ '+str(np.round(rad2, 2))+' Mpc)'
    )
    plt.axvline(x=mstar, color='red', linestyle='--', label = "$m^*$")
    plt.xlabel('$mag$')
    plt.ylabel('N (/arcmin$^2$)')
    plt.legend(loc=2)
    plt.tight_layout()
    plt.yscale("log")
    plt.axis([magmin, magmax, 10**(-6.), max(cl_local_cor) *5])
    plt.savefig(output, dpi=300)
    return


def plot_galdens_3d (radius_test, ratio, rfit, output):

    slope = rfit[1]
    plt.clf()
    plt.scatter(radius_test, ratio)
    plt.plot(
        radius_test, 10**(rfit(np.log10(radius_test))), 
        color='red', label = "linear fit / slope = "+str(round(slope,3))
    )
    plt.xlabel("radius (Mpc)")
    plt.ylabel("cl/bkg 3d")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.legend(loc=2)
    plt.savefig(output, dpi=300)
    return


def plot_raw_mag_counts(magb, bkg_arcmin2, cl_arcmin2, cl_cor, mstar, 
                        magmin, magmax, output):
    """
    Creates a plot displaying the galaxy density in the cluster and in the bkg.
    Also is displayed the corrected galaxy density in the cluster to avoid fluctuations at bright magnitudes
    """
    cond1 = ((magb > magmin) & (magb < magmax) )

    xmin, xmax = 0.95* min(magb[cond1]), 1.05*max(magb[cond1])
    ymin, ymax = 0.90*min(bkg_arcmin2[cond1]), 10.*max(bkg_arcmin2[cond1])

    plt.clf()
    plt.plot(
        magb[cond1],bkg_arcmin2[cond1],
        linestyle='-',  color='blue' , label = 'backgrountled'
    )
    plt.plot(
        magb[cond1],cl_arcmin2[cond1],
        linestyle='--',  color='red' , label = 'cluster'
    )
    plt.plot(
        magb[cond1],cl_cor[cond1],
        linestyle='-',  color='red' , label = 'cluster cor'
    )
    plt.axvline(x=mstar, color='red', linestyle='--', label = "$m^*$")
    plt.xlabel('$mag$')
    plt.legend(loc=2)
    plt.tight_layout()
    plt.yscale("log")
    plt.axis([xmin, xmax, ymin, ymax])
    plt.savefig(output, dpi=300)
    return


def recenter_zp(my_cluster, zp):
    zpmed = np.median(zp)
    return zpmed


def prepare_data_calib_dz(my_cluster, pmem_cfg, hpx_meta, 
                          data_gal, galcat_keys):

    calib_cfg = pmem_cfg['calib_dz']
    dmagmin = pmem_cfg['pmem_specs']['dmagmin']
    dmagmax = pmem_cfg['pmem_specs']['dmagmax']
    data_for_calib = None 
    data_galc = general_local_galcat(
        data_gal, galcat_keys, hpx_meta, my_cluster, 
        calib_cfg['radius_mpc'], calib_cfg['Nsig'], dmagmin, dmagmax
    )
    ng = len(data_galc)
    if ng>0: 
        idcl = np.full(ng, my_cluster['idcl'])
        zcl =  np.full(ng, my_cluster['zcl'])
        mstar =  np.full(ng, my_cluster['mstar'])
        snr =  np.full(ng, my_cluster['snr_cl'])
        ra, dec = data_galc[galcat_keys['key_ra']],\
                  data_galc[galcat_keys['key_dec']]
        zp, mag = data_galc[galcat_keys['key_zp']],\
                  data_galc[galcat_keys['key_mag']]
        racl, deccl, zcl = my_cluster['racl'],\
                           my_cluster['deccl'],\
                           my_cluster['zcl']
        conv_factor = my_cluster['conv_factor']
        dcen = conv_factor * dist_ang(ra, dec, racl, deccl)
        dmag = mag - my_cluster['mstar']
        data_for_calib = np.zeros( (ng), 
                                   dtype = [
                                       (galcat_keys['key_zp'], 'f4'), 
                                       ('zcl', 'f4'), 
                                       (galcat_keys['key_mag'], 'f4'), 
                                       ('mstar', 'f4'),
                                       ('dist2cl_mpc', 'f4'), 
                                       ('snr', 'f4'), 
                                       ('id_cl', 'a30')   
                                   ]
        )
        data_for_calib[galcat_keys['key_zp']] = zp
        data_for_calib['zcl'] = zcl
        data_for_calib[galcat_keys['key_mag']] = mag
        data_for_calib['mstar'] = mstar
        data_for_calib['dist2cl_mpc'] = dcen
        data_for_calib['snr'] = snr
        data_for_calib['id_cl'] = idcl

    return data_for_calib


def local_zp_stats(data_gal_tile, galcat_keys, my_cluster):

    nsig = 5.

    data_gal_zpstats = general_local_galcat(
        data_gal_tile, galcat_keys, my_cluster, 0.5, nsig, 2., 2.
    )
    zp = data_gal_zpstats[galcat_keys['key_zp']]

    zcl, sig_dz = my_cluster['zcl'], my_cluster['sig_dz']
    zpmin, zpmax = zcl - nsig*sig_dz/(1.+zcl), zcl + nsig*sig_dz/(1.+zcl)
    zpmin2, zpmax2 =   zcl - 2.*sig_dz, zcl + 2.*sig_dz

    zpmed = np.median(zp[(zp<zpmax2) & (zp>zpmin2)])
    x = (zp - zpmed) / (1.+ zpmed)
    std0 = mad(x)

    '''
    plt.clf()
    npts = 50
    nzp, bins, patches = plt.hist(zp, npts,range=(zpmin5,zpmax5), density=False,facecolor='b', alpha=0.6) #, label = "zp  /  ng = "+str(nnzp))
    #nzs, bins, patches = plt.hist(zs_incl[cond], npts,range=(zpmin5,zpmax5), density=False,facecolor='r', alpha=0.6, label = "zs  /  ng = "+str(nnzs))
    plt.xlabel("z")
    plt.ylabel("N")
    plt.axvline(x=zcl, color='green', linestyle='-', label = "z$_{cl}$")
    plt.axvline(x=zpmed, color='red', linestyle='--', label = "median(zp)")
    plt.axvline(x=zpmin, color='green', linestyle='--')
    plt.axvline(x=zpmax, color='green', linestyle='--', label = " +/- 2$\sigma$")
    plt.title('cl. id = '+str(idcl).zfill(6)+'    zcl = '+str(round(zcl,2))+'    sig_dz = '+str(round(std0,2)), fontsize=8)
    plt.legend()
    plt.savefig(output, dpi=300)
    '''
    return zpmed, std0


def plot_pdz(my_cluster, zb, pdz, zpmed, output):

    idcl = my_cluster['idcl']
    zcl = my_cluster['zcl']
    sig_dz = my_cluster['sig_dz']

    zpmin5, zpmax5 = zcl - 5.*sig_dz, zcl + 5.*sig_dz
    zpmin, zpmax =   zcl - 2.*sig_dz, zcl + 2.*sig_dz
    pdz_mean = np.sum(pdz, axis=0)/float(len(pdz))
    cond = ((zb<zpmax5) & (zb>zpmin5))

    gauss_kernel = Gaussian1DKernel(3)
    pdz_smoo = convolve(pdz_mean[cond], gauss_kernel, boundary='extend') 

    plt.clf()
    plt.plot(zb[cond], pdz_mean[cond])
    plt.plot(zb[cond], pdz_smoo, color='red')
    plt.xlabel("z")
    plt.ylabel("N")
    plt.axvline(x=zcl, color='green', linestyle='-', label = "z$_{cl}$")
    plt.axvline(x=zpmed, color='red', linestyle='--', label = "median(zp)")
    plt.axvline(x=zpmin, color='green', linestyle='--')
    plt.axvline(x=zpmax, color='green', 
                linestyle='--', label = " +/- 2$\sigma$")
    plt.title(
        'cl. id = '+str(idcl).zfill(6)+'    zcl = '+str(round(zcl,2)), 
        fontsize=8
    )
    plt.legend()
    plt.savefig(output, dpi=300)    
    return 


def cl_dens_ann_arcmin (my_cluster, data_gal, galcat_keys, 
                        radius_cl_min_deg, radius_cl_max_deg, 
                        mag_bin_specs, pmem_specs, bkg_arcmin2, 
                        data_fp_mask, hpx_meta):
    """
    Computes the galaxy density /arcmin2 as a fct of mag in the cluster 
    in the annulus (radius_cl_min_deg, radius_cl_max_deg)
    Corrected density : at mags brighter than m* cl/bkg is 
    at least = cl/bkg(mag=m*)
    Returns the raw and corrected densities 
    """
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    mstar = my_cluster['mstar']
    imag_star = my_cluster['imag_star']
    dmag_densnorm = pmem_specs['dmag_densnorm']
    mag = data_gal[galcat_keys['key_mag']]
    conv_factor = my_cluster['conv_factor']
    dist2cl_mpc = data_gal['dist2cl_mpc']
    dist2cl_deg = np.degrees(dist2cl_mpc / conv_factor)

    in_ann = ((dist2cl_deg<= radius_cl_max_deg) &\
              (dist2cl_deg>=radius_cl_min_deg))
    hpix, frac, area, annulus_area_fraction = hpx_in_annulus (
        racl, deccl, radius_cl_min_deg, radius_cl_max_deg, 
        data_fp_mask, hpx_meta, inclusive=False
    )
    if len(mag[in_ann]) >  0:
        magb, _cl_ = binned_counts(mag[in_ann], mag_bin_specs, pmem_specs)
        if annulus_area_fraction > 0. and pmem_specs['area_correction']:
            _cl_ = _cl_ /annulus_area_fraction
        cl_arcmin2 = _cl_/\
                     (3600.*\
                      area_ann_deg2(
                          radius_cl_min_deg, radius_cl_max_deg
                      )) - bkg_arcmin2
        cond1 = ((cl_arcmin2>=bkg_arcmin2) &\
                 (bkg_arcmin2>0.) &\
                 (magb>mstar-dmag_densnorm) &\
                 (magb<mstar+dmag_densnorm) )
        """
        ratio = cl_arcmin2/bkg_arcmin2
        ratio[(bkg_arcmin2<=0)] = 0.
        ratio_mstar = np.median(ratio[cond1])
        if ratio_mstar >= 1.:
            ratio[0:imag_star] = ratio_mstar
        mean_cl = ratio * bkg_arcmin2
        cl_cor = np.maximum (cl_arcmin2, mean_cl) 
        cl_cor[cl_cor<0.] = 0.
        cl_arcmin2[cl_arcmin2<0.] = 0.
        """
        ratio_mstar = 1.
        if (len(bkg_arcmin2[cond1])>0):
            ratio_mstar = np.median(cl_arcmin2[cond1]/bkg_arcmin2[cond1])
        else:
           print ('    .....cl/bkg ratio not defined')
 
        mean_cl = np.copy(cl_arcmin2)

        if ratio_mstar > 1.:
            #mean_cl[0:imag_star+1] = ratio_mstar * bkg_arcmin2[0:imag_star+1]
            mean_cl = ratio_mstar * bkg_arcmin2

        cl_cor = np.maximum (cl_arcmin2, mean_cl) 
        cl_cor[cl_cor<0.] = 0.
        cl_arcmin2[cl_arcmin2<0.] = 0.
    else:
        cl_cor = np.zeros(vec_bin_pos(mag_bin_specs['max'], mag_bin_specs))
        cl_arcmin2 = np.zeros(vec_bin_pos(mag_bin_specs['max'], mag_bin_specs))

    return cl_cor, cl_arcmin2, 3600.* area_ann_deg2(
        radius_cl_min_deg, radius_cl_max_deg)
                      

def varGaussian_pdf(x, sig0, allx, dx, startpos, endpos):
    """
    estimates a var Gaussian pdf for the observable x
      - x = observed (mean) value
      - sig0: fractional uncertainy (sig0*(1+x))
      - dx = increment in allx array
      - allx = 	xmin+ np.array(range(Nxbin+1))*dx  #array of x centered at the LEFT edge (xmax included!!)
      - Nxbin = (xmax-xmin)/dx  
      - startpos, endpos: starting and ending positions in allx
    RESULT:
    p(x) ~ exp(-0.5*(x'-x)**2/((sig0*(1+x))**2) NOTE: p(x) is a fct of x, not of x': there is the 1+x term in the denominator
    """
    xpos = list(range(startpos,endpos+1)) #positions where x-values should be considered 
    _x = allx[xpos]+dx/2.0                #+-3\sigma CENTROID redshifts considered for the specific source
    ex = np.sqrt((sig0*(1.+_x))**2.+dx**2.) #0th order approximation (in zp-zs, i.e. allx-x)  for the \sigma, where convolution with Gaussian with sigma = dx is applied
    pdf = np.exp(-(_x-x)**2./(2.*(ex**2.)))/ex #var gaussian pdf
    if sum(pdf)!=0. : pdf = pdf/sum(pdf)# normalization equal to the unit. Integration truncated at 3\sigma
    result = {'pos':xpos,'pdf':pdf}
    result = np.core.records.fromarrays(
        list(result.values()), names=list(result.keys())
    )# dictionary => structured array
    return result


def general_local_galcat(data_gal_tile, galcat_keys, hpx_meta, 
                         my_cluster, radius_max_mpc, nsig, dmagmin, dmagmax):
    #extract relevant keys from galcat 

    ra, dec = data_gal_tile[galcat_keys['key_ra']],\
              data_gal_tile[galcat_keys['key_dec']]
    hpix_gal = data_gal_tile['hpix_gal']
    zp, mag = data_gal_tile[galcat_keys['key_zp']],\
              data_gal_tile[galcat_keys['key_mag']]
    #extract cluster data
    racl, deccl, zcl = my_cluster['racl'],\
                       my_cluster['deccl'],\
                       my_cluster['zcl']
    conv_factor = my_cluster['conv_factor']
    radius_bkg_max_deg = np.degrees(radius_max_mpc / conv_factor)
    mstar = my_cluster['mstar']
    zpmin, zpmax = zcl - nsig*my_cluster['sig_dz'],\
                   zcl + nsig*my_cluster['sig_dz'] 
    magmin, magmax = mstar - dmagmin, mstar + dmagmax

    # filter galaxies in mag and zp
    cond = ((zp <= zpmax) & (zp >= zpmin) & (mag < magmax) & (mag > magmin))

    # filter in distance to cl. 
    in_cone = cond_in_hpx_disc(
        hpix_gal[cond], 
        hpx_meta['Nside'], hpx_meta['nest'], 
        racl, deccl, radius_bkg_max_deg
    )
    cond[np.argwhere(cond).T[0]] = in_cone 
    data_gal = data_gal_tile[cond]
    return data_gal


def local_galcat (data_gal_tile, galcat_keys, hpx_meta, my_cluster, 
                  radius_min_mpc, radius_max_mpc, mag_bin_specs):
    """
    From the tile galcat in memory, selects objects in relevant 
    (zp, mag, distance from cluster) ranges. In particular it keeps only 
    galaxies in a cone centered on cluster and radius = maximum local bkg radius
    Output is a structured array
    """
    nsig = (my_cluster['zpmax'] - my_cluster['zcl'])/my_cluster['sig_dz']
    magmin = max(my_cluster['magmin'], mag_bin_specs['min'])
    magmax = min(my_cluster['magmax'], mag_bin_specs['max'])
    dmagmax = magmax - my_cluster['mstar']
    dmagmin = my_cluster['mstar'] - magmin

    local_cat = general_local_galcat(data_gal_tile, galcat_keys, 
                                     hpx_meta, my_cluster, 
                                     radius_max_mpc, nsig, dmagmin, dmagmax)
    t = Table(local_cat)

    # add distance to cluster for the galaxies in the cluster region
    racl, deccl, zcl = my_cluster['racl'],\
                       my_cluster['deccl'],\
                       my_cluster['zcl']
    conv_factor = my_cluster['conv_factor']
    radius_bkg_min_deg = np.degrees(radius_min_mpc / conv_factor)
    radius_bkg_max_deg = np.degrees(radius_max_mpc / conv_factor)
    ra, dec = t[galcat_keys['key_ra']], t[galcat_keys['key_dec']]
    hpix_gal = t['hpix_gal']
    in_cone = cond_in_disc(
        ra, dec, hpix_gal, 
        hpx_meta['Nside'], hpx_meta['nest'], 
        racl, deccl, radius_bkg_min_deg
    )
    dist2cl_mpc = np.ones(len(t))*radius_max_mpc
    dist2cl_mpc[in_cone] =  conv_factor * \
                            dist_ang(
                                ra[in_cone], dec[in_cone], racl, deccl
                            )
    t['dist2cl_mpc'] = dist2cl_mpc
    return t


def local_galcat_zp_test (data_gal_tile, galcat_keys, my_cluster, hpx_meta):
    """
    From the tile galcat in memory, selects objects in relevant 
    (zp, mag, distance from cluster) ranges. In particular it keeps only 
    galaxies in a cone centered on cluster and radius = maximum local bkg radius
    Output is a structured array
    """

    radius_test = 0.5 # Mpc
    #extract relevant keys from galcat 
    id_gal0 = data_gal_tile[galcat_keys['key_id']]
    ra0, dec0  = data_gal_tile[galcat_keys['key_ra']],\
                 data_gal_tile[galcat_keys['key_dec']]
    zp0 = data_gal_tile[galcat_keys['key_zp']]
    mag0 = data_gal_tile[galcat_keys['key_mag']]
    #hpix_gal0 = radec2hpix(ra0, dec0, hpx_meta['Nside'], hpx_meta['nest'])
    hpix_gal0 = hp.ang2pix(
        hpx_meta['Nside'],ra0, dec0, hpx_meta['nest'], lonlat=True)

    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    zcl = my_cluster['zcl']
    mstar = my_cluster['mstar']
    conv_factor = my_cluster['conv_factor']
    sig_dz = my_cluster['sig_dz']

    radius_bkg_max_deg = np.degrees(radius_test / conv_factor)
    zpmin, zpmax = zcl - 3.*sig_dz, zcl + 3.*sig_dz
    magmin, magmax = mstar - 2., mstar + 2.

    # select galaxies only in the cluster cone and with a mag_cut
    hpx_cone = all_hpx_in_annulus(
        racl, deccl, 0., radius_bkg_max_deg, hpx_meta, inclusive=True
    )
    in_cone = np.isin(hpix_gal0, hpx_cone)     # galaxies in bkg region 
    cond_z = ((zp0 <= zpmax) & (zp0 >= zpmin))
    cond_mag = ((mag0 < magmax) & (mag0 > magmin))
    cl_cond = (in_cone & cond_mag & cond_z)

    id_gal, ra, dec, zp, mag = id_gal0[cl_cond],\
                               ra0[cl_cond], dec0[cl_cond],\
                               zp0[cl_cond], mag0[cl_cond] 
    dist2cl_mpc = conv_factor * dist_ang(ra, dec, racl, deccl)
    data_gal = np.zeros( (len(ra)), 
                         dtype = [
                             (galcat_keys['key_id'], 'i8'), 
                             (galcat_keys['key_ra'], 'f8'), 
                             (galcat_keys['key_dec'], 'f8'), 
                             (galcat_keys['key_zp'], 'f8'), 
                             (galcat_keys['key_mag'], 'f8'), 
                             ('dist2cl_mpc', 'f8')
                         ]
    )
    data_gal[galcat_keys['key_id']] = id_gal
    data_gal[galcat_keys['key_ra']] = ra
    data_gal[galcat_keys['key_dec']]= dec
    data_gal[galcat_keys['key_zp']] = zp
    data_gal[galcat_keys['key_mag']]= mag
    data_gal['dist2cl_mpc'] = dist2cl_mpc

    return data_gal


def normalized_central_density(my_cluster, pmem_specs, mdens):
    # mdens = measured density in the specs radius 
    R0 = pmem_specs['radius_densnorm']
    Rc = pmem_specs['radmin_cc']
    slope = my_cluster['slope_dprofile']
 
    if math.isclose(slope, -2., rel_tol=1e-5):
        d0 = mdens / (1. + 2.*np.log (R0/Rc))
    else:
        d0 = mdens * (slope/2. + 1.) / (1. + 0.5 * slope * (Rc/R0)**(slope+2.))
    
    return d0


def  ntot_nbkg ( my_cluster, data_gal, galcat_keys,  
        radial_bin_specs, mag_bin_specs, pmem_specs, 
        data_fp_mask, hpx_meta, bkg_arcmin2, bkg_area_arcmin2):

    conv_factor = my_cluster['conv_factor']
    magmin, magmax = my_cluster['magmin'], my_cluster['magmax']
    rrr0 = pmem_specs['radius_densnorm']
    mag = data_gal[galcat_keys['key_mag']]
    dist2cl_mpc = data_gal['dist2cl_mpc']
    radmin_cc = pmem_specs['radmin_cc']
    slope_dprofile = my_cluster['slope_dprofile']

    magb = mag_binning(mag_bin_specs)
    lradb = np.linspace(np.log10(radial_bin_specs['radius_min_mpc']),\
                        np.log10(radial_bin_specs['radius_max_mpc']),\
                        radial_bin_specs['nstep']) 
    radb = 10**lradb

    rm0_deg = np.degrees(rrr0 / conv_factor)
    cl_arcmin2_cor, cl_arcmin2, area_cl_arcmin2 = cl_dens_ann_arcmin(
        my_cluster, data_gal, galcat_keys, 0., rm0_deg, 
        mag_bin_specs, pmem_specs, bkg_arcmin2, data_fp_mask, hpx_meta
    )
    d0 = np.zeros(len(magb))
    bkg_dens = np.zeros(len(magb))
    for im in range(0, len(magb)):
        magmin, magmax = max(mag_bin_specs['min'], magb[im] - 0.5),\
                         min(mag_bin_specs['max'], magb[im] + 0.5)
        cl_dens = np.sum(
            cl_arcmin2_cor[vec_bin_pos(magmin, mag_bin_specs):\
                           vec_bin_pos(magmax, mag_bin_specs)]
        )
        bkg_dens[im] = np.sum(
            bkg_arcmin2[vec_bin_pos(magmin, mag_bin_specs):\
                        vec_bin_pos(magmax, mag_bin_specs)]
        )
        d0[im] = normalized_central_density(my_cluster, pmem_specs, cl_dens)

    imag = vec_bin_pos(mag, mag_bin_specs)
    dist2cl_mpc[dist2cl_mpc<radmin_cc] = radmin_cc
    area_ref = np.pi * 0.3**2
    area_arcmin2 = area_ref*3600. / np.radians(conv_factor)**2
    ncl = area_arcmin2 * (d0[imag]*(dist2cl_mpc / rrr0)**(slope_dprofile)) 
    nbkg_cl = area_arcmin2 * bkg_dens[imag]
    ntot = np.maximum (ncl + nbkg_cl, 1.)
    nbkg = np.maximum (bkg_area_arcmin2 * bkg_dens[imag], 1.) 

    return ntot, nbkg


def  cl_density_ratio_drad_dmag ( my_cluster, data_gal, galcat_keys,  
        radial_bin_specs, mag_bin_specs, pmem_specs, 
        data_fp_mask, hpx_meta, bkg_arcmin2):

    conv_factor = my_cluster['conv_factor']
    slope_dprofile = my_cluster['slope_dprofile']
    mag = data_gal[galcat_keys['key_mag']]
    dist2cl_mpc = data_gal['dist2cl_mpc']
    radmin_cc = pmem_specs['radmin_cc']
    rrr0 = pmem_specs['radius_densnorm']

    magb = mag_binning(mag_bin_specs)
    rm0_deg = np.degrees(rrr0 / conv_factor)
    cl_arcmin2_cor, cl_arcmin2, area_cl_arcmin2 =  cl_dens_ann_arcmin(
        my_cluster, data_gal, galcat_keys, 0., rm0_deg, 
        mag_bin_specs, pmem_specs, bkg_arcmin2, 
        data_fp_mask, hpx_meta
    )
    d0 = np.zeros(len(magb))
    for im in range(0, len(magb)):
        d0[im] = normalized_central_density(
            my_cluster, pmem_specs, cl_arcmin2_cor[im]
        )
    imag = vec_bin_pos(mag, mag_bin_specs)

    bkg_arcmin2[bkg_arcmin2<=0.] = 0.00001 # patch to avoid pmem = nan when BCGs are not represented in bkg
    dist2cl_mpc[dist2cl_mpc<radmin_cc] = radmin_cc
    cl_local_overdensity = (d0[imag]*(dist2cl_mpc / rrr0)**(slope_dprofile))/\
                           bkg_arcmin2[imag]
    return cl_local_overdensity, cl_arcmin2_cor, cl_arcmin2


def compute_pmem(my_cluster, pmem_cfg, 
                 data_gal, galcat_keys, data_fp_mask, hpx_meta, 
                 bkg_arcmin2, bkg_area_arcmin2, zpb, bkg_zdensity):
    """
    This routine computes Pmem's and their errors for a given 
    cluster ('my cluster')
    Computes first the overdensity relative to bkg of galaxies in bins 
    of radial distance and mag, which is done based on the effective 
    available area of the galaxies thanks to the healpix footprint maps.  
    Returns for each galaxy : Pmem, Pmem_err. It also returns Pmem_rad - 
    which does not not consider density variations along the photo z 
    suupport - for diagnostics
    """

    radial_bin_specs = pmem_cfg['radial_bin_specs']
    mag_bin_specs = pmem_cfg['mag_bin_specs']
    pmem_specs = pmem_cfg['pmem_specs']
    photoz_support = pmem_cfg['photoz_support']

    zpcl = my_cluster['zcl']
    zpmin, zpmax = my_cluster['zpmin'], my_cluster['zpmax']
    sig_dz = my_cluster['sig_dz']

    #extract relevant keys from galcat 
    mag = data_gal[galcat_keys['key_mag']]
    zp = data_gal[galcat_keys['key_zp']]
    dist2cl_mpc = data_gal['dist2cl_mpc']

    cl_local_overdensity, cl_local_cor, cl_local =  cl_density_ratio_drad_dmag(
        my_cluster, data_gal, galcat_keys, radial_bin_specs, 
        mag_bin_specs, pmem_specs, data_fp_mask, hpx_meta, bkg_arcmin2
    )

    #cl_local_overdensity_zp = cl_local_overdensity * gaussian(zp, zpcl, sig_dz) * (zpmax - zpmin)
    bkg_density_zp = interpolate.interp1d(
        zpb, bkg_zdensity, 
        kind = 'linear', 
        bounds_error=False, 
        fill_value='extrapolate'
    )
    cl_local_overdensity_zp = cl_local_overdensity *\
                              gaussian(zp, zpcl, sig_dz)\
                              / bkg_density_zp(zp)
    
    sigcl_dz = photoz_support['sigcl_over_sigg']*\
               (zpmax-zpmin)/(2.*photoz_support['nsig'])

    PgPc = np.exp(
        -(zp-zpcl)**2 / (2.*(sig_dz**2 + sigcl_dz**2))
    ) # normalized convolution in Castignani & Benoist 2016

    pmem = (cl_local_overdensity_zp/(1.+cl_local_overdensity_zp)) * PgPc
    pmem[(cl_local_overdensity_zp<=0.)] = 0. 
    #for i in range(0, len(pmem)):
    #    print (i, pmem[i], PgPc[i],cl_local_overdensity_zp[i], cl_local_overdensity[i])  
    # pmem_err computation 
    ntot, nbkg = ntot_nbkg( 
        my_cluster, data_gal, galcat_keys, 
        radial_bin_specs, mag_bin_specs, pmem_specs, 
        data_fp_mask, hpx_meta, bkg_arcmin2, bkg_area_arcmin2
    )

    pmem_err = (1.-pmem) * ( 1./np.sqrt(nbkg) + 1./ np.sqrt(ntot))
    return pmem, pmem_err


def generate_output_radii(my_cluster, radial_bin_specs, richness_specs):
    """
    sets the logorithmic binning to deliver a vectorial richness
    """
    conv_factor = my_cluster['conv_factor']
    npts = richness_specs['npts']

    lradb = np.linspace(
        np.log10(radial_bin_specs['radius_min_mpc']), 
        np.log10(radial_bin_specs['radius_max_mpc']), npts
    ) 
    radb = 10**lradb
    radb_arcmin = 60. * np.degrees(radb / conv_factor)
    return np.around(radb, 3), np.around(radb_arcmin, 3)


def compute_richness_vec(pmem_cfg, pmem0, pmem_err0, data_gal0, 
                         galcat_keys, r200, my_cluster, data_fp, hpx_meta):
    """
    Computes the richness (and errors) as a sum of Pmems in a set of radii 
    down to a given magnitude.
    Pmems are summed in increasing annuli in order to correct for missing 
    data at each step. 
    Errors on richnesses are derived from the errors on individual Pmems. 
    """
    radial_bin_specs = pmem_cfg['radial_bin_specs']
    richness_specs = pmem_cfg['richness_specs']
    pmem_specs = pmem_cfg['pmem_specs']

    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']

    radb, radius_vec_arcmin = generate_output_radii(my_cluster, 
                                     radial_bin_specs, richness_specs)
 
    cond_richness = ((data_gal0['dist2cl_mpc']/r200 < pmem_specs['dmax']) & 
                     (data_gal0[galcat_keys['key_mag']] < my_cluster['mstar'] +\
                      richness_specs['dmag_faint']) & 
                     (data_gal0[galcat_keys['key_mag']] > my_cluster['mstar'] -\
                      richness_specs['dmag_bright']) )

    data_gal = data_gal0[cond_richness]
    pmem, pmem_err = pmem0[cond_richness], pmem_err0[cond_richness]

    dist2cl_mpc = data_gal['dist2cl_mpc']
    radb_deg = np.degrees(radb / conv_factor)

    rich, rich_err2 = np.zeros(len(radb)), np.zeros(len(radb))    
    for i in range(0, len(radb)):        
        if i == 0: # no area correction
            rich[i] = np.sum(pmem[dist2cl_mpc<radb[i]])
            rich_err2[i] = np.sum(pmem_err[dist2cl_mpc<radb[i]]**2) 
        else:
            drich = np.sum(
                pmem[(dist2cl_mpc<=radb[i]) & (dist2cl_mpc>radb[i-1])]
            )
            drich_err2 = np.sum(
                pmem_err[(dist2cl_mpc<=radb[i]) & (dist2cl_mpc>radb[i-1])]**2
            )

            hpix, frac, area, annulus_area_fraction = hpx_in_annulus(
                racl, deccl, radb_deg[i-1], radb_deg[i], 
                data_fp, hpx_meta, inclusive=False
            )

            if annulus_area_fraction > 0. and pmem_specs['area_correction']: 
                rich[i] = rich[i-1] + drich / annulus_area_fraction
                rich_err2[i] = rich_err2[i-1] +\
                               drich_err2 / annulus_area_fraction
            else: 
                rich[i] = rich[i-1] + drich
                rich_err2[i] = rich_err2[i-1] + drich_err2

    rich_err = np.sqrt(rich_err2)
    xr = radb[(radb<=r200*1.2) & (rich>0.)]
    yr = rich[(radb<=r200*1.2) & (rich>0.)]

    if (len(xr)>2):
        rfit = np.poly1d(np.polyfit(np.log10(xr[yr>0]), 
                                np.log10(yr[yr>0]), 1))
    else:
        rfit = [0., 1.]
    return rich, rich_err, radb, radius_vec_arcmin, rfit


def ngals_cor_from_pmems(pmem_cfg, pmem0, pmem_err0, data_gal0, 
                         galcat_keys, r200, my_cluster, data_fp, hpx_meta):
    """
    Computes the richness (and errors) as a sum of Pmems in a set of radii 
    down to a given magnitude.
    Pmems are summed in increasing annuli in order to correct for missing 
    data at each step. 
    Errors on richnesses are derived from the errors on individual Pmems. 
    """
    radial_bin_specs = pmem_cfg['radial_bin_specs']
    richness_specs = pmem_cfg['richness_specs'] 
    pmem_specs = pmem_cfg['pmem_specs'] 

    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    radb, radius_vec_arcmin = generate_output_radii(
        my_cluster, radial_bin_specs, richness_specs
    )
    cond_richness = ((data_gal0['dist2cl_mpc']/r200 <= 1.) & 
                     (data_gal0[galcat_keys['key_mag']] < my_cluster['mstar'] +\
                      richness_specs['dmag_faint']) & 
                     (data_gal0[galcat_keys['key_mag']] > my_cluster['mstar'] -\
                      richness_specs['dmag_bright']) )

    data_gal = data_gal0[cond_richness]
    pmem, pmem_err = pmem0[cond_richness], pmem_err0[cond_richness]
    dist2cl_mpc = data_gal['dist2cl_mpc']
    radb_deg = np.degrees(radb / conv_factor)

    rich, rich_err2 = np.zeros(len(radb)), np.zeros(len(radb))    
    for i in range(0, len(radb)):
        if i == 0:
            rich[i] = np.sum(pmem[dist2cl_mpc<radb[i]]) 
            rich_err2[i] = np.sum(pmem_err[dist2cl_mpc<radb[i]]**2) 
        else:
            if r200 > radb_deg[i-1]:
                hpix, frac, area, annulus_area_fraction = hpx_in_annulus(
                    racl, deccl, radb_deg[i-1], min(radb_deg[i], r200), 
                    data_fp, hpx_meta, inclusive=False
                )

                if annulus_area_fraction > 0. and pmem_specs['area_correction']:
                    rich[i] = rich[i-1] +\
                              np.sum(
                                  pmem[(dist2cl_mpc<=radb[i]) &\
                                       (dist2cl_mpc>radb[i-1])]
                              ) /annulus_area_fraction
                    rich_err2[i] = rich_err2[i-1] +\
                                   np.sum(
                                       pmem_err[(dist2cl_mpc<=radb[i]) &\
                                                (dist2cl_mpc>radb[i-1])]**2
                                   ) / annulus_area_fraction
                else:
                    rich[i] = rich[i-1] +\
                              np.sum(pmem[ (dist2cl_mpc<=radb[i]) &\
                                           (dist2cl_mpc>radb[i-1]) ]
                              )
                    rich_err2[i] = rich_err2[i-1] +\
                                   np.sum(
                                       pmem_err[(dist2cl_mpc<=radb[i]) &\
                                                (dist2cl_mpc>radb[i-1])]**2
                                   )

    rich_err = np.sqrt(rich_err2)
    return rich[i], rich_err[i]


def n200_raw_from_pmems(pmem, data_gal, galcat_keys, my_cluster, 
                        richness_specs, r200): 
    """
    n200 from Pmems without area correction 
    """
    cond_richness = ((data_gal['dist2cl_mpc']/r200 <= 1.) & 
                     (data_gal[galcat_keys['key_mag']] < my_cluster['mstar'] +\
                      richness_specs['dmag_faint']) & 
                     (data_gal[galcat_keys['key_mag']] > my_cluster['mstar'] -\
                      richness_specs['dmag_bright']) )

    n200_pmem_raw = round(np.sum(pmem[cond_richness]), 3)
    return n200_pmem_raw


def degraded_footprint(data_fp, hpx_meta, Nside_plot):

    '''
    # very slow because operates on the whole sphere
    hpix_map, frac_map = data_fp[hpx_meta['key_pixel']], data_fp[hpx_meta['key_frac']]
    if Nside_plot < hpx_meta['Nside']:
        npix = hp.nside2npix(hpx_meta['Nside'])
        hmap = np.arange(npix)
        pixel0 = np.zeros(len(hmap))
        pixel0[hpix_map]=1
        frac0 = hp.pixelfunc.ud_grade(pixel0, Nside_plot)
        hpix_map_plot = np.argwhere(frac0>0).T[0]
        frac_map_plot = frac0[(frac0>0)]
    else:
        hpix_map_plot, frac_map_plot = np.copy(hpix_map), np.copy(frac_map)
    '''
    # here it assumes frac_map = 1 => to be updated
    hpix_map, frac_map = data_fp[hpx_meta['key_pixel']],\
                         data_fp[hpx_meta['key_frac']]
    ra, dec = hp.pix2ang(
        hpx_meta['Nside'], hpix_map, hpx_meta['nest'], lonlat=True
    )
    pix_out0 = hp.ang2pix(Nside_plot, ra, dec, hpx_meta['nest'], lonlat=True)
    hpix_map_plot, counts = np.unique(pix_out0, return_counts=True)
    nsamp = (float(hpx_meta['Nside'])/float(Nside_plot))**2
    frac_map_plot = counts.astype(float)/nsamp

    data_fp_plot = np.zeros( 
        (len(hpix_map_plot)), 
        dtype={
            'names':(
                hpx_meta['key_pixel'], 
                hpx_meta['key_frac']
            ),
            'formats':(
                'i8', 'f8'
            )
        }
    )
    data_fp_plot[hpx_meta['key_pixel']] = hpix_map_plot
    data_fp_plot[hpx_meta['key_frac']] = frac_map_plot
    return data_fp_plot


'''
def compute_Nside_plot(target, tile_specs, hpx_meta):

    if (target['z']>0.) & ((tile_specs['decmax']-tile_specs['decmin']) > 0.75):  # downgrade resolution on plot to avoid too heavy plots
        k = int(np.log10((tile_specs['decmax'] - tile_specs['decmin'])/0.5)/np.log10(2.)) + 1
        Nside_plot = int(hpx_meta['Nside']/2**k)
    else: 
        Nside_plot = hpx_meta['Nside']
'''

def compute_Nside_plot(hpx_meta, radius_deg, nstep):
    Nside_arr = np.array([hpx_meta['Nside']])
    Nside = hpx_meta['Nside']
    while (Nside >=2):
        Nside = Nside/2
        Nside_arr = np.hstack((Nside_arr, np.array([Nside])))
    size_pix_deg = hp.nside2resol(Nside_arr.astype(int), arcmin=True)/60.
    Nside_plot = int(
        Nside_arr[np.abs(radius_deg/float(nstep) - size_pix_deg).argmin()]
    )
    return Nside_plot


def data_gal_tile_from_mosaic (galcat_mosaic_specs, galcat_keys, 
                               my_cluster, radius_max_mpc):
    """
    From a list of galcat files, selects objects in relevant 
    (zp, mag, distance from cluster) ranges. In particular it keeps only 
    galaxies in a cone centered on cluster and radius = maximum local bkg radius
    Output is a structured array
    """
    racl, deccl = my_cluster['racl'], my_cluster['deccl']
    conv_factor = my_cluster['conv_factor']
    radius_max_deg = np.degrees(radius_max_mpc / conv_factor)
    return read_mosaicFitsCat_in_disc(galcat_mosaic_specs, galcat_keys, 
                                      racl, deccl, radius_max_deg)


def pmem_tile(pmem_cfg, data_cls_analysis, data_cls_all, clcat_keys,
              data_fp, hpx_meta, data_gal, galcat, 
              sig_dz0, cosmo_params, mstar_filename, out_paths, verbose):

    workdir = out_paths['workdir_loc'] 
    path = out_paths['pmem']
    photoz_support = pmem_cfg['photoz_support']    
    galcat_keys = galcat['keys']

    data_gal = add_hpx_to_cat(
        data_gal, 
        data_gal[galcat['keys']['key_ra']], 
        data_gal[galcat['keys']['key_dec']],
        hpx_meta['Nside'], hpx_meta['nest'], 'hpix_gal'
    )

    # Initialize output table "richness"
    data_richness = init_richness_output(
        data_cls_analysis, pmem_cfg['richness_specs'], clcat_keys
    )
    data_members, data_members_tile = None, None 
    flag_init_members = 0
    data_for_calib = None
    flag_init_calib = 0

    for i in range(0, len(data_cls_analysis)):

        data_cluster = data_cls_analysis[i]
        idcl =  data_cluster[clcat_keys['key_id']]
        racl =  data_cluster[clcat_keys['key_ra']]
        deccl = data_cluster[clcat_keys['key_dec']]
        zcl =   data_cluster[clcat_keys['key_zp']]

        if verbose>=1:
            print ('')
            print ('( '+str(i+1)+'/'+str(len(data_cls_analysis))+\
                   ' )   Cluster ID = '+str(idcl)+\
                   '      ra = '+str(round(racl,3))+\
                   '      dec = '+str(round(deccl,3))+\
                   '      zcl = '+str(round(zcl,3)))

        if (zcl < pmem_cfg['global_conditions']['zcl_min'] or 
            zcl > pmem_cfg['global_conditions']['zcl_max']):
            data_richness['flag_pmem'][i] = 1
            continue

        # build "my_cluster" dictionary that describes the useful cl ppties. 
        my_cluster = build_my_cluster (
            i, data_cluster, clcat_keys, pmem_cfg['photoz_support'], sig_dz0, 
            pmem_cfg['mag_bin_specs'], pmem_cfg['pmem_specs'], 
            pmem_cfg['richness_specs'], mstar_filename, cosmo_params
        )

        # local working galcat & footprints 
        data_lgal = local_galcat(
            data_gal, galcat['keys'], hpx_meta, my_cluster, 
            pmem_cfg['bkg_specs']['radius_min_mpc'], 
            pmem_cfg['bkg_specs']['radius_max_mpc'], 
            pmem_cfg['mag_bin_specs']
        )
        if verbose>=1:
            print('    Nr. of galaxies in cluster field = '+str(len(data_lgal)))
        if len(data_lgal) == 0:
            data_richness['flag_pmem'][i] = 2
            continue
        data_lfp = local_footprint(
            my_cluster, data_fp, hpx_meta, pmem_cfg['bkg_specs']
        )
        data_lfp_mask, ncl_masked = footprint_with_cl_masks(
            my_cluster, data_cls_all, clcat_keys, 
            pmem_cfg['periphery_specs'], data_lfp, hpx_meta
        ) 
        if verbose>=1:
            print ('    Nr of masked clusters in periphery : ', ncl_masked)
 
        # test cluster and bkg coverage
        cl_cfc, cl_wcfc = compute_cl_coverfracs(
            my_cluster, pmem_cfg['weighted_coverfrac_specs'], 
            data_lfp, hpx_meta, cosmo_params
        )

        bkg_cfc, bkg_wmask_cfc, bkg_area_deg2 = compute_bkg_coverfracs(
            pmem_cfg['bkg_specs'], my_cluster, 
            data_lfp, data_lfp_mask, hpx_meta
        )

        if verbose>=1: 
            print (
                '    Cluster coverage (%)    raw = '+\
                str(round(100.*cl_cfc, 1))+\
                "     weighted = "+str(round(100.*cl_wcfc, 1))
            )
            print (
                '    Bkg     coverage (%)    raw = '+\
                str(round(100.*bkg_cfc, 1))+\
                " with cl.masks = "+str(round(100.*bkg_wmask_cfc, 1))
            )

        # generate footprint plot 
        if verbose >= 2:
            plot_footprint(
                my_cluster, data_lfp, hpx_meta, 
                pmem_cfg['bkg_specs']['radius_min_mpc'], 
                pmem_cfg['bkg_specs']['radius_max_mpc'], 
                pmem_cfg['weighted_coverfrac_specs']['radius_mpc'], 
                bkg_cfc, cl_cfc, cl_wcfc, 
                os.path.join(
                    workdir, 
                    path['plots'], 
                    'footprint_cl'+str(idcl)+'.png'
                )
            )

            if ncl_masked > 0:
                plot_footprint(
                    my_cluster, data_lfp_mask, hpx_meta, 
                    pmem_cfg['bkg_specs']['radius_min_mpc'], 
                    pmem_cfg['bkg_specs']['radius_max_mpc'], 
                    pmem_cfg['weighted_coverfrac_specs']['radius_mpc'],
                    bkg_wmask_cfc, cl_cfc, cl_wcfc,
                    os.path.join(
                        workdir, 
                        path['plots'], 
                        'footprint_with_clmask_cl'+str(idcl)+'.png'
                    )
                )

        # start feeding richness table 
        data_richness['raw_coverfrac'][i] = round(100.*cl_cfc, 1)
        data_richness['weighted_coverfrac'][i] = round(100.*cl_wcfc, 1)
        data_richness['bkg_raw_coverfrac'][i] = round(100.*bkg_cfc, 1)
        data_richness['bkg_coverfrac'][i] = round(100.*bkg_wmask_cfc, 1)

        if 100.*cl_wcfc < pmem_cfg['global_conditions']['cl_cover_min']:
            data_richness['flag_pmem'][i] = 3
            continue

        if 100.*bkg_wmask_cfc < pmem_cfg['global_conditions']['bkg_cover_min']:
            data_richness['flag_pmem'][i] = 4
            continue


        data_richness, data_members = pmem_1cluster(
            i, pmem_cfg, my_cluster, data_lfp, data_lfp_mask, hpx_meta, 
            data_lgal, galcat, bkg_area_deg2, cosmo_params, out_paths, 
            data_richness, verbose
        )

        # concatenate data_members 
        if data_members is not None:
            if flag_init_members == 0:
                data_members_tile  = np.copy(data_members)
                flag_init_members = 1
            else:
                data_members_tile = np.hstack(
                    (data_members_tile,  data_members)
                )

        # in calib_dz mode produce list of galaxies in 1Mpc cylinders around clusters with SNR>SNRlim
        if (pmem_cfg['calib_dz']['mode'] and 
            my_cluster['snr_cl']>pmem_cfg['calib_dz']['snr_min']): 
            data_for_calib = prepare_data_calib_dz(
                my_cluster, pmem_cfg, hpx_meta, data_gal, galcat['keys']
            )
            if data_for_calib is not None:
                if flag_init_calib == 0:
                    data_for_calib_tile  = np.copy(data_for_calib)
                    flag_init_calib = 1
                else:
                    data_for_calib_tile  = np.hstack(
                        (data_for_calib_tile,  data_for_calib)
                    )
            # write calib file 
            t = Table (data_for_calib_tile)
            t.write(
                os.path.join(
                    workdir, 
                    pmem_cfg['calib_dz']['filename']
                ),
                overwrite=True
            )

    return data_richness, data_members_tile


def pmem_list(pmem_cfg, data_cls_analysis, data_cls_all, clcat_keys,
              hpx_meta, galcat, 
              sig_dz0, cosmo_params, mstar_filename, out_paths, verbose):

    workdir = out_paths['workdir_loc'] 
    path = out_paths['pmem']
    photoz_support = pmem_cfg['photoz_support']    
    galcat_keys = galcat['keys']

    # Initialize output table "richness"
    data_richness = init_richness_output(
        data_cls_analysis, pmem_cfg['richness_specs'], clcat_keys
    )
    data_members, data_members_thread = None, None 
    flag_init_members = 0
    
    for i in range(0, len(data_cls_analysis)):

        data_cluster = data_cls_analysis[i]
        idcl =  data_cluster[clcat_keys['key_id']]
        racl =  data_cluster[clcat_keys['key_ra']]
        deccl = data_cluster[clcat_keys['key_dec']]
        zcl =   data_cluster[clcat_keys['key_zp']]

        if verbose>=1:
            print ('')
            print ('( '+str(i+1)+'/'+str(len(data_cls_analysis))+\
                   ' )   Cluster ID = '+str(idcl)+\
                   '      ra = '+str(round(racl,3))+\
                   '      dec = '+str(round(deccl,3))+\
                   '      zcl = '+str(round(zcl,3)))

        if (zcl < pmem_cfg['global_conditions']['zcl_min'] or 
            zcl > pmem_cfg['global_conditions']['zcl_max']):
            data_richness['flag_pmem'][i] = 1
            continue

        # build "my_cluster" dictionary that describes the useful cl ppties. 
        my_cluster = build_my_cluster(
            i, data_cluster, clcat_keys, pmem_cfg['photoz_support'], sig_dz0, 
            pmem_cfg['mag_bin_specs'], pmem_cfg['pmem_specs'], 
            pmem_cfg['richness_specs'], mstar_filename, cosmo_params
        )

        # build the generic local cat + footprint - specific to pmem_list
        radius_field_deg = np.degrees(
            pmem_cfg['bkg_specs']['radius_max_mpc']/ my_cluster['conv_factor']
        )
        field = {'ra': racl,'dec': deccl}

        field = np.core.records.fromarrays(
            list(field.values()), 
            names=list(field.keys())
        )# dict. in a np struct.array

        data_gal = read_mosaicFitsCat_in_disc(
            galcat, field, radius_field_deg
        )           

        if data_gal is None:
            data_richness['flag_pmem'][i] = 2
            continue

        data_gal = add_hpx_to_cat(
            data_gal, 
            data_gal[galcat['keys']['key_ra']], 
            data_gal[galcat['keys']['key_dec']],
            hpx_meta['Nside'], hpx_meta['nest'], 'hpix_gal'
        )
        data_fp = read_mosaicFootprint_in_disc(
            hpx_meta, field, radius_field_deg
        )

        # local working galcat & footprints 
        data_lgal = local_galcat(
            data_gal, galcat['keys'], hpx_meta, my_cluster, 
            pmem_cfg['bkg_specs']['radius_min_mpc'], 
            pmem_cfg['bkg_specs']['radius_max_mpc'], 
            pmem_cfg['mag_bin_specs']
        )
        if verbose>=1:
            print('    Nr. of galaxies in cluster field = '+\
                  str(len(data_lgal)))
        if len(data_lgal) == 0:
            data_richness['flag_pmem'][i] = 3
            continue
        data_lfp = local_footprint(
            my_cluster, data_fp, hpx_meta, pmem_cfg['bkg_specs'])
        data_lfp_mask, ncl_masked = footprint_with_cl_masks(
            my_cluster, data_cls_all, clcat_keys, 
            pmem_cfg['periphery_specs'], data_lfp, hpx_meta
        ) 
        if verbose>=1:
            print ('    Nr of masked clusters in periphery : ', ncl_masked)
 
        # test cluster and bkg coverage
        cl_cfc, cl_wcfc = compute_cl_coverfracs(
            my_cluster, pmem_cfg['weighted_coverfrac_specs'], 
            data_lfp, hpx_meta, cosmo_params
        )

        bkg_cfc, bkg_wmask_cfc, bkg_area_deg2 = compute_bkg_coverfracs(
            pmem_cfg['bkg_specs'], my_cluster, 
            data_lfp, data_lfp_mask, hpx_meta
        )

        if verbose>=1: 
            print ('    Cluster coverage (%)    raw = '+\
                   str(round(100.*cl_cfc, 1))+ 
                   "     weighted = "+str(round(100.*cl_wcfc, 1)))
            print ('    Bkg     coverage (%)    raw = '+\
                   str(round(100.*bkg_cfc, 1))+
                   " with cl.masks = "+str(round(100.*bkg_wmask_cfc, 1)))

        # generate footprint plot 
        if verbose >= 2:
            plot_footprint(
                my_cluster, data_lfp, hpx_meta, 
                pmem_cfg['bkg_specs']['radius_min_mpc'], 
                pmem_cfg['bkg_specs']['radius_max_mpc'], 
                pmem_cfg['weighted_coverfrac_specs']['radius_mpc'], 
                bkg_cfc, cl_cfc, cl_wcfc, 
                os.path.join(
                    workdir, 
                    path['plots'], 
                    'footprint_cl'+str(idcl)+'.png'
                )
            )

            if ncl_masked > 0:
                plot_footprint(
                    my_cluster, data_lfp_mask, hpx_meta, 
                    pmem_cfg['bkg_specs']['radius_min_mpc'], 
                    pmem_cfg['bkg_specs']['radius_max_mpc'], 
                    pmem_cfg['weighted_coverfrac_specs']['radius_mpc'],
                    bkg_wmask_cfc, cl_cfc, cl_wcfc,
                    os.path.join(
                        workdir, 
                        path['plots'], 
                        'footprint_with_clmask_cl'+str(idcl)+'.png'
                    )
                )

        # start feeding richness table 
        data_richness['raw_coverfrac'][i] = round(100.*cl_cfc, 1)
        data_richness['weighted_coverfrac'][i] = round(100.*cl_wcfc, 1)
        data_richness['bkg_raw_coverfrac'][i] = round(100.*bkg_cfc, 1)
        data_richness['bkg_coverfrac'][i] = round(100.*bkg_wmask_cfc, 1)

        if bkg_wmask_cfc < pmem_cfg['global_conditions']['bkg_cover_min'] or \
           cl_wcfc < pmem_cfg['global_conditions']['cl_cover_min']:
            data_richness['flag_pmem'][i] = 4
            continue

        data_richness, data_members = pmem_1cluster(
            i, pmem_cfg, my_cluster, data_lfp, data_lfp_mask, hpx_meta, 
            data_lgal, galcat, bkg_area_deg2, cosmo_params, out_paths, 
            data_richness, verbose
        )

        # concatenate data_members 
        if data_members is not None:
            if flag_init_members == 0:
                data_members_thread  = np.copy(data_members)
                flag_init_members = 1
            else:
                data_members_thread  = np.hstack(
                    (data_members_thread,  data_members)
                )
    return data_richness, data_members_thread


def pmem_1cluster(cl_index, pmem_cfg, my_cluster, data_fp, data_fp_mask, 
                  footprint, data_gal, galcat, bkg_area_deg2,
                  cosmo_params, out_paths, data_richness, verbose):

    workdir, path = out_paths['workdir_loc'], out_paths['pmem']
    data_members = None
    i = cl_index
    
    # compute mean galaxy number counts as a fct of magnitude and zp
    magb, counts_bkg_mpc2, counts_bkg_arcmin2, zpb, bkg_zdensity = bkg_mag_zp_counts(
        my_cluster, data_gal, galcat['keys'], 
        pmem_cfg['bkg_specs'], pmem_cfg['mag_bin_specs'], 
        pmem_cfg['pmem_specs'], pmem_cfg['richness_specs'],
        data_fp_mask, footprint
    ) 

    # mean bkg for storage only 
    md_bkg_mpc2, md_bkg_arcmin2 = mean_bkg(
        my_cluster, 
        counts_bkg_mpc2, counts_bkg_arcmin2, 
        pmem_cfg['mag_bin_specs'], pmem_cfg['richness_specs']
    ) 
    data_richness['md_bkg_mpc2'][i] = round(md_bkg_mpc2, 2)
    data_richness['md_bkg_arcmin2'][i] = round(md_bkg_arcmin2, 2)

    # compute cluster's R200/N200
    r200, n200, no_failure, flag_r200 = compute_r200_n200(
        my_cluster, data_gal, galcat['keys'], 
        counts_bkg_mpc2, 
        data_fp_mask, footprint, pmem_cfg,cosmo_params, 
        verbose, out_paths
    )
    r200_arcmin = 60. * np.degrees(r200 / my_cluster['conv_factor'])

    if not no_failure:
        data_richness['flag_pmem'][i] = 5
        return data_richness, data_members
                
    if verbose >= 1: 
        print ('    r200 (Mpc) / n200  = '+str(round(r200,2))+\
               '  /  '+str(round(n200, 2))+' [area corrected]')
        print ('    slope (rad-overd)  = '+\
               str(round(my_cluster['slope_dprofile'], 2)))
    
    if verbose >= 3: 
        plot_mag_counts(
            my_cluster, magb, counts_bkg_arcmin2, pmem_cfg['pmem_specs'], 
            data_gal, galcat['keys'], 
            pmem_cfg['radial_bin_specs'], pmem_cfg['mag_bin_specs'],
            data_fp_mask, footprint,
            os.path.join(
                workdir, 
                path['plots'], 
                'mag_counts_cl'+my_cluster['idcl']+'.png'
            )
        )
    
    # pmem
    pmem, pmem_err = compute_pmem(
        my_cluster, pmem_cfg,
        data_gal, galcat['keys'], data_fp_mask, footprint, 
        counts_bkg_arcmin2, bkg_area_deg2*3600., zpb, bkg_zdensity
    )

    # n200 from Pmems without / with area correction 
    n200_pmem_raw = n200_raw_from_pmems(
        pmem, data_gal, galcat['keys'], 
        my_cluster, pmem_cfg['richness_specs'], r200
    )

    n200_pmem_cor, n200_pmem_cor_err = ngals_cor_from_pmems(
        pmem_cfg, pmem, pmem_err, data_gal, galcat['keys'], r200, 
        my_cluster, data_fp, footprint
    )
    n_pmem_500kpc, n_pmem_500kpc_err = ngals_cor_from_pmems(
        pmem_cfg, pmem, pmem_err, data_gal, galcat['keys'], 0.5, 
        my_cluster, data_fp, footprint
    )
    if pmem_cfg['richness_specs']['external_radius']:
        external_radius[i] = my_cluster['external_radius']
        n_pmem_extrad[i], n_pmem_extrad_err[i] = ngals_cor_from_pmems(
            pmem_cfg, pmem, pmem_err, data_gal, galcat['keys'], 
            my_cluster['external_radius'], 
            my_cluster, data_fp, footprint
        )
    print ('    n200 (cor / raw) = ', 
           np.round(n200_pmem_cor, 2), ' / ', np.round(n200_pmem_raw, 2))

    # compute vector richness
    rich_vec, richerr_vec, rad_vec_mpc, rad_vec_arcmin, rfit = compute_richness_vec(
        pmem_cfg,
        pmem, pmem_err, data_gal, galcat['keys'], 
        r200, my_cluster, data_fp, footprint
    )

    # Feed richness table 
    cond_pmem = (
        (data_gal['dist2cl_mpc']/r200 < pmem_cfg['pmem_specs']['dmax']) & 
        (pmem>0.)
    )

    data_richness['nmem'][i] = len(pmem[cond_pmem])
    #data_richness['ncl_los'][i] = ncl_los
    data_richness['mstar'][i] = my_cluster['mstar']
    data_richness['r200_mpc'][i] = round(r200, 3)
    data_richness['n200'][i] = round(n200, 3)
    data_richness['n200_pmem_raw'][i] = round(n200_pmem_raw, 3)
    data_richness['n200_pmem'][i] = round(n200_pmem_cor, 3) 
    data_richness['n200_pmem_err'][i] = round(n200_pmem_cor_err, 3)
    data_richness['r200_arcmin'][i] = round(r200_arcmin, 3)
    data_richness['slope_dprofile'][i] = round(my_cluster['slope_dprofile'], 3)
    data_richness['slope_rad_over_rich'][i] = round(rfit[1], 3)
    data_richness['radius_vec_mpc'][i] = rad_vec_mpc
    data_richness['radius_vec_arcmin'][i] = rad_vec_arcmin
    data_richness['richness'][i] = rich_vec
    data_richness['richness_err'][i] = richerr_vec
    data_richness['flag_r200'][i] = flag_r200
    data_richness['n500kpc_pmem'][i] = round(n_pmem_500kpc, 3)
    data_richness['n500kpc_pmem_err'][i] = round(n_pmem_500kpc_err, 3)

    if verbose >= 1:
        print ('    raw sum(pmem)  in r200 = '+str(n200_pmem_raw))
        print ('    cor sum(pmem)  in r200 = '+\
               str(round(n200_pmem_cor,2))+\
               ' +/- '+str(round(n200_pmem_cor_err, 2)))
        print ('    rad-rich slope         = '+str(round(rfit[1],3)))

    # Final diagnostic plots
    if verbose >= 2:
        plot_pmems(
            my_cluster, 
            data_gal[cond_pmem], galcat['keys'], 
            pmem[cond_pmem], r200, n200_pmem_cor, out_paths
        )
    if verbose >= 2:
        plot_richness_vec (
            rad_vec_mpc, 
            rich_vec, richerr_vec, rfit, r200, 
            my_cluster['idcl'], out_paths
        )

    # Prepare Pmem table : ngal entries / cluster
    cond_pmem = ((data_gal['dist2cl_mpc']/r200 < 1.5) & (pmem>0.))
    nsat = len(pmem[cond_pmem])

    if nsat == 0 :
        data_richness['flag_pmem'][i] = 6
        return data_richness, data_members

    data_members = init_members_output(nsat)

    data_members['id_gal']   = data_gal[galcat['keys']['key_id']][cond_pmem]
    data_members['id_cl']    = np.full(
        len(data_gal[cond_pmem]), my_cluster['idcl']
    )
    data_members['zcl']      = np.full(
        len(data_gal[cond_pmem]), my_cluster['zcl']
    )
    data_members['pmem']     = pmem[cond_pmem]
    data_members['pmem_err'] = pmem_err[cond_pmem]
    data_members['dist2cl_mpc'] =  data_gal['dist2cl_mpc'][cond_pmem]
    data_members['dist2cl_over_r200'] = data_gal['dist2cl_mpc'][cond_pmem]/r200
    data_members['ra']       = data_gal[galcat['keys']['key_ra']][cond_pmem]
    data_members['dec']      = data_gal[galcat['keys']['key_dec']][cond_pmem]
    data_members['zp']       = data_gal[galcat['keys']['key_zp']][cond_pmem]
    data_members['zp-zcl_over_sig_dz'] = (data_gal[galcat['keys']['key_zp']][cond_pmem] -\
                                          my_cluster['zcl'])/my_cluster['sig_dz']
    data_members['mag']      =  data_gal[galcat['keys']['key_mag']][cond_pmem]
    data_members['mag-mstar'] = data_gal[galcat['keys']['key_mag']][cond_pmem] -\
                                my_cluster['mstar']

    return data_richness, data_members


def tile_radius_pmem(admin, pmem_cfg, cosmo_params):
    """_summary_

    Args:
        tiling (_type_): _description_

    Returns:
        _type_: _description_
    """

    if not admin['target_mode']:

        cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
        frame_mpc = pmem_cfg['bkg_specs']['radius_max_mpc']
        zmin = pmem_cfg['global_conditions']['zcl_min']
        conv_factor = cosmo.angular_diameter_distance(zmin)# radian*conv=mpc    
        frame_deg = np.degrees(frame_mpc / conv_factor.value)

        Nside_tile = admin['tiling']['Nside']
        tile_radius0 = (2.*\
                        hp.nside2pixarea(Nside_tile, degrees=True))**0.5 / 2.
        tile_radius = tile_radius0 + frame_deg

    return tile_radius


def eff_tiles_for_pmem(data_cls, clcat, tiles, admin):
    flag = np.zeros(len(tiles))
    for it in range(0, len(tiles)):
        for j in range(0, tiles['nhpix'][it]):
            tile_specs = {
                'Nside':tiles['Nside'][it],
                'nest': tiles['nest'][it],
                'hpix': tiles['hpix'][it][j]
            }
            data_cls_tile = filter_hpx_tile(
                data_cls, clcat, tile_specs
            )
            if len(data_cls_tile)>0:
                flag[it] = 1
    return tiles[flag==1]


def run_pmem_tile(config, dconfig, thread_id):
    # read config file
    with open(config) as fstream:
        param_cfg = yaml.safe_load(fstream)
    with open(dconfig) as fstream:
        param_data = yaml.safe_load(fstream)

    survey, ref_filter  = param_cfg['survey'], param_cfg['ref_filter']
    maglim = param_cfg['maglim_pmem']
    galcat = param_data['galcat'][survey]
    clcat = param_data['clcat'][param_cfg['clusters']]
    out_paths = param_cfg['out_paths']
    admin = param_cfg['admin']
    footprint = param_data['footprint'][survey]
    zp_metrics = param_data['zp_metrics'][survey][ref_filter]
    magstar_file = param_data['magstar_file'][survey][ref_filter]
    workdir = out_paths['workdir']
    data_cls = read_FitsCat(param_data['clcat'][param_cfg['clusters']]['cat'])
    all_tiles = read_FitsCat(
        os.path.join(workdir, admin['tiling']['tiles_filename'])
    )
    tiles = all_tiles[(all_tiles['thread_id']==int(thread_id))]    
    print ('THREAD ', int(thread_id))

    # select correct magnitude 
    for it in range(0, len(tiles)):
        tile_dir = os.path.join(
            workdir, 
            'tiles', 
            'tile_'+str(int(tiles['id'][it])).zfill(3)
        )
        print ('..... Tile ', int(tiles['id'][it]))

        create_directory(tile_dir)
        create_pmem_directories(tile_dir, out_paths['pmem'])
        out_paths['workdir_loc'] = tile_dir # local update 
        #tile_radius_deg = tile_radius_pmem(
        #    admin, param_cfg['pmem_cfg'], param_cfg['cosmo_params']
        #)
        tile_radius_deg = tiles['radius_tile_deg'][it]
        data_gal_tile = read_mosaicFitsCat_in_disc(
            galcat, tiles[it], tile_radius_deg
        )   
        data_gal_tile = data_gal_tile\
                        [data_gal_tile[galcat['keys']['key_mag']]<=maglim]

        data_fp_tile = read_mosaicFootprint_in_disc(
            footprint, tiles[it], tile_radius_deg
        )
        tile_specs = create_tile_specs(
            tiles[it], admin, 
            None, None, 
            data_fp_tile, footprint
        )

        data_cls_disc = filter_disc_tile(data_cls, clcat, tiles[it])
        
        data_cls_tile = []
        for j in range(0, tiles['nhpix'][it]):
            tile_specs = {
                'Nside':tiles['Nside'][it],
                'nest': tiles['nest'][it],
                'hpix': tiles['hpix'][it][j]
            }
            if j == 0:
                data_cls_tile = filter_hpx_tile(data_cls, clcat, tile_specs)
            else:
                data_cls_tile = np.hstack((
                data_cls_tile, 
                filter_hpx_tile(data_cls, clcat, tile_specs)
                ))

        print ('Nr of clusters in tile ', len(data_cls_tile))


        '''
        t = Table (data_gal_tile)#, names=names)
        t.write(os.path.join(tile_dir, "galcat.fits"),overwrite=True)
        t = Table (data_fp_tile)#, names=names)
        t.write(os.path.join(tile_dir, "footprint.fits"),overwrite=True)
        '''

        if len(data_cls_tile) > 0:
            if not os.path.isfile(
                    os.path.join(
                        tile_dir, 
                        out_paths['pmem']['results'], 
                        "richness.fits"
                    )
            ):
                data_richness, data_members = pmem_tile(
                    param_cfg['pmem_cfg'], 
                    data_cls_tile, data_cls, clcat['keys'], 
                    data_fp_tile, footprint, 
                    data_gal_tile, galcat, 
                    zp_metrics['sig_dz0'], param_cfg['cosmo_params'],
                    magstar_file, out_paths, param_cfg['verbose']
                )

                # write outputs to fits
                t = Table (data_richness)#, names=names)
                t.write(
                    os.path.join(
                        tile_dir, 
                        out_paths['pmem']['results'], 
                        "richness.fits"
                    ),overwrite=True
                )
                t = Table (data_members)#, names=names)
                t.write(
                    os.path.join(
                        tile_dir, 
                        out_paths['pmem']['results'], 
                        "pmem.fits"
                    ),overwrite=True
                )
    return


def run_pmem_list(data_cls, config, dconfig, thread_id):
    # read config file
    with open(config) as fstream:
        param_cfg = yaml.safe_load(fstream)
    globals().update(param_cfg)
    with open(dconfig) as fstream:
        param_data = yaml.safe_load(fstream)
    globals().update(param_data)

    survey, ref_filter  = param_cfg['survey'], param_cfg['ref_filter']
    galcat = param_data['galcat'][survey]
    out_paths = param_cfg['out_paths']
    clcat = param_data['clcat'][survey]
    admin = param_cfg['admin']
    footprint = param_data['footprint'][survey]
    zp_metrics = param_data['zp_metrics'][survey][ref_filter]
    magstar_file = param_data['magstar_file'][survey][ref_filter]

    # select correct magnitude 
    galcat['keys']['key_mag'] = galcat['keys']['key_mag'][ref_filter]

    workdir = out_paths['workdir']
    print ('THREAD ', int(thread_id))
    print ('Nr of clusters in thread ', 
           len(data_cls[data_cls['thread_id']==thread_id]))

    thread_dir = os.path.join(
        workdir, 'threads', 'thread_'+str(thread_id).zfill(3)
    )

    create_directory(os.path.join(workdir, 'threads'))
    create_directory(thread_dir)
    create_pmem_directories(thread_dir, out_paths['pmem'])
    out_paths['workdir_loc'] = thread_dir # local update 

    if not os.path.isfile(
            os.path.join(
                thread_dir, out_paths['pmem']['results'], "richness.fits")
    ):
        data_richness, data_members = pmem_list(
            param_cfg['pmem_cfg'], 
            data_cls[data_cls['thread_id']==thread_id], 
            data_cls, clcat['keys'], 
            footprint, galcat, 
            zp_metrics['sig_dz0'], param_cfg['cosmo_params'],
            magstar_file, out_paths, param_cfg['verbose']
        )

        # write outputs to fits
        t = Table (data_richness)#, names=names)
        t.write(
            os.path.join(
                thread_dir, 
                out_paths['pmem']['results'], 
                "richness.fits"
            ),overwrite=True
        )
        t = Table (data_members)#, names=names)
        t.write(
            os.path.join(
                thread_dir, out_paths['pmem']['results'], "pmem.fits"
            ),overwrite=True
        )


def concatenate_calib_dz(all_tiles, pmem_cfg, workdir, calib_file):
    # concatenate all tiles 
    print ('Concatenate Calib files')
    list_calibs = []
    for it in range(0, len(all_tiles)):
        tile_dir = tile_dir_name(
            workdir, int(all_tiles['id'][it]) 
        )
        if os.path.isfile(
                os.path.join(tile_dir, pmem_cfg['calib_dz']['filename'])
        ):
            list_calibs.append(os.path.join(tile_dir))
    # calib_file written on disc
    data_calib = concatenate_clusters(
        list_calibs, pmem_cfg['calib_dz']['filename'], calib_file
    ) 
    return data_calib


def pmem_concatenate_tiles(all_tiles, out_paths, rich_file, pmem_file):
    # concatenate all tiles 
    print ('Concatenate Pmems')
    list_clusters = []
    list_members = []
    for it in range(0, len(all_tiles)):
        tile_dir = tile_dir_name(
            out_paths['workdir'], int(all_tiles['id'][it]) 
        )
        list_clusters.append(
            os.path.join(tile_dir, out_paths['pmem']['results'])
        )
        list_members.append(
            os.path.join(tile_dir, out_paths['pmem']['results'])
        )
    data_richness = concatenate_clusters(
        list_clusters, 'richness.fits', rich_file
    ) 
    data_members  = concatenate_clusters(
        list_members, 'pmem.fits', pmem_file
    )
    return data_richness, data_members


def pmem_concatenate_threads(all_tiles, out_paths, rich_file, pmem_file):
    # concatenate all tiles 
    print ('Concatenate Pmems')
    list_clusters = []
    list_members = []
    for it in range(0, len(all_tiles)):
        tile_dir = thread_dir_name(
            out_paths['workdir'], int(all_tiles['id'][it]) 
        )
        list_clusters.append(
            os.path.join(tile_dir, out_paths['pmem']['results'])
        )
        list_members.append(
            os.path.join(tile_dir, out_paths['pmem']['results'])
        )
    data_richness = concatenate_clusters(
        list_clusters, 'richness.fits', rich_file
    ) 
    data_members  = concatenate_clusters(
        list_members, 'pmem.fits', pmem_file
)
    return data_richness, data_members


def tiles_with_clusters(out_paths, all_tiles):
    flag = np.zeros(len(all_tiles))
    for it in range(0, len(all_tiles)):
        tile_dir = tile_dir_name(
            out_paths['workdir'], int(all_tiles['id'][it]) 
        )
        if os.path.isfile(
                os.path.join(
                    tile_dir, out_paths['pmem']['results'], "richness.fits"
                )
        ):
            flag[it] = 1
    return all_tiles[flag==1]




