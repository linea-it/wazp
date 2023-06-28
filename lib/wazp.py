import matplotlib
matplotlib.use('Agg')
import scipy.constants 
from astropy.io import fits
from astropy import wcs
from astropy.stats import sigma_clip
import  astropy.coordinates
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.cosmology.core import FlatLambdaCDM as flat
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import os
import math
from math import cos 
import healpy as hp
from scipy import interpolate
from scipy.interpolate import griddata
from skimage.feature import peak_local_max
import logging 
import yaml
import subprocess

from .utils import join_struct_arrays, dist_ang
from .utils import _mstar_, makeHealpixMap, radec_window_area
from .utils import area_ann_deg2, hpx_in_annulus, sub_hpix, cond_in_disc
from .utils import create_directory, tile_radius, concatenate_clusters
from .utils import read_mosaicFitsCat_in_disc, read_mosaicFootprint_in_disc
from .utils import read_FitsCat, read_mosaicFitsCat_in_hpix
from .utils import read_mosaicFootprint_in_hpix, add_hpx_to_cat
from .utils import add_clusters_unique_id, create_tile_specs
from .utils import hpx_degrade

def tile_dir_name(workdir, tile_nr):
    return os.path.join(workdir, 'tiles', 'tile_'+str(tile_nr).zfill(3))


def create_wazp_directories(workdir):

    create_directory(workdir)
    create_directory(os.path.join(workdir, 'tiles'))
    create_directory(os.path.join(workdir, 'gbkg'))
    create_directory(os.path.join(workdir, 'calib'))
    create_directory(os.path.join(workdir, 'footprint'))
    create_directory(os.path.join(workdir, 'config'))
    create_directory(os.path.join(workdir, 'tmp'))
    return


def create_tile_directories(root, path): 
    """
    creates the relevant directories for writing results/plots
    """
    if not os.path.exists(os.path.join(root, path['results'])):
        os.mkdir(os.path.join(root, path['results']))
    if not os.path.exists(os.path.join(root, path['plots'])):
        os.mkdir(os.path.join(root, path['plots']))
    if not os.path.exists(os.path.join(root, path['files'])):
        os.mkdir(os.path.join(root, path['files']))
    return


def compute_zpslices(zp_metrics, wazp_cfg, zs_targ, output): 

    sig0 = np.float64(zp_metrics['sig_dz0'])
    zpmin, zpmax = np.float64(zp_metrics['zpmin']), \
                   np.float64(zp_metrics['zpmax'])
    nsamp_slice = np.float64(wazp_cfg['nsamp_slice'])
    nsig = np.float64(wazp_cfg['nsig_dz'])

    zsl=zpmin
    for i in range(0,100):
        sig = (sig0[0] + zsl*sig0[1])*(1+zsl)
        zsl = zsl + sig/nsamp_slice
        if zsl > zpmax:
            break
        imax = i

    zsl = np.zeros(imax+1)
    zsl[0] = zpmin

    for i in range(0,imax):
        sig = (sig0[0] + zsl[i]*sig0[1])*(1+zsl[i])
        im1 = i
        i+=1
        zsl[i] = zsl[im1] + sig/nsamp_slice

    if zs_targ>0.:
        iref = np.argmin(np.absolute(zsl - zs_targ))
        imin = max(0, iref-3)
        imax = min(len(zsl)-1, iref+4)
        zsl = zsl[imin:imax]

    zsl_min = zsl - nsig*(sig0[0] + zsl*sig0[1])*(1.+zsl) 
    zsl_max = zsl + nsig*(sig0[0] + zsl*sig0[1])*(1.+zsl) 
    sig_dz = (sig0[0] + zsl*sig0[1]) * (1.+zsl)
    nsl = len(zsl)

    zsl_min[zsl_min<=0.001] = 0.001
    
    all_cols = fits.ColDefs([
        fits.Column(name='id', format='J',array= np.arange(nsl)),
        fits.Column(name='zsl', format='D',array= zsl),
        fits.Column(name='zsl_min', format='D',array= zsl_min),
        fits.Column(name='zsl_max', format='D',array= zsl_max),
        fits.Column(name='sig_dz', format='D',array= sig_dz)])
    hdu = fits.BinTableHDU.from_columns(all_cols)    
    hdu.writeto(output,overwrite=True)

    return read_FitsCat(output)


def zp_weight_fct(zsl, zsl_min, zsl_max, x):
    sig = (zsl_max - zsl_min)/4.
    if x < zsl-sig:
        weight =  (1./sig)*x + 2. - (1./sig)*zsl
    if x > zsl+sig:
        weight = -(1./sig)*x + 2. + (1./sig)*zsl
    if (x >= zsl-sig) &  (x <= zsl+sig):
        weight = 1.
    return weight

def zp_weight (zsl, zsl_min, zsl_max, zp):
    weight_fct_vec = np.vectorize(zp_weight_fct)
    weight = weight_fct_vec(zsl, zsl_min, zsl_max,zp)
    return weight

def detlum_weight(mag, mstar, power):
    weight = np.ones(len(mag))
    mag1 = mag[(mag<mstar)]
    weight[(mag<mstar)] = 10**((-mag1 + mstar)/power)
    return weight


def lum_weight(mag, mstar, power):
    weight = np.ones(len(mag))
    weight = 10**((-mag + mstar)/power)
    return weight


def bkg_from_hpx_counts (dat_footprint, footprint, ra, dec, weights):

    Nside, nest = footprint['Nside'], footprint['nest']
    ghpx = hp.ang2pix(Nside, ra, dec, nest, lonlat=True)
    fhpx = dat_footprint[footprint['key_pixel']]
    idkept = np.isin(ghpx, fhpx)
    ghpx_kept_unique = np.unique(ghpx[idkept])
    id_count0 = np.isin(
        fhpx, ghpx_kept_unique, assume_unique=True, invert=True
    )
    fhpx_count0 = fhpx[id_count0]
    counts0 = np.zeros(len(fhpx_count0))
    counts = makeHealpixMap(
        ra[idkept], dec[idkept], weights[idkept], Nside, nest
    )
    all_counts = np.concatenate((counts[counts>0], counts0))
    bkg_arcmin2 = np.mean(all_counts)/(3600. *\
                                       hp.nside2pixarea(Nside, degrees=True))
    return bkg_arcmin2


def bkg_tile_slice (dat_galcat, dat_footprint, galcat, footprint, 
                    zpslices, mstar_file, wazp_cfg, cosmo_params, 
                    dmag_faint, weight_mode):

    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
    ra, dec, weight = select_galaxies_in_slice(
        dat_galcat, galcat, wazp_cfg, 
        zpslices, mstar_file, dmag_faint, weight_mode
    )
    conv_factor = cosmo.angular_diameter_distance(zpslices['zsl'])
    area_mpc2 = (np.pi * conv_factor.value / (60. * 180.))**2
    bkg_arcmin2 = bkg_from_hpx_counts(
        dat_footprint, footprint, ra, dec, weight
    )
    bkg_mpc2 = bkg_arcmin2/area_mpc2
    return bkg_arcmin2, bkg_mpc2


def bkg_tile (dat_galcat, dat_footprint, galcat, footprint, 
              zpslices, mstar_file, wazp_cfg, cosmo_params, 
              dmag_faint, weight_mode):

    nsl = len(zpslices)
    bkg_arcmin2, bkg_mpc2 = np.zeros(nsl), np.zeros(nsl)

    for iz in range (0, nsl):
        bkg_arcmin2[iz], bkg_mpc2[iz] = bkg_tile_slice (
            dat_galcat, dat_footprint, galcat, footprint, 
            zpslices[iz], mstar_file, wazp_cfg, cosmo_params, 
            dmag_faint, weight_mode
        )

    data_bkg = np.zeros(nsl, 
                        dtype={   
                            'names':(
                                'zsl', 
                                'bkg_arcmin2', 
                                'bkg_mpc2'
                            ),
                            'formats':(
                                'f4',
                                'f4',
                                'f4'
                            )
                        })
    data_bkg['zsl'] = zpslices['zsl']
    data_bkg['bkg_arcmin2'] = bkg_arcmin2
    data_bkg['bkg_mpc2'] = bkg_mpc2
    return data_bkg


def create_wcs_at_z(wazp_cfg, tile_specs, z, cosmo_params):

    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
    racen, deccen = tile_specs['ra'], tile_specs['dec']
    pix_mpc = 1./float(wazp_cfg['resolution'])
    conv_factor = cosmo.angular_diameter_distance(z)# radian*conv=mpc    
    pix_deg = np.degrees( pix_mpc / conv_factor.value)

    nxy = int(2.*tile_specs['radius_tile_deg']/pix_deg) + 1
    if (nxy % 2) == 0:
        nxy+=1

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nxy/2., nxy/2.]
    w.wcs.cdelt = np.array([-pix_deg, pix_deg])
    w.wcs.crval = [racen, deccen]
    w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    return w, nxy


def vmap_from_hpx (dat_hpx, footprint, tile_specs, wazp_cfg, cosmo_params, zsl):

    Nside, nest = footprint['Nside'], footprint['nest']
    hpx = dat_hpx[footprint['key_pixel']]

    # build vmap image header
    wvmap, nxy = create_wcs_at_z(wazp_cfg, tile_specs, zsl, cosmo_params)
    vmap = np.zeros((nxy,nxy))

    # convert vmap pixels to RA DEC 
    ix, iy = np.linspace(0,nxy-1,nxy).astype(int), np.linspace(0,nxy-1,nxy).astype(int)
    xv0, yv0 = np.meshgrid (ix, iy)
    xvr0, yvr0 = np.ravel(xv0), np.ravel(yv0)
    ra_map0, dec_map0 = wvmap.all_pix2world(xvr0+0.5,yvr0+0.5,1)[0],\
                        wvmap.all_pix2world(xvr0+0.5,yvr0+0.5,1)[1]

    # get mask info for each pixel of vmap
    hpx_map = hp.ang2pix(Nside, ra_map0, dec_map0, nest, lonlat=True)
    vmap[xvr0[np.isin(hpx_map, hpx)], yvr0[np.isin(hpx_map, hpx)]] = 1.
    return vmap

def map2fits(imap, wazp_cfg, tile, zsl, cosmo_params, fitsname):

    w, nxy = create_wcs_at_z(wazp_cfg, tile, zsl, cosmo_params)
    # write xycat_fi to a file for mr_filter
    header = w.to_header()
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = imap.T 
    hdu.writeto(fitsname, overwrite=True)
    return


def map_detlum_weight(mag, mstar, wazp_cfg):

    weight = np.ones(len(mag))
    if wazp_cfg['map_lum_weight_mode'] == True:
        weight = detlum_weight(mag, mstar, wazp_cfg['lum_weight_map_power'])
    return weight

def map_lum_weight(mag, mstar, wazp_cfg):

    weight = np.ones(len(mag))
    if wazp_cfg['map_lum_weight_mode'] == True:
        weight = lum_weight(mag, mstar, wazp_cfg['lum_weight_map_power'])
    return weight


def pixelized_radec(ra_map, dec_map, weight_map, w, nxy):

    xmap=w.all_world2pix(ra_map,dec_map,1)[0] - 0.5
    ymap=w.all_world2pix(ra_map,dec_map,1)[1] - 0.5
    xycat, xedges, yedges = np.histogram2d(
        np.round(xmap, 1), np.round(ymap, 1), 
        bins=nxy, range=((0,nxy),(0,nxy)), weights=weight_map
    )
    return xycat


def select_galaxies_in_slice(dat_galcat, galcat, wazp_cfg, zpslices, 
                             mstar_file, dmag_faint, weight_mode):
    ra, dec = dat_galcat[galcat['keys']['key_ra']],\
              dat_galcat[galcat['keys']['key_dec']]
    mag = dat_galcat[galcat['keys']['key_mag']]
    zp =  dat_galcat[galcat['keys']['key_zp']]
    
    zsl_min, zsl_max = zpslices['zsl_min'], zpslices['zsl_max']
    mstar = _mstar_ (mstar_file,  zpslices['zsl'])
    #dmag_det =    wazp_cfg['dmag_det']
    dmag_bright = wazp_cfg['dmag_bright']
    mag_min, mag_max_det = mstar - dmag_bright, mstar + dmag_faint

    cond_for_wmap = ((zp<np.float64(zsl_max)) & 
                     (zp>np.float64(zsl_min)) &
                     (mag<=np.float64(mag_max_det)) & 
                     (mag>np.float64(mag_min)))  
    ra_map, dec_map = ra[cond_for_wmap], dec[cond_for_wmap]
    mag_map, zp_map = mag[cond_for_wmap], zp[cond_for_wmap]
    
    if weight_mode == "none":
        weight_map = np.ones(len(mag_map))
    if weight_mode == "detlum":
        weight_map = map_detlum_weight(mag_map, mstar, wazp_cfg)
    if weight_mode == "lum":
        weight_map = map_lum_weight(mag_map, mstar, wazp_cfg)
    if weight_mode == "zplum":
        weight_map = zp_weight(
            zpslices['zsl'], 
            zsl_min, zsl_max, zp_map
        ) * map_lum_weight(mag_map, mstar, wazp_cfg)
    if weight_mode == "zp":
        weight_map = zp_weight(zpslices['zsl'], zsl_min, zsl_max, zp_map)

    return ra_map, dec_map, weight_map


def select_zps_cylinder(dcyl, dgal, galcat, zpslices, mstar_file, cosmo, wazp_specs):

    rad_zdet, dmag_zdet = wazp_specs['rad_zdet'], wazp_specs['dmag_zdet']

    nsig=3
    sig_dz = (zpslices["zsl_max"] - zpslices["zsl_min"])/4. 
    zsl_min, zsl_max = zpslices['zsl_min'], zpslices['zsl_max']
    zsl = zpslices['zsl']

    # get galaxies 
    rag, decg = dgal[galcat['keys']['key_ra']],  dgal[galcat['keys']['key_dec']]
    magg, zpg = dgal[galcat['keys']['key_mag']], dgal[galcat['keys']['key_zp']]
    magstar = _mstar_ (mstar_file, dcyl['z_init']) 
    maglim = magstar + dmag_zdet
    zmin_z = zsl[dcyl['cyl_isl_min']] - nsig*sig_dz[dcyl['cyl_isl_min']]
    zmax_z = zsl[dcyl['cyl_isl_max']] + nsig*sig_dz[dcyl['cyl_isl_max']]
    cond =  ((zpg<zmax_z) & (zpg>zmin_z) & (magg<maglim))
    conv_factor = cosmo.angular_diameter_distance(dcyl['z_init'])# radian*conv=mpc    
    rad_deg = np.degrees(rad_zdet / conv_factor.value)
    cond_disc = cond_in_disc(rag[cond], decg[cond], dgal['hpx_tmp'][cond], 
                             wazp_specs['Nside_tmp'], wazp_specs['nest_tmp'], 
                             dcyl['ra'], dcyl['dec'], rad_deg)

    return zpg[cond][cond_disc], zpg[cond][~cond_disc]


def select_cells_from_footprint(Nside_cell, data_fp, footprint):

    hpix_map, frac_map = data_fp[footprint['key_pixel']],\
                         data_fp[footprint['key_frac']]
    pix_out, frac_out = hpx_degrade(
        hpix_map, frac_map, footprint['Nside'], footprint['nest'], 
        Nside_cell, footprint['nest']
    )
    return pix_out[frac_out>0.99]
        

def stats_counts_in_cells(counts, ncmax):

    if ncmax is not None:
        if len(counts) > ncmax:
            counts = np.random.choice(counts, size=ncmax, replace=False)
    else:
        counts = np.random.choice(counts, size=len(counts), replace=False)

    ngcmax = np.amax(counts)
    ncells = len(counts)
    p = np.zeros(ngcmax+1) # P(N)
    ng, nc = np.unique(counts, return_counts=True)
    p[ng] = nc
    p = p.astype(float)/float(ncells)    
    nbar, mu2 = 0., 0.
    for ng in range(0,ngcmax+1):
        nbar += float(ng)*p[ng]
    for ng in range(0,ngcmax+1):
        mu2 += (float(ng) - nbar)**2 * p[ng] / nbar**2
    ksi2b = mu2 - 1./nbar    
    return nbar, ksi2b


def counts_in_cells(ra, dec, Nside, data_fp, footprint, ncmax):

    # select galaxies in cells 
    hpix_cells_inf = select_cells_from_footprint(Nside, data_fp, footprint)
    hpixg = hp.ang2pix(Nside, ra, dec, footprint['nest'], lonlat=True)
    hpixgc = hpixg[np.isin(hpixg, hpix_cells_inf)]
    # counts in cells 
    hpixgc_unique, counts = np.unique(hpixgc, return_counts=True)
    hpixgc_nogal = hpix_cells_inf[np.isin(hpix_cells_inf, hpixgc_unique, invert=True)]
    counts_all = np.hstack((counts, np.zeros(len(hpixgc_nogal)).astype(int)))
    return  stats_counts_in_cells(counts_all, ncmax)


def bkg_global_survey(galcat, footprint, tiles_filename, zpslices_filename, 
                      tiling, cosmo_params, mstar_file, wazp_cfg, output):

    area_min = wazp_cfg['gbkg_area']
    # select galcat and associated footprint to reach some minimum area in the survey 
    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
    tiles = read_FitsCat(tiles_filename)
    zpslices = read_FitsCat(zpslices_filename)
    irev = np.argsort(-tiles['eff_area_deg2'])
    area = tiles['eff_area_deg2'][irev]
    hpix = tiles['hpix'][irev]

    print ('.... min area for bkg (deg2) = ', area_min)

    if np.sum(area) <= area_min:
        nt = len(area)
    else:
        nt = np.argwhere(np.cumsum(area)>area_min).T[0][0]+1

    for i in range(0, nt): 
        data_gal_hpix = read_mosaicFitsCat_in_hpix(
            galcat, hpix[i], tiling['Nside'], tiling['nest']
        )   
        data_fp_hpix = read_mosaicFootprint_in_hpix(
            footprint, hpix[i], tiling['Nside'], tiling['nest']
        )
        if i == 0:
            data_gal = np.copy(data_gal_hpix)
            data_fp =  np.copy(data_fp_hpix)
        else:
            data_gal = np.hstack((data_gal_hpix, data_gal))
            data_fp  = np.hstack((data_fp_hpix, data_fp))

    bkg_global(data_gal, galcat, data_fp, footprint, 
               zpslices_filename, cosmo_params, 
               mstar_file, wazp_cfg, output)
    return


def bkg_global(data_gal, galcat, data_fp, footprint, 
               zpslices_filename, cosmo_params, 
               mstar_file, wazp_cfg, output):


    ncmax = wazp_cfg['ncmax']
    beta_default = -0.8
    radius_snr_mpc = wazp_cfg['radius_snr_mpc']
    # select galcat and associated footprint to reach 
    # some minimum area in the survey 
    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
    zpslices = read_FitsCat(zpslices_filename)

    beta_snr, ksi2_snr = np.zeros(len(zpslices)), np.zeros(len(zpslices))
    nbar_rich, nbar_snr = np.zeros(len(zpslices)), np.zeros(len(zpslices))
    nbar0_rich, nbar0_snr = np.zeros(len(zpslices)), np.zeros(len(zpslices))
    lbar0_rich, lbar0_snr = np.zeros(len(zpslices)), np.zeros(len(zpslices))
    lbar_rich, lbar_snr = np.zeros(len(zpslices)), np.zeros(len(zpslices))

    # effective used area 
    area_eff = float(len(data_fp)) *\
               hp.nside2pixarea(footprint['Nside'], degrees=True)

    # loop over zsl and select resolution 
    jinf, jsup = np.zeros(len(zpslices)).astype(int),\
                 np.zeros(len(zpslices)).astype(int)
    cell_area = np.zeros(len(zpslices))
    for i in range(0, len(zpslices)):
        conv_factor = cosmo.angular_diameter_distance( zpslices['zsl'][i])
        radius_snr_deg = np.degrees(radius_snr_mpc / conv_factor.value)
        cell_area[i] = np.pi * radius_snr_deg**2
        for j in range(0, 15):
            if cell_area[i] >= hp.nside2pixarea(2**j, degrees=True):
                jsup[i] = j
                jinf[i] = (j-1)
                break

    # degrade footprint to cell Nside and keep hpixels with detfrac = 1
    for i in range(0, len(zpslices)):
        # stats - for wazp_snr
        # weight modes should be same as in add_peaks_attributes 
        ra, dec, Nweight = select_galaxies_in_slice(
            data_gal, galcat, wazp_cfg, zpslices[i], 
            mstar_file, wazp_cfg['dmag_rich'], 'none'
        )
        ra, dec, Lweight = select_galaxies_in_slice(
            data_gal, galcat, wazp_cfg, zpslices[i], 
            mstar_file, wazp_cfg['dmag_rich'], 'lum'
        )
        #   mean number of galaxies / cell weighted by lum or not
        lbar0_rich[i] = cell_area[i] * np.sum(Lweight)/area_eff
        nbar0_rich[i] = cell_area[i] * np.sum(Nweight)/area_eff

        # stats - for wazp_richness
        # weight modes should be same as in add_peaks_attributes 
        ra, dec, Nweight = select_galaxies_in_slice(
            data_gal, galcat, wazp_cfg, zpslices[i], 
            mstar_file, wazp_cfg['dmag_det'], 'zp'
        )
        ra, dec, Lweight = select_galaxies_in_slice(
            data_gal, galcat, wazp_cfg, zpslices[i], 
            mstar_file, wazp_cfg['dmag_det'], 'zplum'
        )
        lbar0_snr[i] = cell_area[i] *  np.sum(Lweight)/area_eff
        nbar0_snr[i] = cell_area[i] * np.sum(Nweight)/area_eff

        # From counts in cells compute ksi2 without considering weights 
        # used as a ref for Ncounts and Lcounts 
        # used to compute SNR @ dmag_det
        # select galaxies in slice with dmag_rich and no lum weight
        ra, dec, weight = select_galaxies_in_slice(
            data_gal, galcat, wazp_cfg, zpslices[i], 
            mstar_file, wazp_cfg['dmag_det'], 'none'
        )

        #stats Counts in Cells 
        nbar_inf, ksi2_inf = counts_in_cells(
            ra, dec, 2**jinf[i], data_fp, footprint, wazp_cfg['ncmax']
        )
        nbar_sup, ksi2_sup = counts_in_cells(
            ra, dec, 2**jsup[i], data_fp, footprint, wazp_cfg['ncmax']
        )

        #interpolate at nominal aperture area
        if ksi2_inf>0. and ksi2_sup>0:
            beta_snr[i] = 2.*np.log(ksi2_sup/ksi2_inf) /\
                          np.log(hp.nside2pixarea(2**jsup[i])/\
                                 hp.nside2pixarea(2**jinf[i]))
            lg_ksi2 = np.log(ksi2_sup) +\
                      np.log(ksi2_inf/ksi2_sup)*\
                      np.log(cell_area[i]/\
                             hp.nside2pixarea(2**jsup[i], degrees=True))/\
                      np.log(
                          hp.nside2pixarea(2**jinf[i])/\
                          hp.nside2pixarea(2**jsup[i])
                      )
        else:
            beta_snr[i] = beta_default
            lg_ksi2 = max(ksi2_inf, ksi2_sup)

        nbar_snr[i] = nbar_sup*cell_area[i]/\
                      hp.nside2pixarea(2**jsup[i], degrees=True)
        ksi2_snr[i] = np.exp(lg_ksi2)
        
    # write output 
    all_cols = fits.ColDefs([
        fits.Column(name='zp', format='D',array= zpslices['zsl']),
        fits.Column(name='nbar_rich', format='D',array= nbar0_rich),
        fits.Column(name='nbar_snr', format='D',array= nbar0_snr),
        fits.Column(name='lbar_rich', format='D',array= lbar0_rich),
        fits.Column(name='lbar_snr', format='D',array= lbar0_snr),
        fits.Column(name='ksi2', format='D',array= ksi2_snr),
        fits.Column(name='slope', format='D',array= beta_snr)])
    hdu = fits.BinTableHDU.from_columns(all_cols)    
    hdu.writeto(output, overwrite=True)
    return read_FitsCat(output) 


def compute_catimage(ra_map, dec_map, weight_map, zpslices, wazp_cfg, tile_specs, 
                     cosmo_params):
    w, nxy = create_wcs_at_z(wazp_cfg, tile_specs, zpslices['zsl'], cosmo_params)
    xycat = pixelized_radec (ra_map, dec_map, weight_map, w, nxy)
    return xycat


def randoms_in_spherical_cap(tile, bkg_arcmin2):

    area_sph = 3600.*4*np.pi*(np.degrees(1.))**2 # arcmin2
    for i in range(0, 20):
        nside = hp.order2nside(i)
        pdens = float(hp.nside2npix(nside))/area_sph
        if pdens >= bkg_arcmin2:
            Nside_samp = nside#*2
            break
    pixels_in_disc = hp.query_disc(
        nside=Nside_samp, nest=False, 
        vec=hp.ang2vec(tile['ra'], tile['dec'], lonlat=True),
        radius = np.radians(tile['radius_tile_deg']), 
        inclusive=False
    )
    area = 3600.*area_ann_deg2(0., tile['radius_tile_deg'])
    nsamp = int(bkg_arcmin2*area)
    pix_samp = np.random.choice(pixels_in_disc, nsamp, replace=False) 

    return hp.pix2ang(Nside_samp, pix_samp, False, lonlat=True)


def compute_filled_catimage(ra_map, dec_map, weight_map, zpslices, wazp_cfg, tile, 
                            cosmo_params, data_footprint, footprint, bkg_arcmin2):

    # find edge pixels of the footprint 
    nlist = hp.get_all_neighbours(
        footprint['Nside'], 
        data_footprint[footprint['key_pixel']], None, footprint['nest']
    )
    mask = np.isin(nlist, data_footprint[footprint['key_pixel']])
    edge_pixels = data_footprint[footprint['key_pixel']][(np.sum(mask, axis=0)<8)]

    # find empty footprint pixels at resolution Nside*2
    sub_edge_hpix = sub_hpix(edge_pixels, footprint['Nside'], footprint['nest'])
    shpix_filled = np.unique(
        hp.ang2pix(
            footprint['Nside']*2, 
            ra_map, dec_map, footprint['nest'], lonlat=True
        )
    )
    hpix_filled = np.unique(
        hp.ang2pix(
            footprint['Nside'], 
            ra_map, dec_map, footprint['nest'], lonlat=True
        )
    )

    empty_edge_hpix = edge_pixels[np.isin(edge_pixels, hpix_filled, invert=True)]
    empty_sub_edge_hpix = sub_edge_hpix[np.isin(sub_edge_hpix, shpix_filled, invert=True)]
    # keep only sub pixels not neighbouring the pixels with actual galaxies 
    nlist_esep = hp.get_all_neighbours(
        2*footprint['Nside'], empty_sub_edge_hpix, None, footprint['nest']
    )
    mask_esep  = np.isin(nlist_esep, shpix_filled, invert=True)
    empty_sub_edge_hpix = empty_sub_edge_hpix[(np.sum(mask_esep, axis=0)==8)]
    
    # build uniform randoms in spherical cap with proper density 
    ra_ran, dec_ran = randoms_in_spherical_cap(tile, bkg_arcmin2)

    # keep randoms outside of the footprint or in empty sub-edge pixels
    hpx_ran1 = hp.ang2pix(
        footprint['Nside'], 
        ra_ran, dec_ran, footprint['nest'], lonlat=True
    )
    ra_ranf1 = ra_ran[np.isin(
        hpx_ran1, data_footprint[footprint['key_pixel']], invert=True
    )]
    dec_ranf1 = dec_ran[np.isin(
        hpx_ran1, data_footprint[footprint['key_pixel']], invert=True
    )]
    hpx_ran2 = hp.ang2pix(
        footprint['Nside']*2, 
        ra_ran, dec_ran, footprint['nest'], lonlat=True
    )
    ra_ranf2 =   ra_ran[np.isin(hpx_ran2, empty_sub_edge_hpix)]
    dec_ranf2 = dec_ran[np.isin(hpx_ran2, empty_sub_edge_hpix)]

    # stack galaxies + randoms 
    ra_ranf1, dec_ranf1 = np.hstack((ra_ranf1, ra_ranf2)),\
                          np.hstack((dec_ranf1, dec_ranf2))
    ra_all, dec_all   = np.hstack((ra_ranf1,  ra_map)),\
                        np.hstack((dec_ranf1,  dec_map))
    weight_all = np.hstack((np.ones(len(ra_ranf1)), weight_map))

    # build catalogue image 
    w, nxy = create_wcs_at_z(wazp_cfg, tile, zpslices['zsl'], cosmo_params)
    xycat = pixelized_radec (ra_all, dec_all, weight_all, w, nxy)
    
    return xycat


def run_mr_filter(filled_catimage, wmap, wazp_cfg):
    
    path_mr = wazp_cfg['path_mr_filter']

    # run mr_filter to build wavelet map
    scale_min_pix = wazp_cfg['scale_min_mpc'] * float(wazp_cfg['resolution'])
    scale_max_pix = wazp_cfg['scale_max_mpc'] * float(wazp_cfg['resolution'])
    smin = int(round(math.log10(scale_min_pix)/math.log10(2.)))
    smax = int(round(math.log10(scale_max_pix)/math.log10(2.)))


    if smin == 0:
        subprocess.run((
            os.path.join(path_mr, 'mr_filter')+\
            ' -m 10 -i 3 -s 3.,3. -n '+str(smax+1)+\
            ' -f 3 -K -C 2 -p -e0 -A '+\
            filled_catimage+' '+wmap
        ), check=True, shell=True, stdout=subprocess.PIPE).stdout

    if smin == 1:
        subprocess.run((
            os.path.join(path_mr, 'mr_filter')+\
            ' -m 10 -i 3 -s 10.,3.,3. -n '+str(smax+1)+\
            ' -f 3 -K -C 2 -p -e0 -A '+\
            filled_catimage+' '+wmap
        ), check=True, shell=True, stdout=subprocess.PIPE).stdout
    if smin == 2:
        subprocess.run((
            os.path.join(path_mr, 'mr_filter')+\
            ' -m 10 -i 3 -s 10.,10.,3.,3. -n '+str(smax+1)+\
            ' -f 3 -K -C 2 -p -e0 -A '+\
            filled_catimage+' '+wmap
        ), check=True, shell=True, stdout=subprocess.PIPE).stdout
    if smin == 3:
        subprocess.run((
            os.path.join(path_mr, 'mr_filter')+\
            ' -m 10 -i 3 -s 10.,10.,10.,3.,3. -n '+str(smax+1)+\
            ' -f 3 -K -C 2 -p -e0 -A '+\
            filled_catimage+' '+wmap
        ), check=True, shell=True, stdout=subprocess.PIPE).stdout
    return


def fits2map(wmap):
    hdulist = fits.open(wmap,ignore_missing_end=True)
    hdu = hdulist[0]
    wmap_t = hdulist[0].data
    w = wcs.WCS(hdu.header)
    wmap_data = wmap_t.T
    return wmap_data


def wmap2peaks(wmap, wazp_specs, tile_specs, zsl, cosmo_params):
    wmap_thresh = wazp_specs['wmap_thresh']
    wmap_data = fits2map(wmap)
    w, nxy = create_wcs_at_z(wazp_specs, tile_specs, zsl, cosmo_params)

    # peak detection on wmap 
    local_maxi = peak_local_max(wmap_data, min_distance=8)  # to be reviewed 
    npeaks0 = len(local_maxi) # nr of peaks in the filled_map 
    iobj0, jobj0 = local_maxi[:,0]+1., local_maxi[:,1]+1. # pixel coord of the peaks
    iobj = iobj0[(wmap_data[iobj0.astype(int),jobj0.astype(int)]> wmap_thresh) ]
    jobj = jobj0[(wmap_data[iobj0.astype(int),jobj0.astype(int)]> wmap_thresh) ]
    ra_peak, dec_peak =  w.all_pix2world(iobj,jobj,1)[0],\
                         w.all_pix2world(iobj,jobj,1)[1]
    return ra_peak, dec_peak, iobj, jobj


def filter_peaks(tile, zsl, cosmo_params, resolution, ra0, dec0, ip0, jp0):

    if tile['nhpix']>0: # not target mode
        cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
        err_mpc = (2./float(resolution))   # +/- 2 pixels around the tile 
        conv_factor = cosmo.angular_diameter_distance(zsl)
        err_deg = np.degrees( err_mpc/ conv_factor.value)
        dx, dy = err_deg*np.cos(np.radians(dec0)), err_deg
        
        ghpx  = hp.ang2pix(
            tile['Nside'], ra0,    dec0,    tile['nest'], lonlat=True
        )
        ghpx1 = hp.ang2pix(
            tile['Nside'], ra0-dx, dec0-dy, tile['nest'], lonlat=True
        )
        ghpx2 = hp.ang2pix(
            tile['Nside'], ra0-dx, dec0+dy, tile['nest'], lonlat=True
        )
        ghpx3 = hp.ang2pix(
            tile['Nside'], ra0+dx, dec0+dy, tile['nest'], lonlat=True
        )
        ghpx4 = hp.ang2pix(
            tile['Nside'], ra0+dx, dec0-dy, tile['nest'], lonlat=True
        )
        cond_filter = ((np.isin(ghpx,  tile['hpix'][0:tile['nhpix']])) | 
                       (np.isin(ghpx1, tile['hpix'][0:tile['nhpix']])) | 
                       (np.isin(ghpx2, tile['hpix'][0:tile['nhpix']])) | 
                       (np.isin(ghpx3, tile['hpix'][0:tile['nhpix']])) | 
                       (np.isin(ghpx4, tile['hpix'][0:tile['nhpix']])))
    else:
        cond_filter = (np.degrees(dist_ang(ra0, dec0, tile['ra'], tile['dec'])) <= 
                       tile['radius_filter_deg'])
    ra, dec = ra0[cond_filter], dec0[cond_filter]
    ip, jp  = ip0[cond_filter], jp0[cond_filter]
    return    ra, dec, ip, jp 


def coverfrac_disc(ra, dec, data_fp, footprint, radius_deg):
    
    coverfrac = np.zeros(len(ra))
    for i in range(0, len(ra)):
        hpx_in_ann, frac_in_ann, area_deg2, coverfrac[i] = hpx_in_annulus (
            ra[i], dec[i], 0., radius_deg, 
            data_fp, footprint, False
        )
    return coverfrac


def init_peaks_table(ra_peaks, dec_peaks, iobj, jobj, coverfrac, wradius, zsl):
    """
    initialize the table of peaks
    """
    data_peaks = np.zeros( (len(ra_peaks)), 
          dtype={
              'names':(
                  'id', 
                  'ra', 'dec', 
                  'iobj', 'jobj', 
                  'z', 
                  'coverfrac', 
                  'wradius_mpc'
              ),
              'formats':(
                  'i8', 
                  'f8', 'f8', 
                  'f4', 'f4', 
                  'f4', 
                  'f4', 
                  'f4'
              )
          }) 
    data_peaks['id'] = np.arange(len(ra_peaks))
    data_peaks['ra']   = ra_peaks
    data_peaks['dec']  = dec_peaks
    data_peaks['iobj']   = iobj
    data_peaks['jobj']  = jobj
    data_peaks['coverfrac']    = coverfrac
    data_peaks['wradius_mpc']    = wradius
    data_peaks['z']    = zsl*np.ones(len(ra_peaks))

    return Table(data_peaks)


def compute_flux_aper(rap, decp, hpix, weight, aper, Nside, nest):

    pixels_in_disc = hp.query_disc(
        nside=Nside, nest=nest, 
        vec=hp.ang2vec(rap, decp, lonlat=True),
        radius = np.radians(aper), inclusive=False
    )
    Nraw = np.sum(weight[np.isin(hpix, pixels_in_disc)])
    pixelized_area = float(len(pixels_in_disc)) *\
                     hp.nside2pixarea(Nside, degrees=True)
    return Nraw*area_ann_deg2(0., aper)/pixelized_area


def compute_flux_aper_vec(rap, decp, aper, dat_galcat, galcat, wazp_cfg, 
                          zpslices, mstar_file, dmag_faint, weight_mode):

    ra, dec, weight = select_galaxies_in_slice(
        dat_galcat, galcat, 
        wazp_cfg, zpslices, mstar_file, dmag_faint, weight_mode
    )

    Nside = 16384
    nest = False
    hpix = hp.ang2pix(Nside, ra, dec, nest, lonlat=True)
    Naper = np.zeros(len(rap))
    for i in range(0, len(rap)):
        Naper[i] = compute_flux_aper(
            rap[i], decp[i], hpix, weight, aper, Nside, nest
        )

    return Naper


def add_peaks_attributes(data_peaks, dat_galcat, galcat, 
                         dat_footprint, footprint, 
                         wazp_cfg, zpslices, gbkg, 
                         mstar_file, aper_deg, cosmo_params):

    Nbkg_rich, Lbkg_rich = gbkg['nbar_rich'], gbkg['lbar_rich']
    Nbkg_snr, Lbkg_snr = gbkg['nbar_snr'], gbkg['lbar_snr']
    sig_N =  (gbkg['nbar_snr']*(1.+gbkg['ksi2']))**0.5
    sig_L =  (gbkg['lbar_snr']*(1.+gbkg['ksi2']))**0.5

    area = 3600.*area_ann_deg2(0., aper_deg) # arcmin2
    Nraw_rich = compute_flux_aper_vec(
        data_peaks['ra'], data_peaks['dec'], aper_deg, 
        dat_galcat, galcat, wazp_cfg, zpslices, mstar_file, 
        wazp_cfg['dmag_rich'], 'none'
    )
    Lraw_rich = compute_flux_aper_vec(
        data_peaks['ra'], data_peaks['dec'], aper_deg, 
        dat_galcat, galcat, wazp_cfg, zpslices, mstar_file, 
        wazp_cfg['dmag_rich'], 'lum'
    )
    Nraw_snr = compute_flux_aper_vec(
        data_peaks['ra'], data_peaks['dec'], aper_deg, 
        dat_galcat, galcat, wazp_cfg, zpslices, mstar_file, 
        wazp_cfg['dmag_det'], 'zp'
    )
    Lraw_snr = compute_flux_aper_vec(
        data_peaks['ra'], data_peaks['dec'], aper_deg, 
        dat_galcat, galcat, wazp_cfg, zpslices, mstar_file, 
        wazp_cfg['dmag_det'], 'zplum'
    )

    snr_n = np.maximum((Nraw_snr - Nbkg_snr) / sig_N, np.zeros(len(Nraw_snr)))
    snr_l = np.maximum((Lraw_snr - Lbkg_snr) / sig_L, np.zeros(len(Nraw_snr)))
    snr = (snr_n * snr_l)**0.5 
    rank_n = np.maximum((Nraw_snr - Nbkg_snr) / sig_N, np.zeros(len(Nraw_snr)))
    rank_l = np.maximum((Lraw_snr - Lbkg_snr) / sig_L, np.zeros(len(Nraw_snr)))
    rank = (rank_n * rank_l)**0.5 

    Naper_snr = Nraw_snr - Nbkg_snr
    Laper_snr = Lraw_snr - Lbkg_snr
    Naper_rich = Nraw_rich - Nbkg_rich
    Laper_rich = Lraw_rich - Lbkg_rich
    
    # add to Table 
    data_peaks['snr'] = snr
    data_peaks['snr_n'] = snr_n
    data_peaks['snr_l'] = snr_l
    data_peaks['rank'] = rank

    data_peaks['Naper_rich'] = Naper_rich
    data_peaks['Laper_rich'] = Laper_rich
    data_peaks['Naper_snr'] = Naper_snr
    data_peaks['Laper_snr'] = Laper_snr

    data_peaks['Nbkg_snr'] = Nbkg_snr*np.ones(len(Naper_snr))
    data_peaks['Lbkg_snr'] = Lbkg_snr*np.ones(len(Naper_snr))
    data_peaks['Nbkg_rich'] = Nbkg_rich*np.ones(len(Naper_snr))
    data_peaks['Lbkg_rich'] = Lbkg_rich*np.ones(len(Naper_snr))

    return


def init_cylinders (keyrank, peak_ids, wazp_specs):

    ncyl = keyrank.shape[0]
    nsl = keyrank.shape[1]
    
    isl_mode = np.zeros(ncyl).astype(int)
    ip = np.zeros(ncyl).astype(int)
    isl_min = np.zeros(ncyl).astype(int)
    isl_max = (nsl-1) * np.ones(ncyl).astype(int)

    for i in range(0,ncyl):
        keyrank_i = keyrank[i,]
        isl_mode[i] = np.argmax(keyrank_i)
        ip[i] = peak_ids[i, isl_mode[i]]
        if isl_mode[i] < nsl-1:
            for isl in range(isl_mode[i]+1,nsl):
                if keyrank_i[isl] < 0.:
                    isl_max[i] = isl - 1
                    break
        else:
            isl_max[i] = isl_mode[i]

        if isl_mode[i] > 0:
            for isl in range(isl_mode[i]-1,-1,-1):
                if keyrank_i[isl] < 0.:
                    isl_min[i] = isl + 1
                    break
        else:
            isl_min[i] = isl_mode[i]

    icyl = np.arange(ncyl)

    data_cylinders = np.zeros((ncyl), 
    dtype={
              'names':('id', 'peak_id', 'cyl_length', 
                       'cyl_isl_min', 'cyl_isl_max', 'cyl_isl_mode'),
              'formats':('i4', 'i4', 'i4', 'i4', 'i4', 'i4')})
    data_cylinders['id'] = icyl
    data_cylinders['peak_id'] = ip
    data_cylinders['cyl_length']    = isl_max - isl_min + 1
    data_cylinders['cyl_isl_min']   = isl_min
    data_cylinders['cyl_isl_max']   = isl_max
    data_cylinders['cyl_isl_mode']   = isl_mode

    return data_cylinders


def key_cylinder(key_cyl, key_1, length, i0, i1, isl, nslices, type):

    if type == "int":
        key_match=-np.ones(length).astype(int)
    else:
        key_match=-np.ones(length)   
    key_match[i0] = key_1[i1]
    key_cyl[:,isl] = key_match

    # append peaks that were not matched
    key_new = key_1[~np.isin(np.arange(key_1.size), i1)]
    if type == "int":
        key_cyl_new = -np.ones((len(key_new),nslices)).astype(int)
    else:
        key_cyl_new = -np.ones((len(key_new),nslices))
    key_cyl_new[:,isl] = key_new
    key_cyl = np.concatenate((key_cyl,key_cyl_new))

    return key_cyl

def append_peaks_infos_to_cylinders(data_cylinders_init, peaks_list, zpslices_specs, 
                                    ip_cyl, ra_cyl, dec_cyl, rank_cyl, snr_cyl):    


    zsl = zpslices_specs['zsl']
    nsl = len(zsl)
    z_init = zsl[data_cylinders_init['cyl_isl_mode']]
    ncl = len(z_init)

    snr = np.zeros(ncl)
    ra = np.zeros(ncl)
    dec = np.zeros(ncl)

    for icl in range(0, ncl):
        isl = data_cylinders_init['cyl_isl_mode'][icl]
        ip = data_cylinders_init['peak_id'][icl]
        snr[icl] = peaks_list[isl]['snr'][ip]
        ra[icl] = peaks_list[isl]['ra'][ip]
        dec[icl] = peaks_list[isl]['dec'][ip]
        
    data_extra = np.zeros(
        (ncl), dtype={
            'names':(
                'ra', 'dec', 'z_init', 'snr', 
                'ip_cyl', 'ra_cyl', 'dec_cyl', 
                'rank_cyl', 'snr_cyl'
            ),
            'formats':(
                'f8','f8','f8', 'f8', 
                str(nsl)+'i8', str(nsl)+'f8', str(nsl)+'f8', 
                str(nsl)+'f8', str(nsl)+'f8' 
            ) 
        }
    )

    data_extra['z_init'] = z_init
    data_extra['snr'] = np.around(snr, 3)
    data_extra['ra'] = ra
    data_extra['dec'] = dec
    data_extra['ip_cyl'] = ip_cyl
    data_extra['ra_cyl'] = ra_cyl
    data_extra['dec_cyl'] = dec_cyl
    data_extra['rank_cyl'] = np.around(rank_cyl, 3)
    data_extra['snr_cyl'] = np.around(snr_cyl, 3)

    arrays = [data_cylinders_init, data_extra]
    data_cylinders = join_struct_arrays(arrays)

    return data_cylinders


def make_cylinders(peaks_list, zpslices_specs, wazp_specs, cosmo_params ):

    rad_mpc = wazp_specs['radius_slice_matching']
    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
    zsl = zpslices_specs['zsl']

    flag_min = 0
    npeaks_all = 0
    nsl = len(peaks_list)

    for isl in range(0,nsl):
        npeaks_all += len(peaks_list[isl])

    for isl in range(0,nsl):
        if flag_min == 0:
            if len(peaks_list[isl]) > 0:
                islmin = isl
                flag_min = 1

        if flag_min == 1:
            if len(peaks_list[isl])== 0: # no detection in intermediate slice 
                continue

            if isl == islmin: # all peaks are new cylinders 
                dat = peaks_list[isl]  
                np0 = len(dat)
                # initialize output 
                ip_cyl = -np.ones((np0,nsl)).astype(int)
                ip_cyl[:,islmin] = np.arange(np0)
                rank_cyl, snr_cyl =np.ones((np0,nsl))*(-1), np.ones((np0,nsl))*(-1)
                rank_cyl[:,islmin], snr_cyl[:,islmin] = dat['rank'], dat['snr']
                ra_cyl, dec_cyl =np.ones((np0,nsl))*(-1), np.ones((np0,nsl))*(-1)
                ra_cyl[:,islmin], dec_cyl[:,islmin] = dat['ra'], dat['dec']
            else:
                if len(peaks_list[isl-1])== 0: # no detection in intermediate slice 
                    dat = peaks_list[isl]  
                    np0 = len(dat)
                    ip_cyl_new = -np.ones((np0,nsl)).astype(int)
                    ip_cyl_new[:,isl] = np.arange(np0)
                    rank_cyl_new, snr_cyl_new = np.ones((np0,nsl))*(-1),\
                                                np.ones((np0,nsl))*(-1)
                    ra_cyl_new, dec_cyl_new = np.ones((np0,nsl))*(-1),\
                                              np.ones((np0,nsl))*(-1)
                    rank_cyl_new[:,isl], snr_cyl_new[:,isl] = dat['rank'], dat['snr']
                    ra_cyl_new[:,isl], dec_cyl_new[:,isl] = dat['ra'], dat['dec']

                    ip_cyl = np.concatenate((ip_cyl,ip_cyl_new))
                    rank_cyl, snr_cyl = np.concatenate((rank_cyl,rank_cyl_new)),\
                                        np.concatenate((snr_cyl,snr_cyl_new))
                    ra_cyl, dec_cyl = np.concatenate((ra_cyl,ra_cyl_new)),\
                                      np.concatenate((dec_cyl,dec_cyl_new))
                else:
                    ra_0, dec_0 = ra_cyl[:,isl-1], dec_cyl[:,isl-1]
                    np0 = len(ra_0)

                    dat = peaks_list[isl]  # open next slice 
                    ra_1, dec_1 = dat['ra'], dat['dec']
                    rank_1, snr_1 = dat['rank'], dat['snr']
                    np1 = len(dat)
                    id_1 = np.arange(np1)

                    zmean = (zsl[isl-1] + zsl[isl])/2.
                    conv_factor = cosmo.angular_diameter_distance(zmean)
                    rad_deg = np.degrees(rad_mpc / conv_factor.value)
                    c0 = SkyCoord(ra=ra_0*u.degree, dec=dec_0*u.degree)
                    c1 = SkyCoord(ra=ra_1*u.degree, dec=dec_1*u.degree)
                    i0,i1,sep2d,dist3d = astropy.coordinates.search_around_sky(
                        c0, c1, rad_deg*u.degree, storekdtree='kdtree_sky'
                    )

                    ip_cyl = key_cylinder(
                        ip_cyl,  id_1 ,   np0, i0, i1, isl, nsl, type='int'
                    )
                    rank_cyl = key_cylinder(
                        rank_cyl,rank_1, np0, i0, i1, isl, nsl, type='float'
                    )
                    snr_cyl = key_cylinder(
                        snr_cyl, snr_1,  np0, i0, i1, isl, nsl, type='float'
                    )
                    ra_cyl = key_cylinder(
                        ra_cyl,    ra_1, np0, i0, i1, isl, nsl, type='float'
                    )
                    dec_cyl = key_cylinder(
                        dec_cyl,  dec_1, np0, i0, i1, isl, nsl, type='float'
                    )

    if npeaks_all>0:
        data_init = init_cylinders (rank_cyl, ip_cyl, wazp_specs)
        data_cylinders = append_peaks_infos_to_cylinders(
            data_init, peaks_list, zpslices_specs, 
            ip_cyl, ra_cyl, dec_cyl, rank_cyl, snr_cyl
        )    
        ncyl = len(data_cylinders)
        print('')
        print('..............Number of cylinders : '+
              str(ncyl))
        print('..............Ratio npeaks / ncyl : '+
              str(np.round(float(npeaks_all)/float(ncyl), 2)) )
    else:
        data_cylinders = None
    return data_cylinders


def peak_fwhm(smoo, imax):
    smooo = smoo[imax:len(smoo)-1] - smoo[imax]/2.
    idp = np.argwhere(smooo<=0.).T[0]
    if len(idp) > 0:
        idpp = np.amin(idp) 
        idporg = idpp +imax
    else:
        idporg = len(smoo)-1

    smooob = smoo[0:imax-1] - smoo[imax]/2.
    idm = np.argwhere(smooob<=0.).T[0]
    if len(idm) > 0:
        idmorg = np.amax(idm) 
    else:
        idmorg = 0
    delta = min(imax - idmorg , idporg - imax)
    z_precision_index = float(delta) / 4. - 1.
    
    return z_precision_index


def plot_zhist_in_cylinder(dcyl, zin, wazp_specs, zpslices_specs, tile_specs, 
                           mstar_file, ztest_faint, zbkg, cosmo, out_paths):
    
    nsig = 3.
    z_cyl = dcyl['z_init']
    id_cyl = dcyl['id']
    zsl = zpslices_specs["zsl"]
    sig_dz = (zpslices_specs["zsl_max"] - zpslices_specs["zsl_min"])/4. 
    zmin_z = zsl[dcyl['cyl_isl_min']] - nsig*sig_dz[dcyl['cyl_isl_min']]
    zmax_z = zsl[dcyl['cyl_isl_max']] + nsig*sig_dz[dcyl['cyl_isl_max']]
    isl_min, isl_max = dcyl['cyl_isl_min'], dcyl['cyl_isl_max']
    isl_mode = dcyl['cyl_isl_mode']
    conv_factor = cosmo.angular_diameter_distance(dcyl['z_init'])
    rad_deg_faint = np.degrees(wazp_specs['rad_zdet'] / conv_factor.value)
    area_cl_arcmin2 = 3600. * np.pi * rad_deg_faint**2
    tile_area_arcmin2 = 3600. * tile_specs['disc_eff_area_deg2']
    magstar = _mstar_ (mstar_file, z_cyl) 
    maglim_faint = magstar + wazp_specs['dmag_zdet']

    plt.clf()
    npts = int(10.*(zmax_z-zmin_z)/sig_dz[isl_mode])+1
    nw_faint, bins, patches = plt.hist(
        ztest_faint, npts,
        range=(zmin_z,zmax_z), 
        density=False,
        facecolor='g', alpha=0.6, 
        label = "mag < "+str(round(maglim_faint, 2))
    ) #, weights = weights)
    #nw_bright, bins, patches = plt.hist(ztest_bright, npts,range=(zmin_z,zmax_z), density=False,facecolor='r', alpha=0.6, label = "mag < "+str(round(maglim_bright, 2)))
    nw_bkg, bin_edges = np.histogram(
        zbkg, 
        bins=npts, 
        range=(zmin_z,zmax_z), 
        weights=None, density=None
    )
        
    xx = (bins[0:len(bins)-1] + bins[1:len(bins)])/2. 

    gauss_kernel = Gaussian1DKernel(4)
    smoo_faint = convolve(
        nw_faint, gauss_kernel, boundary='extend'
    )
    smoo_bkg = convolve(
        nw_bkg,  gauss_kernel, boundary='extend'
    ) * area_cl_arcmin2/tile_area_arcmin2
    smoo_faint_eff = smoo_faint - smoo_bkg
    smoo_faint_eff[smoo_faint_eff<0.] = 0.
    imax_faint = np.argmax(smoo_faint)

    plt.axvline(
        x=zmin_z, 
        color='black', linestyle='--', 
        label = "slices in cylinder"
    )
    for ii in range(isl_min, isl_max+1):
        plt.axvline(
            x=zsl[ii], 
            color='black', linestyle='--'
        )

    plt.axvline(
        x=z_cyl, 
        color='black', linestyle='-', 
        label = "initial redshift"
    )
    plt.axvline(x=xx[imax_faint], color='green', linestyle='--')
    plt.plot(xx,smoo_faint,'-',color='green', label = 'smoothed counts')
    plt.plot(xx,smoo_bkg,'-',color='blue', label = 'background')

    for ii in range(0, len(zin)):
        if ii == 0:
            plt.axvline(
                x=zin[ii], color='red', linestyle='-', 
                label = "refined redshift "
            )
        else:
            plt.axvline(
                x=zin[ii], color='red', linestyle='--', 
                label = "2ndary detection"
            )
    plt.axis([zmin_z,zmax_z,0.,max(nw_faint)*1.1])
    plt.xlabel('redshift')
    plt.ylabel('counts')
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig(
        os.path.join(
            out_paths['workdir_loc'], 
            out_paths['wazp']['plots'],
            'zhist_cyln'+str(id_cyl)+'.png'
        ),
        dpi=200
    )
    return


def plot2(dcyl, zin, wazp_specs, zpslices_specs, tile_specs, mstar_file, 
          ztest_faint, zbkg, cosmo, out_paths):

    nsig = 3.
    zsl = zpslices_specs["zsl"]
    z_cyl = dcyl['z_init']
    id_cyl = dcyl['id']
    sig_dz = (zpslices_specs["zsl_max"] - zpslices_specs["zsl_min"])/4. 
    zmin_z = zsl[dcyl['cyl_isl_min']] - nsig*sig_dz[dcyl['cyl_isl_min']]
    zmax_z = zsl[dcyl['cyl_isl_max']] + nsig*sig_dz[dcyl['cyl_isl_max']]
    isl_mode = dcyl['cyl_isl_mode']
    isl_min, isl_max = dcyl['cyl_isl_min'], dcyl['cyl_isl_max']
    conv_factor = cosmo.angular_diameter_distance(dcyl['z_init'])
    rad_deg_faint = np.degrees(wazp_specs['rad_zdet'] / conv_factor.value)
    area_cl_arcmin2 = 3600. * np.pi * rad_deg_faint**2
    tile_area_arcmin2 = 3600. * tile_specs['disc_eff_area_deg2']
    magstar = _mstar_ (mstar_file, z_cyl) 
    maglim_faint = magstar + wazp_specs['dmag_zdet']

    plt.clf()
    npts = int(10.*(zmax_z-zmin_z)/sig_dz[isl_mode])+1
    nw_faint, bins, patches = plt.hist(
        ztest_faint, 
        npts,range=(zmin_z,zmax_z), 
        density=False,facecolor='g', alpha=0.6, 
        label = "mag < "+str(round(maglim_faint, 2))
    ) #, weights = weights)
    #nw_bright, bins, patches = plt.hist(ztest_bright, npts,range=(zmin_z,zmax_z), density=False,facecolor='r', alpha=0.6, label = "mag < "+str(round(maglim_bright, 2)))
    nw_bkg, bin_edges = np.histogram(
        zbkg, bins=npts, range=(zmin_z,zmax_z), 
        weights=None, density=None
    )
        
    xx = (bins[0:len(bins)-1] + bins[1:len(bins)])/2. 

    gauss_kernel = Gaussian1DKernel(4)
    smoo_faint =  convolve(nw_faint,  gauss_kernel, boundary='extend')
    smoo_bkg =    convolve(nw_bkg,    gauss_kernel, boundary='extend')*\
                  area_cl_arcmin2/tile_area_arcmin2
    smoo_faint_eff = smoo_faint - smoo_bkg
    smoo_faint_eff[smoo_faint_eff<0.] = 0.
    contrast_faint = smoo_faint_eff / smoo_bkg

    plt.plot(xx, contrast_faint, '--', color='red')
    plt.axvline(
        x=zsl[isl_min], color='black', linestyle='--', 
        label = "slices in cylinder"
    ) # show all slices in cylinder
    for zz in zsl[isl_min:isl_max+1]:
        plt.axvline(
            x=zz, 
            color='black', linestyle='--'
        ) # show all slices in cylinder
    plt.axvline(
        x=z_cyl, 
        color='black', linestyle='-', 
        label = "initial redshift"
    )
    for ii in range(0, len(zin)):
        if ii == 0:
            plt.axvline(
                x=zin[ii], 
                color='red', linestyle='-', 
                label = "refined redshift "
            )
        else:
            plt.axvline(
                x=zin[ii], 
                color='red', linestyle='--', 
                label = "2ndary detection"
            )
    plt.axis([zmin_z,zmax_z,0.,max(contrast_faint)*1.1])
    plt.xlabel('redshift')
    plt.ylabel('contrast')
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig(
        os.path.join(
            out_paths['workdir_loc'], 
            out_paths['wazp']['plots'],
            'zhist_cylc'+str(id_cyl)+'.png'
        ),
        dpi=200
    )
    return


def z_local_maxima(dcyl, tile_specs, wazp_specs, zpslices_specs, 
                   ztest_faint, zbkg_faint, cosmo):

    gauss_kernel = Gaussian1DKernel(4)
    nsig = 3
    sig_dz = (zpslices_specs["zsl_max"] - zpslices_specs["zsl_min"])/4. 
    isl_mode = dcyl['cyl_isl_mode']
    nsl = len(zpslices_specs["zsl"])
    zsl = zpslices_specs['zsl']
    rad_zdet_faint = wazp_specs['rad_zdet']
    zmin_z = zsl[dcyl['cyl_isl_min']] - nsig*sig_dz[dcyl['cyl_isl_min']]
    zmax_z = zsl[dcyl['cyl_isl_max']] + nsig*sig_dz[dcyl['cyl_isl_max']]
    isl_min, isl_max = dcyl['cyl_isl_min'], dcyl['cyl_isl_max']

    conv_factor = cosmo.angular_diameter_distance(dcyl['z_init'])
    rad_deg_faint = np.degrees(rad_zdet_faint / conv_factor.value)
    area_cl_arcmin2 = 3600. * np.pi * rad_deg_faint**2
    tile_area_arcmin2 = 3600. * tile_specs['disc_eff_area_deg2']

    npts = int(10.*(zmax_z-zmin_z)/sig_dz[isl_mode])+1
    nw_faint, bins =  np.histogram(
        ztest_faint , npts,range=(zmin_z,zmax_z), 
        weights=None, density=None
    )
    nw_bkg, bins =  np.histogram(
        zbkg_faint  , npts,range=(zmin_z,zmax_z), 
        weights=None, density=None
    )
    xx = (bins[0:len(bins)-1] + bins[1:len(bins)])/2. 
    smoo_faint =  convolve(nw_faint,  gauss_kernel, boundary='extend')
    smoo_bkg =    convolve(nw_bkg,    gauss_kernel, boundary='extend')*\
                  area_cl_arcmin2/tile_area_arcmin2
    smoo_faint_eff = smoo_faint - smoo_bkg
    smoo_faint_eff[smoo_faint_eff<0.] = 0.

    contrast_faint = smoo_faint_eff / smoo_bkg
    imax_faint = np.argmax(smoo_faint)

    # build support in z 
    if isl_mode > 0 & isl_mode < nsl-1:
        zmin_cen = zsl[isl_mode] -1.5 * sig_dz[isl_mode]
        zmax_cen = zsl[isl_mode] +1.5 * sig_dz[isl_mode]
    if isl_mode == 0 :
        zmin_cen = zsl[isl_mode] -3.0 * sig_dz[isl_mode]
        zmax_cen = zsl[isl_mode] +1.5 * sig_dz[isl_mode]
    if isl_mode == nsl-1:
        zmin_cen = zsl[isl_mode] -1.5 * sig_dz[isl_mode]
        zmax_cen = zsl[isl_mode] +3.0 * sig_dz[isl_mode]

    #init peaks 
    z_precision = -1.*np.ones(1)
    z_contrast  = -1.*np.ones(1)
    ibest = isl_mode*np.ones(1).astype(int) 
    zin = np.array([zsl[isl_mode]])
    ncl_in_cyl  = 1
    
    # find all local maxima
    if np.amax(smoo_faint_eff) > 0.:
        arr = ((np.r_[True, smoo_faint_eff[1:] > smoo_faint_eff[:-1]]) & 
               (np.r_[smoo_faint_eff[:-1] > smoo_faint_eff[1:], True]))
        imax_all = np.argwhere(arr).T[0]
        isort = np.argsort(-smoo_faint_eff[imax_all])
        imax_sort = imax_all[isort]
        sorted_ids = imax_all[isort]
        zall = xx[imax_all[isort]] # sorted z's 
        zcond = ((zall<=zmax_cen) & (zall>=zmin_cen))
        imax_insort = sorted_ids[zcond]
        if len(imax_insort)>0:
            zin = xx[imax_insort]
            z_precision = np.zeros(len(zin))
            z_contrast  = np.zeros(len(zin))
            ibest       = np.zeros(len(zin)).astype(int)
            ncl_in_cyl  = len(zin)
            for ii in range(0, len(zin)):
                z_precision[ii] = peak_fwhm(smoo_faint_eff, imax_insort[ii])
                z_contrast[ii] = contrast_faint[imax_insort[ii]]
                ibest[ii] = isl_min + np.argmin( 
                    np.absolute( zsl[isl_min: isl_max+1] - zin[ii] ) 
                )

                #nz_peak = len(ztest_faint[(ztest_faint<=zin[0]+sig_dz[ibest]) & 
                #(ztest_faint>=zin[0]-sig_dz[ibest])])
            
    return ncl_in_cyl, zin, z_precision, z_contrast, ibest



def cylinders2clusters (zpslices_specs, wazp_specs, tile_specs, clcat, data_cyl, 
                        mstar_file, cosmo_params, 
                        data_gal, galcat, out_paths, hpx_meta, verbose):

    clkeys = clcat['wazp']['keys']
    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])

    ip_fcyl = data_cyl['ip_cyl']
    ra_fcyl, dec_fcyl = data_cyl['ra_cyl'], data_cyl['dec_cyl']
    snr_fcyl = data_cyl['snr_cyl']

    # get_cylinders
    ra_cyl, dec_cyl, z_cyl = data_cyl['ra'], data_cyl['dec'], data_cyl['z_init']
    snr_cyl = data_cyl['snr']
    isl_mode = data_cyl['cyl_isl_mode']
    isl_min, isl_max = data_cyl['cyl_isl_min'], data_cyl['cyl_isl_max']
    cyl_nsl = isl_max - isl_min + 1
    ncyl = len(data_cyl)

    # read zp specs
    sig_dz = (zpslices_specs["zsl_max"] - zpslices_specs["zsl_min"])/4. 
    zsl = zpslices_specs["zsl"]

    # init of output fields 
    ra_init, dec_init = np.zeros(2*ncyl), np.zeros(2*ncyl)
    z_init, snr_init =  np.zeros(2*ncyl), np.zeros(2*ncyl)
    ra_cl, dec_cl = np.zeros(2*ncyl), np.zeros(2*ncyl)
    z_cl, snr_cl =  np.zeros(2*ncyl), np.zeros(2*ncyl)
    icyl = np.zeros(2*ncyl).astype(int) # maximum length assuming 2 clusters / cylinder 
    isl_cl = np.zeros(2*ncyl).astype(int)
    isl_cl_init = np.zeros(2*ncyl).astype(int)
    z_cl = np.zeros(2*ncyl)
    ncl_in_cyl = np.zeros(2*ncyl).astype(int)
    z_precision = np.zeros(2*ncyl)
    z_contrast = np.zeros(2*ncyl)
    cyl_nsl_out = np.zeros(2*ncyl).astype(int)
    cyl_isl_min_out = np.zeros(2*ncyl).astype(int)
    cyl_isl_max_out = np.zeros(2*ncyl).astype(int)

    ncl = 0
    for i in range (0,ncyl):
        #print ('icyl ', i)
        ztest_faint, zbkg_faint = select_zps_cylinder(
            data_cyl[i], data_gal, galcat, zpslices_specs, 
            mstar_file, cosmo, wazp_specs
        )
        ncl_in_cyl0, zin, z_precision0, z_contrast0, ibest = z_local_maxima(
            data_cyl[i], tile_specs, wazp_specs, 
            zpslices_specs, ztest_faint, zbkg_faint, cosmo
        )
        #if (verbose >=2 and data_cyl[i]['snr']>10.) or 
        # (verbose >=2 and z_precision0[0]==-1.):
        if (verbose >=2 and z_precision0[0]==-1.):
            plot_zhist_in_cylinder(
                data_cyl[i], zin, wazp_specs, zpslices_specs, tile_specs, 
                mstar_file, ztest_faint, zbkg_faint, cosmo, out_paths
            )
            #plot2(data_cyl[i], zin, wazp_specs, zpslices_specs, tile_specs, 
            # mstar_file, ztest_faint, zbkg_faint, cosmo, out_paths)

        for ii in range(0, ncl_in_cyl0):
            jsl = ibest[ii]
            ra_init[ncl], dec_init[ncl] = ra_cyl[i], dec_cyl[i]
            z_init[ncl], snr_init[ncl] =   z_cyl[i], snr_cyl[i]
            ra_cl[ncl], dec_cl[ncl] = ra_fcyl[i][jsl], dec_fcyl[i][jsl]
            z_cl[ncl],  snr_cl[ncl] = zin[ii],      snr_fcyl[i][jsl]         
            icyl[ncl] = i
            cyl_nsl_out[ncl] = cyl_nsl[i]
            cyl_isl_min_out[ncl], cyl_isl_max_out[ncl] = isl_min[i], isl_max[i]
            isl_cl_init[ncl] = isl_mode[i]
            isl_cl[ncl] = jsl
            ncl_in_cyl[ncl] = ncl_in_cyl0
            z_precision[ncl] = z_precision0[ii]
            z_contrast[ncl] = z_contrast0[ii]
            ncl+=1

    print ('              Nr of clusters : '+str(ncl))

    data_clusters = np.zeros(
        ncl, dtype={
            'names':(
                'tile', 'id_in_tile', 
                clkeys['key_ra'], clkeys['key_dec'], 
                clkeys['key_zp'], clkeys['key_snr'],
                'ra_init', 'dec_init', 'z_init', 'snr_init', 
                'id_cyl', 'cyl_nsl', 'cyl_isl_min', 'cyl_isl_max', 
                'isl_cl_init', 'isl_cl', 'id_peak','zpeak_fwhm', 
                'zpeak_contrast', 'ncl_in_cyl'
            ),
            'formats':(
                'i8', 'i8', 
                'f8', 'f8', 
                'f4', 'f4',
                'f8', 'f8', 'f4', 'f4', 
                'i4', 'i4', 
                'i4', 'i4', 'i4', 'i4', 'i4', 'f4', 'f4', 'i4'
            )
        }
    )
    isl = isl_cl[0:ncl]
    data_clusters['tile'] =  tile_specs['id']*np.ones(ncl).astype(int)
    data_clusters['id_in_tile'] =  np.arange(ncl)
    data_clusters[clkeys['key_ra']] =  ra_fcyl[icyl[0:ncl], isl]
    data_clusters[clkeys['key_dec']] = dec_fcyl[icyl[0:ncl], isl]
    data_clusters[clkeys['key_zp']] =   z_cl[0:ncl]
    data_clusters[clkeys['key_snr']] = snr_fcyl[icyl[0:ncl], isl]
    data_clusters['id_peak'] = ip_fcyl[icyl[0:ncl], isl]

    data_clusters['ra_init'] = ra_init[0:ncl]
    data_clusters['dec_init'] = dec_init[0:ncl]
    data_clusters['z_init'] = z_init[0:ncl]
    data_clusters['snr_init'] = snr_init[0:ncl]

    data_clusters['id_cyl'] = icyl[0:ncl]
    data_clusters['isl_cl_init'] = isl_cl_init[0:ncl]
    data_clusters['isl_cl'] = isl_cl[0:ncl]
    data_clusters['cyl_nsl'] = cyl_nsl_out[0:ncl]
    data_clusters['cyl_isl_min'] = cyl_isl_min_out[0:ncl]
    data_clusters['cyl_isl_max'] = cyl_isl_max_out[0:ncl]
    data_clusters['zpeak_fwhm'] = z_precision[0:ncl] 
    data_clusters['zpeak_contrast'] = z_contrast[0:ncl] 

    return data_clusters


def cl_duplicates_filtering(data_clusters_in, wazp_specs, clcat, 
                            zpslices_specs, cosmo_params, mode):
    # mode can be tile or survey
    # if mode = survey => search for duplicates coming from diff tiles

    clkeys = clcat['wazp']['keys']
    dmpc = wazp_specs['duplic_dist_mpc']
    nsigdz = wazp_specs['duplic_nsigdz']
    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])

    idecr = np.argsort(-data_clusters_in[clkeys['key_snr']])
    data_cl = data_clusters_in[idecr]
    snr = data_cl[clkeys['key_snr']]
    zcl = data_cl[clkeys['key_zp']]
    ra, dec =  data_cl[clkeys['key_ra']], data_cl[clkeys['key_dec']]
    clid =  data_cl['id_in_tile']
    tile_id =  data_cl['tile']
    flagdp = np.zeros(len(zcl)).astype(int)
    zsl_min, zsl_max = zpslices_specs['zsl_min'], zpslices_specs['zsl_max']
    zsl     = zpslices_specs['zsl']
    sig = (zsl_max - zsl_min)/4.

    fsig = interpolate.interp1d(
        zsl, sig, kind = 'linear', bounds_error=False, fill_value='extrapolate'
    )
    sig_dz = fsig(zcl)

    Nside_tmp, nest_tmp = wazp_specs['Nside_tmp'], wazp_specs['nest_tmp']  
    clhpx = hp.ang2pix(Nside_tmp, ra, dec, nest_tmp, lonlat=True)

    for i in range(0, len(data_cl)):
        if flagdp[i] == 0:
            conv_factor = cosmo.angular_diameter_distance(zcl[i])
            radius_deg = np.degrees(dmpc / conv_factor.value)
            if mode == 'tile':
                cond = ( (np.absolute(zcl[i]-zcl)<nsigdz*sig_dz[i]) &\
                         (clid[i]!=clid))
            if mode == 'survey':
                cond = ( (np.absolute(zcl[i]-zcl)<nsigdz*sig_dz[i]) &\
                         (tile_id[i]!=tile_id)) 
            in_cone = cond_in_disc(
                ra[cond], dec[cond], clhpx[cond], 
                Nside_tmp, nest_tmp,
                ra[i], dec[i], radius_deg
            )
            cond[np.argwhere(cond).T[0]] = in_cone 
            flagdp[cond] = 1
    data_clusters_out = data_cl[flagdp==0]
    print ('              Nr. of duplicates = '+
           str(len(data_clusters_in) - len(data_clusters_out))+' / '+
           str(len(data_clusters_in)))
    print ('              Final Nr of clusters : '+
           str(len(data_clusters_out)))

    return data_clusters_out


def append_infos_to_clusters(target, data_clusters_init, cosmo_params):
    # add distance to center for each cluster 

    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])

    ra_target, dec_target = target['ra'], target['dec']
    zcl = data_clusters_init['z']

    dist_arcmin = 60. * np.degrees(dist_ang(
        data_clusters_init['ra'], data_clusters_init['dec'], ra_target, dec_target
    ))
    conv_factor = cosmo.angular_diameter_distance(zcl)
    dist_mpc = np.radians(dist_arcmin/60.) * conv_factor.value 
           
    data_extra = np.zeros( 
        (len(zcl)), 
        dtype={
            'names':(
                'dist_arcmin', 'dist_mpc'
            ),
            'formats':(
                'f8','f8'
            )
        }
    )

    data_extra['dist_arcmin'] = dist_arcmin
    data_extra['dist_mpc'] = dist_mpc
    arrays = [data_clusters_init, data_extra]
    data_clusters = join_struct_arrays(arrays)

    return data_clusters


def wave_radius(wmap_data, ip, jp, wazp_cfg):

    dwmap = ndi.distance_transform_edt((wmap_data>wazp_cfg['wmap_thresh'])*wmap_data)
    radius_mpc = dwmap[ip, jp]/float(wazp_cfg['resolution'])
    return radius_mpc


def wazp_tile_slice(tile, dat_galcat, dat_footprint, galcat, footprint,
                    zpslices, gbkg, mstar_file, wazp_cfg, cosmo_params, 
                    paths, verbose):
    
    isl = zpslices['id']                                 
    cosmo = flat(H0=cosmo_params['H'], Om0=cosmo_params['omega_M_0'])
    conv_factor = cosmo.angular_diameter_distance( zpslices['zsl']) 
    xycat_fitsname = os.path.join(
        paths['workdir_loc'], paths['wazp']['files'], 
        'xycat_'+str(isl)+'.fits'
    )
    xycat_fi_fitsname = os.path.join(
        paths['workdir_loc'], paths['wazp']['files'], 
        'xycat_fi_'+str(isl)+'.fits'
    )
    wmap_fitsname =  os.path.join(
        paths['workdir_loc'], paths['wazp']['files'], 
        'wmap_'+str(isl)+'.fits'
    )
    peaks_fitsname = os.path.join(
        paths['workdir_loc'], paths['wazp']['files'], 
        "peaks_"+str(isl)+".fits"
    )

    # select objects for computing density maps 
    ra_map, dec_map, weight_map = select_galaxies_in_slice(
        dat_galcat, galcat, wazp_cfg, zpslices, mstar_file, 
        wazp_cfg['dmag_det'], 'detlum'
    )
    xycat = compute_catimage(
        ra_map, dec_map, weight_map, 
        zpslices, wazp_cfg, tile, cosmo_params
    ) 
    # compute bkg without weights for filling image holes => mr_filter
    if wazp_cfg['map_filling']:
        bkg_arcmin2, bkg_mpc2 = bkg_tile_slice(
            dat_galcat, dat_footprint, galcat, footprint, 
            zpslices, mstar_file, wazp_cfg, cosmo_params, 
            wazp_cfg['dmag_det'], 'none'
        )
        xycat_fi = compute_filled_catimage(
            ra_map, dec_map, weight_map, 
            zpslices, wazp_cfg, tile, cosmo_params, 
            dat_footprint, footprint, bkg_arcmin2
        )
    else:
        xycat_fi = np.copy(xycat)

    # build density map /  extract peaks /compute attributes and filter 
    if not os.path.isfile(wmap_fitsname):
        map2fits(
            xycat_fi, wazp_cfg, 
            tile, zpslices['zsl'], 
            cosmo_params, xycat_fi_fitsname
        )
        run_mr_filter(xycat_fi_fitsname, wmap_fitsname, wazp_cfg)
    wmap_data = fits2map(wmap_fitsname)
    rap0, decp0, ip0, jp0 = wmap2peaks(
        wmap_fitsname, wazp_cfg, tile, zpslices['zsl'], cosmo_params
    )
    rap, decp, ip, jp = filter_peaks(
        tile, zpslices['zsl'], cosmo_params, wazp_cfg['resolution'], 
        rap0, decp0, ip0, jp0
    ) # to keep inner tile peaks
    wradius_mpc = wave_radius(
        wmap_data, ip.astype(int), jp.astype(int), wazp_cfg
    )
    coverfrac = coverfrac_disc(
        rap, decp, 
        dat_footprint, footprint, 
        np.degrees( wazp_cfg['radius_snr_mpc'] / conv_factor.value)
    )
    pcond = ((coverfrac>0.) & (wradius_mpc>wazp_cfg['radius_snr_mpc']))
    data_peaks = init_peaks_table(
        rap[pcond], decp[pcond], ip[pcond], jp[pcond], 
        coverfrac[pcond], wradius_mpc[pcond], zpslices['zsl']
    )

    # compute fluxes in given aper 
    add_peaks_attributes(
        data_peaks, dat_galcat, galcat, 
        dat_footprint, footprint, 
        wazp_cfg, zpslices, gbkg, 
        mstar_file, 
        np.degrees( wazp_cfg['radius_snr_mpc'] / conv_factor.value), 
        cosmo_params
    )

    if verbose >=1:
        print ('..............peaks filtering inner tile in / out ', 
               len(rap0) , len(rap))
        print ('..............peaks filtering wradius    in / out ', 
               len(rap) , len(rap[pcond]))
        print ('..............peaks filtering SNR        in / out ', 
               len(data_peaks), 
               len(data_peaks[data_peaks['snr']>=\
                              np.float64(wazp_cfg['snr_min'])]))

        eff_area_mpc2 = tile['eff_area_deg2'] * \
                        (np.pi * conv_factor.value / 180.)**2
        framed_eff_area_mpc2 = tile['disc_eff_area_deg2'] *\
                               (np.pi * conv_factor.value / 180.)**2
        print ('         area mpc2 / deg2 ', 
               np.round(eff_area_mpc2, 2), np.round(tile['eff_area_deg2'], 2))
        print ('         galaxies density/arcmin2 /mpc2', 
               np.round(
                   float(len(ra_map))/(tile['disc_eff_area_deg2']*3600.), 3
               ),
               np.round(float(len(ra_map))/framed_eff_area_mpc2, 3))
        print ('         clusters density/mpc2 ', 
               np.round(
                   float(len(data_peaks[
                       data_peaks['snr']>=wazp_cfg['snr_min']
                   ]))/eff_area_mpc2, 3))

    if verbose >=2:
        map2fits(
            xycat, 
            wazp_cfg, tile, zpslices['zsl'], cosmo_params, xycat_fitsname
        )
        t = Table (data_peaks[data_peaks['snr']>=wazp_cfg['snr_min']])
        t.write(peaks_fitsname,overwrite=True)

    return data_peaks[data_peaks['snr']>=wazp_cfg['snr_min']]


def add_hpx_to_cat(data_gal, ra, dec, Nside_tmp, nest_tmp, keyname):
    ghpx = hp.ang2pix(Nside_tmp, ra, dec, nest_tmp, lonlat=True)
    t = Table (data_gal)
    t[keyname] = ghpx
    return t

def wazp_tile(tile_specs, data_gal_tile, data_fp_tile, galcat, footprint, 
            zpslices, gbkg, zp_metrics, mstar_file, 
            wazp_cfg, clcat, cosmo_params, out_paths, verbose ): 

    print ('..........Start wazp tile catalog construction')
    # add hpx to galcat to speed up condition_in_disc around all detections
    data_gal_tile = add_hpx_to_cat(
        data_gal_tile, data_gal_tile[galcat['keys']['key_ra']], 
        data_gal_tile[galcat['keys']['key_dec']],
        wazp_cfg['Nside_tmp'], wazp_cfg['nest_tmp'], 'hpx_tmp'
    )

    # compute mean bkg tile 
    #print ('..........Compute mean bkg in tile')
    #data_bkg = bkg_tile (data_gal_tile, data_fp_tile, galcat, footprint, 
    #                     zpslices, mstar_file, wazp_cfg, cosmo_params, wazp_cfg['dmag_det'], 'lum')

    Nclusters = 0
    if not os.path.isfile(
            os.path.join(
                out_paths['workdir_loc'], out_paths['wazp']['results'], 
                'clusters0.npy'
            )
    ):
        peaks_list = []
        npeaks_tot = 0
        for isl in range (0, len(zpslices)):
            if not os.path.isfile(
                    os.path.join(
                        out_paths['workdir_loc'], out_paths['wazp']['files'], 
                        'peaks_'+str(isl)+'.npy'
                    )
            ):
                print ('.............. Detection in slice ', isl)
                data_peaks = wazp_tile_slice(
                    tile_specs, data_gal_tile, data_fp_tile, galcat, footprint,
                    zpslices[isl], gbkg[isl], mstar_file, wazp_cfg, cosmo_params, 
                    out_paths, verbose)
                np.save(
                    os.path.join(
                        out_paths['workdir_loc'], out_paths['wazp']['files'], 
                        'peaks_'+str(isl)+'.npy'
                    ), data_peaks
                )
                npeaks_tot += len(data_peaks)
            else:
                print ('.............. Use existing detections in slice ', isl)
                data_peaks = np.load(
                    os.path.join(
                        out_paths['workdir_loc'], out_paths['wazp']['files'], 
                        'peaks_'+str(isl)+'.npy'
                    )
                )
                npeaks_tot += len(data_peaks)
            peaks_list.append(data_peaks)        

        if npeaks_tot>0:
            print ('..........Start cylinders')
            data_cylinders = make_cylinders(
                peaks_list, zpslices, wazp_cfg, cosmo_params
            )
            if verbose >=1:
                t = Table (data_cylinders)
                t.write(os.path.join(
                    out_paths['workdir_loc'], out_paths['wazp']['results'], 
                    "cylinders.fits"),overwrite=True
                )

            if data_cylinders is not None:    
                print ('..........Start cylinders_2_clusters')
                data_clusters0 = cylinders2clusters(
                    zpslices, wazp_cfg, tile_specs, clcat, data_cylinders, 
                    mstar_file, cosmo_params, 
                    data_gal_tile, galcat, out_paths, footprint, verbose
                )
                np.save(
                    os.path.join(
                        out_paths['workdir_loc'], out_paths['wazp']['results'], 
                        'clusters0.npy'
                    ), data_clusters0
                )
                if verbose >=1:
                    t = Table (data_clusters0)
                    t.write(os.path.join(
                        out_paths['workdir_loc'], out_paths['wazp']['results'], 
                        "clusters0.fits"
                    ), overwrite=True)
        
        else:
            data_clusters0 = None
            print ('..........No clusters in this tile')
    else:
        print ('..........Use existing clusters')
        data_clusters0 = np.load(
            os.path.join(
                out_paths['workdir_loc'], out_paths['wazp']['results'], 
                'clusters0.npy'
            )
        )
        
    print ('..........Start filtering ')
    if data_clusters0 is not None:
        data_clusters = cl_duplicates_filtering(
            data_clusters0, wazp_cfg, clcat, zpslices, cosmo_params, 'tile'
        )
        Nclusters = len(data_clusters)
    else:
        data_clusters = None
        Nclusters = 0

    # write final tile recap for final concatenation of clusters
    tile_info = np.zeros( 1, 
                          dtype={'names':('id', 'eff_area_deg2', 'Nclusters'),
                                 'formats':('i8', 'f8', 'i8')}) 
    tile_info['id'] = tile_specs['id']
    tile_info['eff_area_deg2'] = tile_specs['eff_area_deg2']
    tile_info['Nclusters'] = Nclusters
    
    return data_clusters, tile_info 


def run_wazp_tile(config, dconfig, thread_id):
    # read config file
    with open(config) as fstream:
        param_cfg = yaml.safe_load(fstream)
    with open(dconfig) as fstream:
        param_data = yaml.safe_load(fstream)

    survey, ref_filter  = param_cfg['survey'], param_cfg['ref_filter']
    maglim = param_cfg['maglim_det']
    galcat = param_data['galcat'][survey]
    clcat = param_cfg['clcat']
    out_paths = param_cfg['out_paths']
    admin = param_cfg['admin']
    footprint = param_data['footprint'][survey]
    zp_metrics = param_data['zp_metrics'][survey][ref_filter]
    magstar_file = param_data['magstar_file'][survey][ref_filter]
    wazp_cfg = param_cfg['wazp_cfg']

    workdir = out_paths['workdir']
    all_tiles = read_FitsCat(
        os.path.join(workdir, admin['tiling']['tiles_filename'])
    )
    tiles = all_tiles[(all_tiles['thread_id']==int(thread_id))]    
    print ('THREAD ', int(thread_id))

    zpslices = read_FitsCat(
        os.path.join(workdir, param_cfg['wazp_cfg']['zpslices_filename'])
    )
    gbkg = read_FitsCat(
        os.path.join(workdir, 'gbkg', param_cfg['wazp_cfg']['gbkg_filename'])
    )

    for it in range(0, len(tiles)):
        tile_dir = os.path.join(
            workdir, 'tiles', 
            'tile_'+str(int(tiles['id'][it])).zfill(3)
        )
        print ('..... Tile ', int(tiles['id'][it]))

        create_directory(tile_dir)
        create_tile_directories(tile_dir, out_paths['wazp'])
        out_paths['workdir_loc'] = tile_dir # local update 
        tile_radius_deg = tiles['radius_tile_deg'][it]
        data_gal_tile = read_mosaicFitsCat_in_disc(
            galcat, tiles[it], tile_radius_deg
        )   
        data_gal_tile = data_gal_tile\
                        [data_gal_tile[galcat['keys']['key_mag']]<=\
                         np.float64(maglim)]
        data_fp_tile = read_mosaicFootprint_in_disc(
            footprint, tiles[it], tile_radius_deg
        )
        tile_specs = create_tile_specs(
            tiles[it], admin, 
            None, None, 
            data_fp_tile, footprint
        )

        if param_cfg['verbose'] >=2:
            t = Table (data_gal_tile)
            t.write(os.path.join(tile_dir, "galcat.fits"),overwrite=True)
            t = Table (data_fp_tile)
            t.write(os.path.join(tile_dir, "footprint.fits"),overwrite=True)
        
        if not os.path.isfile(
                os.path.join(
                    tile_dir, out_paths['wazp']['results'], 
                    "clusters.fits"
                )
        ):
            data_clusters, tile_info = wazp_tile(
                tile_specs, data_gal_tile, data_fp_tile, galcat, footprint, 
                zpslices, gbkg, zp_metrics, magstar_file, 
                wazp_cfg, clcat, param_cfg['cosmo_params'], 
                out_paths, param_cfg['verbose'] ) 

            if data_clusters is not None:
                t = Table (data_clusters)#, names=names)
                t.write(
                    os.path.join(
                        tile_dir, out_paths['wazp']['results'], 
                        "clusters.fits"
                    ),overwrite=True
                )
            tile_info = Table(tile_info)
            tile_info.write(os.path.join(
                out_paths['workdir_loc'], out_paths['wazp']['results'], 
                "tile_info.fits"
            ), overwrite=True)
    return


def tiles_with_clusters(out_paths, all_tiles):
    flag = np.zeros(len(all_tiles))
    for it in range(0, len(all_tiles)):
        tile_dir = tile_dir_name(
            out_paths['workdir'], int(all_tiles['id'][it])
        )
        if os.path.isfile(
                os.path.join(
                    tile_dir, out_paths['wazp']['results'], 
                    "tile_info.fits"
                )
        ):
            if read_FitsCat(
                    os.path.join(
                        tile_dir, out_paths['wazp']['results'], 
                        "tile_info.fits"
                    )
            )[0]['Nclusters'] > 0:
                flag[it] = 1
            else:
                print ('warning : no detection in tile ', tile_dir)
        else:
            print ('warning : missing tile ', tile_dir)
    return all_tiles[flag==1]


def wazp_concatenate(all_tiles, zpslices_filename, wazp_cfg, clcat, 
                     cosmo_params, out_paths):

    zpslices = read_FitsCat(zpslices_filename)
    # concatenate all tiles 
    print ('Concatenate clusters')
    list_clusters = []
    for it in range(0, len(all_tiles)):
        tile_dir = tile_dir_name(
            out_paths['workdir'], int(all_tiles['id'][it]) 
        )
        list_clusters.append(
            os.path.join(tile_dir, out_paths['wazp']['results'])
        )
    data_clusters0 = concatenate_clusters(
        list_clusters, 'clusters.fits', 
        os.path.join(out_paths['workdir'], 'tmp', 'clusters0.fits')
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
        data_clusters0[condzmin & condzmax], wazp_cfg, clcat, zpslices, cosmo_params, 'survey'
    )

    # create unique index with decreasing SNR 
    data_clusters = add_clusters_unique_id(
        data_clusters0f, clcat['wazp']['keys']
    )
    return data_clusters 


def official_wazp_cat(data_cl, clkeys, richness_specs, rich_min, wazp_file): 


    data_cl = data_cl[data_cl['n200_pmem']>=rich_min]

    npts = str(richness_specs['npts'])

    data = np.zeros( 
        (len(data_cl)), 
        dtype={
            'names':(
                clkeys['key_id'], 
                clkeys['key_ra'], clkeys['key_dec'], 
                clkeys['key_zp'], clkeys['key_snr'],
                'r200_mpc', 'n200', 'n200_err', 
                'n500kpc', 'n500kpc_err', 
                'raw_coverfrac', 'weighted_coverfrac',
                'bkg_coverfrac', 
                'md_bkg_arcmin2', 'md_bkg_mpc2', 
                'slope_dprofile', 'radius_vec_arcmin',
                'radius_vec_mpc', 
                'richness', 'richness_err', 
                'nmem', 'mstar', 'flag_pmem'
            ),
            'formats':(
                'a30', 
                'f8', 'f8', 
                'f4', 'f4', 
                'f4', 'f4', 'f4', 
                'f4', 'f4', 
                'f4', 'f4', 
                'f4', 
                'f4', 'f4', 
                'f4', npts+'f8', 
                npts+'f8', 
                npts+'f8', npts+'f8', 
                'i8', 'f4', 'i8'
            )
        }
    ) 

    data[clkeys['key_id']] =   data_cl[clkeys['key_id']] 
    data[clkeys['key_ra']] =   data_cl[clkeys['key_ra']] 
    data[clkeys['key_dec']] =  data_cl[clkeys['key_dec']] 
    data[clkeys['key_snr']] =  data_cl[clkeys['key_snr']] 
    data[clkeys['key_zp']] =   data_cl[clkeys['key_zp']] 

    data['r200_mpc'] =   data_cl['r200_mpc'] 
    data['n200'] =   data_cl['n200_pmem'] 
    data['n200_err'] =   data_cl['n200_pmem_err'] 
    data['n500kpc'] =   data_cl['n500kpc_pmem'] 
    data['n500kpc_err'] =   data_cl['n500kpc_pmem_err'] 

    data['raw_coverfrac'] =   data_cl['raw_coverfrac'] 
    data['weighted_coverfrac'] =   data_cl['weighted_coverfrac'] 
    data['bkg_coverfrac'] =   data_cl['bkg_coverfrac'] 
    data['md_bkg_arcmin2'] =   data_cl['md_bkg_arcmin2'] 
    data['md_bkg_mpc2'] =   data_cl['md_bkg_mpc2'] 
    data['slope_dprofile'] =   data_cl['slope_dprofile'] 

    data['radius_vec_arcmin'] =   data_cl['radius_vec_arcmin'] 
    data['radius_vec_mpc'] =   data_cl['radius_vec_mpc'] 
    data['richness'] =   data_cl['richness'] 
    data['richness_err'] =   data_cl['richness_err'] 

    data['nmem'] =   data_cl['nmem'] 
    data['mstar'] =   data_cl['mstar'] 
    data['flag_pmem'] =   data_cl['flag_pmem'] 

    Table(data).write(wazp_file, overwrite=True)
    return


def wmap_at_zcl(target, data_primary_clusters, r200_mpc, cluster_keys, 
                wazp_specs, tile_specs, zpslices_specs, cosmo_params, 
                dat_footprint, hpx_meta, dat_galcat, galcat_keys, 
                mstar_filename, workdir, path_outputs, path):
    
    zcl = data_primary_clusters[cluster_keys['key_zp']]
    wmap_list = []
    ncl = len(zcl)
    if target['z']>0.:
        ncl = 1

    for i in range (0, len(zcl)):

        xycat_fi_fitsname = os.path.join(
            workdir, path_outputs, 
            'xycat_fi_cl'+str(i)+'.fits'
        )
        wmap_fitsname = os.path.join(
            workdir, path_outputs, 
            'wmap_cl'+str(i)+'.fits'
        )
        zslice = {
            'zsl': zcl[i], 
            'zsl_min': zcl[i] - 2.*fsig(zcl[i]),
            'zsl_max': zcl[i] + 2.*fsig(zcl[i])
                 }
        ra_map, dec_map, weight_map = select_galaxies_in_slice(
            dat_galcat, galcat_keys, wazp_specs, zslice, mstar_filename, 
            wazp_specs['dmag_det'], 'detlum'
        )
        xycat_fi = compute_catimage(
            ra_map, dec_map, weight_map, 
            zslice, wazp_specs, tile_specs, cosmo_params
        ) 
        map2fits(xycat_fi, wazp_specs, tile_specs, zcl[i], cosmo_params, 
                 xycat_fi_fitsname)
        run_mr_filter(xycat_fi_fitsname, wmap_fitsname, wazp_specs) 
        wmap_list.append(wmap_fitsname)

    return wmap_list


def update_config(param_cfg, param_data):

    input_data_structure = param_data['input_data_structure']
    workdir = param_cfg['out_paths']['workdir']
    cosmo_params = param_cfg['cosmo_params']
    survey = param_cfg['survey']
    ref_filter = param_cfg['ref_filter']

    # update mag with selected filter in config 
    param_data['galcat'][survey]['keys']['key_mag'] = \
          param_data['galcat'][survey]['keys']['key_mag'][ref_filter]
    # update final cluster cat output
    param_cfg['clcat']['wazp']['cat'] = os.path.join(
        workdir, 'tmp', 'clusters.fits'
    )
    # add wazp case to clcat in data.cfg for Pmem execution
    param_data['clcat'] = {'wazp': param_cfg['clcat']['wazp']}

    zmax = param_data['zp_metrics'][survey][ref_filter]['zpmax']
    mstar_file = param_data['magstar_file'][survey][ref_filter]
    magstar = _mstar_(mstar_file, zmax) 
    dmag_max = param_cfg['wazp_cfg']['dmag_det']
    maglim_det = _mstar_(mstar_file, zmax) + dmag_max
    print ('.... adopted maglim_det  = ', np.round(maglim_det, 2))

    dmag_max = param_cfg['pmem_cfg']['pmem_specs']['dmagmax']
    maglim_pmem = _mstar_(mstar_file, zmax) + dmag_max
    print ('.... adopted maglim_pmem = ', np.round(maglim_pmem, 2))

    param_cfg['maglim_det'] = np.float64(maglim_det)
    param_cfg['maglim_pmem'] = np.float64(maglim_pmem)

    # zmax is set to allow pmem to work on all wazp detections 
    param_cfg['pmem_cfg']['global_conditions']['zcl_max'] = np.float64(zmax+0.2)
    param_cfg['pmem_cfg']['mag_bin_specs']['max'] = np.float64(
        int(maglim_pmem)+1
    )

    param_cfg['admin']['tiling']['nest'] = input_data_structure[survey]['nest']
    param_data['galcat'][survey]['mosaic']['Nside'] = input_data_structure[survey]['Nside']
    param_data['galcat'][survey]['mosaic']['nest'] = input_data_structure[survey]['nest']
    param_data['footprint'][survey]['mosaic']['Nside'] = input_data_structure[survey]['Nside']
    param_data['footprint'][survey]['mosaic']['nest'] = input_data_structure[survey]['nest']

    return param_cfg, param_data


