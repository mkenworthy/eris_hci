from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage

def phase_to_PSF(aper,phase,pupil_grid,focal_grid, wlen, bwidth, nwlens=7):
    '''phase_to_PSF - makes the three focal plane PSFS for the ERIS gvAPP coronagraph
    wlen - central wavlength
    bwidth - FWHM of the filter used in the same units as wlen
    nwlens - the number of (equally spaced) wavelength images generated over the bandwidth of the filter'''
    wf_p1 = Wavefront(aper*np.exp(1j*phase)) # Coro PSF 1
    wf_p2 = Wavefront(aper*np.exp(-1j*phase))# Coro PSF 2
    wf_lt = Wavefront(aper) # Leakage Term PSF

    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    focalplane = 0
    fp_p1 = 0
    fp_p2 = 0
    fp_lt = 0
    for wl in np.linspace((wlen-(bwidth/2.))/wlen, \
                          (wlen+(bwidth/2.))/wlen, \
                          nwlens): # normalised to unity wavelength
        wf_p1.wavelength = wl
        wf_p2.wavelength = wl
        wf_lt.wavelength = wl
        print(f'Current wavlength is {wl:7.5f} microns ({nwlens} in total)')
        print('Making 1 of 3 PSFs...')
        fp_p1 += prop(wf_p1).intensity

        print('Making 2 of 3 PSFs...')
        fp_p2 += prop(wf_p2).intensity

        print('Making 3 of 3 PSFs...')
        fp_lt += prop(wf_lt).intensity

    Strehl = prop(wf_lt).intensity.max()/prop(wf_p1).intensity.max()

    return(fp_p1, fp_p2, fp_lt)



def make_ERIS_PSF(wlen0,bw,Npix):

    print(f'Starting to make PSF with wavelength {wlen0} and bandwidth {bwidth}')

    dtel = 8.4 # telescope diameter (meters)
    pscale = 12.25 # marcsec / pixel for ERIS

    sampling = .206265 * (wlen0 / dtel) / (pscale*1e-3)

    sampling = sampling * (385./406.) # 0.95 scale factor empirically measured

    print(f'setting the sampling to {sampling:5.3f} pixels')

    pupil_grid = make_pupil_grid(Npix)
    focal_grid = make_focal_grid(q=sampling, num_airy=100)

    Pupil_flattened = Field(Pupil.ravel(),pupil_grid)
    vAPP_flattened = Field(vAPP.ravel(),pupil_grid)
    (fp1, fp2, fplt) = phase_to_PSF(Pupil_flattened,vAPP_flattened,pupil_grid,focal_grid, wlen0, bwidth)

    fig, ax = plt.subplots(1,1,figsize=(8,8))

    lt_flux_frac = 0.02
    p1_flux_frac = 0.5
    p2_flux_frac = 0.5

    print('generating the PSF...')
    gvapp = (p1_flux_frac-(lt_flux_frac/2.)) * fp1 + \
        (p2_flux_frac-(lt_flux_frac/2.)) * fp2 + \
         lt_flux_frac * fplt

    imshow_field(np.log10(gvapp/gvapp.max()),vmin = -5.0,vmax = 0)
    ax.set_title(f'gvAPP PSF {filtname} with central wavelength {wlen0} microns and bandwidth {bwidth} microns')

    print('Flip and rotating the PSF...')

    normPSF = (gvapp/gvapp.max()).shaped
    # flip the image horizontally
    normPSF=normPSF[::-1,:]
    plt.imshow(np.log10(normPSF),origin='lower')


    normPSFrot = ndimage.rotate(normPSF, 35.85, reshape=False)

    return normPSFrot


# read in gvAPP phase and amplitude
vAPP_name = 'ERIS_final_gvAPP.fits.gz'
vAPP = read_fits(vAPP_name)
Pupil_name = 'ERIS_final_amplitude.fits.gz'
Pupil = read_fits(Pupil_name)

Npix = vAPP.shape[0]

wlen0, bwidth, filtname = 4.05, 0.02, "Bra" # central wavelength and FWHM bwidth of the filter (microns)

normPSFrot = make_ERIS_PSF(wlen0,bwidth,Npix)
fout = f'ERIS_gvAPP_PSF_{filtname}_{wlen0}_{bwidth}.fits'
print(f'Writing image out to {fout}')
hdu2 = fits.PrimaryHDU(normPSFrot)
hdu2.writeto(fout, overwrite=True)

wlen0, bwidth, filtname = 3.97, 0.10, "Bra-cont" # central wavelength and FWHM bwidth of the filter (microns)

normPSFrot = make_ERIS_PSF(wlen0,bwidth,Npix)
fout = f'ERIS_gvAPP_PSF_{filtname}_{wlen0}_{bwidth}.fits'
print(f'Writing image out to {fout}')
hdu2 = fits.PrimaryHDU(normPSFrot)
hdu2.writeto(fout, overwrite=True)

wlen0, bwidth, filtname = 2.17, 0.02, "Brg" # central wavelength and FWHM bwidth of the filter (microns)

normPSFrot = make_ERIS_PSF(wlen0,bwidth,Npix)
fout = f'ERIS_gvAPP_PSF_{filtname}_{wlen0}_{bwidth}.fits'
print(f'Writing image out to {fout}')
hdu2 = fits.PrimaryHDU(normPSFrot)
hdu2.writeto(fout, overwrite=True)


wlen0, bwidth, filtname = 2.17, 0.02, "B_M" # central wavelength and FWHM bwidth of the filter (microns)
wlen0=(4.49+5.06)/2
bwidth = 5.06-4.49
normPSFrot = make_ERIS_PSF(wlen0,bwidth,Npix)
fout = f'ERIS_gvAPP_PSF_{filtname}_{wlen0}_{bwidth}.fits'
print(f'Writing image out to {fout}')
hdu2 = fits.PrimaryHDU(normPSFrot)
hdu2.writeto(fout, overwrite=True)

