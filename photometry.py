import photutils
import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
import pandas as pd
import matplotlib.pyplot as plt


def cut_stamp(frames, center, radius):
    stamps = frames[:, center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
    return stamps


def centroid(stamp):
    total_flux = np.sum(stamp)
    y_coords, x_coords = np.indices(stamp.shape) 
    Cx = np.sum(x_coords * stamp) / total_flux
    Cy = np.sum(y_coords * stamp) / total_flux
    return Cx, Cy


def aperture_phot(stamp, centro, radio, skyradio_int, skyradio_ext):
    aperture = CircularAperture(centro, r=radio)
    annulus = CircularAnnulus(centro, r_in=skyradio_int, r_out=skyradio_ext)
    
    # APERTURE ON SOURCE
    phot_table = aperture_photometry(stamp, [aperture, annulus])
    
    # SKY FLUX EXTIMATION
    annulus_mask = annulus.to_mask(method='center')
    annulus_data = annulus_mask.multiply(stamp)
    sky_pixels = annulus_data[annulus_data != 0]
    sky_mean, sky_median, sky_std = sigma_clipped_stats(sky_pixels, sigma=3.0)
    sky_sum = sky_median * aperture.area
    
    # FLUX - BACKGROUND
    flux_source = phot_table['aperture_sum_0'][0]
    net_flux = flux_source - sky_sum
    
    # ERROR
    # Poisson in source (sqrt(N_counts))
    error_source = np.sqrt(flux_source)
    
    # Error in sky (ruido por píxel * sqrt(n_pix en apertura))
    error_sky = sky_std * np.sqrt(aperture.area)
    
    # Error total (suma en cuadratura)
    net_error = np.sqrt(error_source**2 + error_sky**2)
    
    return net_flux, net_error


def estimate_background(image, box_size=(50, 50), filter_size=(3, 3), sigma=3.0):
    sigma_clip = SigmaClip(sigma=sigma)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        image,
        box_size=box_size,
        filter_size=filter_size,
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator
    )
    return bkg.background, bkg.background_rms


def extract_sources(image, background, background_rms, threshold_sigma=5.0, npixels=10):
    image_sub = image - background
    threshold = threshold_sigma * background_rms
    segm = detect_sources(
        image_sub,
        threshold=threshold,
        npixels=npixels
    )
    
    # Crear catálogo
    cat = SourceCatalog(image_sub, segm)
    #catalog = cat.to_table().to_pandas() 
    return segm, cat

# TODO
def psf_phot():
    return

def optimize_parameters():
    return

def plot_image(image,title, nsigma=1):
    median = np.median(image)
    std = np.std(image)
    plt.imshow(image, 
           vmin=median-nsigma*std,
           vmax=median+nsigma*std,
           cmap='inferno',
           origin='lower')
    plt.title(title)
    plt.show()