import os
import alignement
import calibrations
import photometry
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from astropy.time import Time
from astropy.io import fits
import logging

######################################################################################
# INITIALIZATION
######################################################################################

extension = '.fit'
data_dir = 'WASP_41'
dark_path = 'calibration/darks/dark_30.fit'
flat_path = 'calibration/flat_b.fit'
dark = fits.open(dark_path)[0]
flat = fits.open(flat_path)[0]

# stars
stars = [[4784-3700, 3192-2400],
         [4922-3700, 3201-2400],
         [4706-3700, 2965-2400], # magnitude 12.56
         [4479-3700, 2934-2400]]

stamp_size = 10
radio_apertura = 3
radiosky_in = 5
radiosky_out = 9

######################################################################################
# CALIBRATION & ALIGNEMENT
######################################################################################

# load science images
data_files = glob(os.path.join(data_dir, '*'+extension))
data_files.sort() # TODO: SORT BY DATEOBS
sciences = []
times = []
for data_file in data_files:
    print(f"processing image: {data_file}")
    data_fits = fits.open(data_file)[0]
    sci = calibrations.calibrate(data_fits, flat, dark, subframe=[[3700,5700],[2400,3800]])
    sciences.append(sci)

    # time indexation
    ref_time = Time(data_fits.header['DATE-OBS'], format='isot').jd
    times.append(ref_time)

# alignement
print("start alignement")
original_stack = sciences # np.array(sciences) <- this works
aligned_stack = np.array(alignement.phase_correlation_alignment(original_stack))
print("alignement done")

######################################################################################
# PHOTOMETRY
######################################################################################

print("start photometry")

stamps = []
for star in stars:
    star_stamps = photometry.cut_stamp(aligned_stack, star, stamp_size)
    stamps.append(star_stamps)
    plt.imshow(np.sum(star_stamps,axis=0))
    plt.title(f"Sum stamp, star: {star}")
    plt.show()

stamps = np.array(stamps)
fluxes = np.zeros((len(stars), len(aligned_stack)))
fluxes_err =  np.zeros((len(stars), len(aligned_stack)))

# photometry for every star and stamp
for star in enumerate(stars):
    star_number = star[0]
    for stamp in enumerate(stamps[star_number]):
        stamp_number = stamp[0]
        print(f"processing star : {star_number}, stamp: {stamp_number}")

        centroide = photometry.centroid(stamp[1])
        flux, flux_err = photometry.aperture_phot(stamp[1], centroide, radio_apertura, radiosky_in, radiosky_out)
        fluxes[star_number, stamp_number] = flux
        fluxes_err[star_number, stamp_number] = flux_err

print("photometry done.")

######################################################################################
# SAVING
######################################################################################

fluxes = fluxes.transpose()   
fluxes_err = fluxes_err.transpose()
fluxes_df = pd.DataFrame()
fluxes_df['time'] = np.array(times)
for star in enumerate(stars):
    fluxes_df['flux_star_'+str(star[0])] = fluxes[:,star[0]]
    fluxes_df['err_flux_star_'+str(star[0])] = fluxes_err[:,star[0]]
fluxes_df.to_csv('fluxes.csv', index=False)