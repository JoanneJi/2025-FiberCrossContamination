"""
This module contains the implementation of the cross-correlation method to calculate RV.

Created by Chenyang Ji in 2024.11.22.
"""

import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
# ----------
import matplotlib.pyplot as plt


# speed of light in m/s
c = 2.99792458e8

# using interpolation to get the min of CCF
def interp_min(vgrid: np.ndarray, corrvalue: np.ndarray) -> float:
    # returns just the RV
    corr_interp = interpolate.splrep(vgrid, corrvalue, s=0)
    vgrid1side = np.logspace(-3, np.log10(1e4), num=400)  # 4 times more than original
    # vgrid1side - starting small to avoid digitization for early chunks at 1 cm/s level
    vgrid_rs = np.concatenate((np.flipud(vgrid1side) * (-1.0), np.zeros(1), vgrid1side))
    corrvalue_rs = interpolate.splev(vgrid_rs, corr_interp, der=0)
    minarg = np.argmin(corrvalue_rs)

    return vgrid_rs[minarg]

# cross-correlation function
def get_rv(model_wave: np.ndarray, model_flux: np.ndarray, obs_wave: np.ndarray, obs_flux: np.ndarray, weight: np.ndarray, psf: np.ndarray) -> float:
    vgrid1side = np.logspace(-1, np.log10(1e4), 100)
    vgrid = np.concatenate((np.flipud(-vgrid1side), np.zeros(1), vgrid1side))
    ccf = np.zeros_like(vgrid)

    for i, rv in enumerate(vgrid):
        shifted_obs_wave = obs_wave / (1 + rv/c)
        # add padding and select wavelength range from template
        padding = 0.1
        chind = np.where((model_wave >= min(shifted_obs_wave)-padding) & (model_wave <= max(shifted_obs_wave)+padding))[0]
        trace_model_wave = model_wave[chind]
        trace_model_flux = model_flux[chind]

        # convolve with psf if psf is not all nan
        if not np.all(np.isnan(psf)):
            trace_model_flux_conv = np.convolve(trace_model_flux, psf, mode='same')
        else:
            trace_model_flux_conv = trace_model_flux

        # interpolate the model spectrum to the shifted observed spectrum
        f = interp1d(trace_model_wave, trace_model_flux_conv, kind='cubic')  # maybe no need to extrapolate?? test later
        shifted_model_flux = f(shifted_obs_wave)

        # calculate the CCF and RV
        ccf[i] = np.sum((obs_flux - shifted_model_flux)**2 * weight)

    best_rv = interp_min(vgrid, ccf)
    # print('CCF:', min(ccf), 'RV:', best_rv)

    # # plot the spectrum for testing, just uncomment the following lines if you want to plot
    # plt.figure(figsize=(10, 4), dpi=200)
    # plt.plot(shifted_obs_wave, obs_flux, linewidth=0.8, label='observed flux', zorder=3)
    # plt.plot(shifted_obs_wave, shifted_model_flux, linewidth=0.8, label='shifted model flux', zorder=5)
    # # plt.plot(model_wave, model_flux, linewidth=0.8, label='model flux', zorder=5)
    # sigma = np.zeros_like(weight)
    # sigma[weight!=0] = 1 / np.sqrt(weight[weight!=0])
    # plt.fill_between(shifted_obs_wave, obs_flux - sigma, obs_flux + sigma, color='gray', edgecolor=None, alpha=0.8, zorder=6)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('normalized value')
    # plt.legend()
    # plt.tight_layout()
    # plt.xlim([min(shifted_obs_wave), max(shifted_obs_wave)])
    # plt.xlim(409, 409.5)
    # plt.ylim(0, 1.2)
    # plt.savefig('./output/ccf_spec.png', dpi=300)

    # # plot the CCF for testing, just uncomment the following lines if you want to plot
    # plt.figure(figsize=(6, 4), dpi=200)
    # plt.scatter(vgrid, ccf)
    # plt.axvline(x=best_rv, color='r', linestyle='--', linewidth=1.2)
    # plt.text(best_rv, min(ccf), 'RV = {:.2f} m/s'.format(best_rv), color='r', fontsize=8, ha='right')
    # plt.xlabel('RV (m/s)')
    # plt.ylabel('CCF')
    # plt.tight_layout()
    # plt.savefig('./output/ccf_test.png', dpi=300)

    return best_rv
