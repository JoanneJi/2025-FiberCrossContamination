"""
This module contains the implementation of the least square method to calculate RV.

The cost function is right, while minimization does not work... Maybe could be fixed later.

Created by Chenyang Ji in 2024.11.20.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
# ----------
import matplotlib.pyplot as plt

# speed of light in m/s
c = 2.99792458e8

# residual, based on sum((obs^2 - model^2)*weight)
def cost_function(v: float, model_wave: np.ndarray, model_flux: np.ndarray, obs_wave: np.ndarray, obs_flux: np.ndarray, weight: np.ndarray, psf: np.ndarray) -> np.ndarray:
    # remove Doppler shift from the observed spectrum
    shifted_obs_wave = obs_wave / (1 + v/c)
    # add padding and select wavelength range from template
    padding = 0.1
    chind = np.where((model_wave >= min(shifted_obs_wave)-padding) & (model_wave <= max(shifted_obs_wave)+padding))[0]
    trace_model_wave = model_wave[chind]
    trace_model_flux = model_flux[chind]

    # convolve with psf if psf is not None
    if psf is not None:
        trace_model_flux_conv = np.convolve(trace_model_flux, psf, mode='same')
    else:
        trace_model_flux_conv = trace_model_flux

    # interpolate the model spectrum to the shifted observed spectrum
    f = interp1d(trace_model_wave, trace_model_flux_conv, kind='cubic', fill_value="extrapolate")  # maybe no need to extrapolate?? test later
    shifted_model_flux = f(shifted_obs_wave)

    # define a residual function
    residual = (obs_flux - shifted_model_flux) * np.sqrt(weight)

    # ----------
    # plot the residual for testing, just uncomment the following lines if you want to plot
    # if v == 1000:
    #     plt.figure(figsize=(10, 4), dpi=200)
    #     plt.plot(shifted_obs_wave, obs_flux, linewidth=0.8, label='observed flux', zorder=3)
    #     plt.plot(shifted_obs_wave, shifted_model_flux, linewidth=0.8, label='shifted model flux', zorder=5)
    #     # plt.plot(model_wave, model_flux, linewidth=0.8, label='model flux', zorder=5)
    #     sigma = 1 / np.sqrt(weight)
    #     plt.fill_between(shifted_obs_wave, obs_flux - sigma, obs_flux + sigma, color='gray', edgecolor=None, alpha=0.8, zorder=4)
    #     plt.xlabel('Wavelength (nm)')
    #     plt.ylabel('normalized value')
    #     plt.legend()
    #     plt.tight_layout()
    #     # plt.xlim([min(shifted_obs_wave), max(shifted_obs_wave)])
    #     plt.xlim(409, 409.3)
    #     plt.savefig('./output/rv_test.png', dpi=300)

    return np.sum(residual**2)

# get RV by minimizing the residual
def get_rv(model_wave: np.ndarray, model_flux: np.ndarray, obs_wave: np.ndarray, obs_flux: np.ndarray, weight: np.ndarray, psf: np.ndarray) -> float:
    # initial guess for redshift
    v0 = 0
    print('cost function at initial guess: ', cost_function(v0, model_wave, model_flux, obs_wave, obs_flux, weight, psf)**2)
    # test the Jacobian
    v_perturb = 100
    c1 = cost_function(v0, model_wave, model_flux, obs_wave, obs_flux, weight, psf)
    c2 = cost_function(v0 + v_perturb, model_wave, model_flux, obs_wave, obs_flux, weight, psf)
    numerical_gradient = (c2 - c1) / v_perturb
    print("Numerical gradient:", np.sum(numerical_gradient))

    # bounds
    # bounds = (-1e4, 1e4)
    # minimize the residual
    res = minimize(
    cost_function,
    v0,
    args=(model_wave, model_flux, obs_wave, obs_flux, weight, psf),
    # bounds=bounds,
    method='Powell',
)

    print(res)
    return res.x[0]

