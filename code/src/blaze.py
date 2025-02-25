#!/usr/bin/env python3
import numpy as np

# grating info
# G = 41.59  # 3-slice
# G = 31.6  # 2-slice
# tb = np.deg2rad(75.5)
# gamma = np.deg2rad(1.0)
# theta = 0.0

# blaze function calculation
def get_beta(wave, m, G, tb, theta, gamma):
    alpha = tb + theta
    v = m*wave*G*1e-6/np.cos(gamma) - np.sin(alpha)
    return np.arcsin(v)

def get_sincv(wave, m, G, tb, theta, gamma):
    alpha = tb + theta
    f = 1/(G*1e-6)*np.cos(tb)
    f2 = f - f*np.tan(tb)*np.tan(theta)
    beta = get_beta(wave, m, G, tb, theta, gamma)
    v = np.pi*f2/wave*(np.sin(alpha-tb) + np.sin(beta-tb))
    return np.sin(v)/v

def BF(wave, m, G, tb, theta, gamma):
    sincv = get_sincv(wave, m, G, tb, theta, gamma)
    func = sincv**2
    return func

def get_b(wavelength, m, G, tb, theta, gamma):
    y2 = BF(wavelength, m, G, tb, theta, gamma)
    blaze = y2/max(y2)
    return blaze

