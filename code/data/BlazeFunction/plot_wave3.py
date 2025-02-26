#!/usr/bin/env python3
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import scipy.integrate as intg
import scipy.signal as sg
import os

# print('getcwd (blaze): ', os.getcwd())
# print('__file__ (blaze): ', __file__)

def vacuum_to_air(wl_vac, unit='Angstrom', ref='Ciddor1996'):
    """
    Convert vacuum wavelength to air wavelength.

    Args:
        wl_vac (float): Vacuum wavelength in unit of Angstrom
        ref (str): Reference.
    Returns:
        float: Air wavelength
    See also:
        * :func: `air_to_vacuum`
    """

    if ref in ['IAU', 'Edlen1953']:
        a, b1, b2, c1, c2 = 6.4328e-5, 2.94981e-2, 2.5540e-4, 146.0, 41.0
        eq = 1
    elif ref=='Edlen1966':
        a, b1, b2, c1, c2 = 8.34213e-5, 2.406030e-2, 1.5997e-4, 130.0, 38.9
        eq = 1
    elif ref=='Peck1972':
        a, b1, b2, c1, c2 = 0.0, 5.791817e-2, 1.67909e-3, 238.0185, 57.362
        eq = 1
    elif ref=='Birch1994':
        a, b1, b2, c1, c2 = 8.34254e-5, 2.406147e-2, 1.5998e-4, 130.0, 38.9
        eq = 1
    elif ref=='Ciddor1996':
        a, b1, b2, c1, c2 = 0.0, 5.792105e-2, 1.67917e-3, 238.0185, 57.362
        eq = 1
    elif ref=='SDSS':
        eq = 2

    if eq == 1:
        # convert wavelength to wavenumber in micron
        if unit == 'Angstrom':
            wn = 1e4/wl_vac
        elif unit == 'micron':
            wn = 1./wl_vac
        elif unit == 'nm':
            wn = 1e3/wl_vac
        n = 1. + a + b1/(c1-wn**2) + b2/(c2-wn**2)

    elif eq == 2:
        # convert wavelength to wavenumber in Angstrom
        if unit == 'Angstrom':
            wn = 1/wl_vac
        elif unit == 'micron':
            wn = 1e-4/wl_vac
        elif unit == 'nm':
            wn = 1e-1/wl_vac

        n = 1. + 2.735182e-4 + 131.4182*wn**2 + 2.76249e8*wn**4

    return wl_vac/n


def air_to_vacuum(wl_air, unit='Angstrom' ,ref='Ciddor1996'):
    """
    Convert air wavelength to vacuum wavelength.

    Args:
        wl_air (float): Air wavelength in unit of Angstrom
        ref (str): Reference.
    Returns:
        float: Air wavelength
    See also:
        * :func:`vacuum_to_air`
    """

    wl = wl_air
    for i in range(3):
        r = wl/vacuum_to_air(wl, unit=unit, ref=ref)
        wl = wl_air*r
    return wl


def get_blazewave(G, tb, m, t, gamma):
    return 2*np.sin(tb)*np.cos(t)*np.cos(gamma)/m/(G*1e-6)

def get_beta(wave, m, G, tb, t, gamma):
    alpha = tb + t
    v = m*wave*G*1e-6/np.cos(gamma) - np.sin(alpha)
    return np.arcsin(v)

def get_sincv(wave, m, G, tb, t, gamma):
    alpha = tb + t
    f = 1/(G*1e-6)*np.cos(tb)
    #f2 = f*np.cos(alpha)/np.cos(t)
    f2 = f - f*np.tan(tb)*np.tan(t)
    beta = get_beta(wave, m, G, tb, t, gamma)
    v = np.pi*f2/wave*(np.sin(alpha-tb) + np.sin(beta-tb))
    return np.sin(v)/v

def BF(wave, m, G, tb, t, gamma):
    sincv = get_sincv(wave, m, G, tb, t, gamma)
    func = sincv**2
    return func




f = fits.open('./code/packages/BlazeFunction/Kurucz_1984.fits')
data = f[1].data
f.close()

ww1 = 4000
ww2 = 7850
i1 = np.searchsorted(data['wavelength'], ww1)
i2 = np.searchsorted(data['wavelength'], ww2)
wave = data['wavelength'][i1:i2]
flux = data['flux'][i1:i2]

fig = plt.figure(figsize=(10, 4), dpi=200)
ax = fig.add_axes([0.06, 0.15, 0.69, 0.75])
ax.plot(wave, flux, '-', c='C3', lw=0.5, alpha=0.4, label='Solar Spectrum')

t = Table.read('./code/packages/BlazeFunction/linelist_UV.dat', format='ascii.fixed_width_two_line')
for row in t:
    wl_air  = row['wl_air']
    species = row['species']
    # if ww1 < wl_air < ww2:
    #     ax.axvline(wl_air, ls='--', c='C3')
    #     ax.text(wl_air, 0.8, species, ha='left')
t = Table.read('./code/packages/BlazeFunction/linelist_opt.dat', format='ascii.fixed_width_two_line')
for row in t[1:]:
    wl_air  = row['wl_air']
    species = row['species']
    # if ww1 < wl_air < ww2:
    #     ax.axvline(wl_air, ls='--', c='C3')
    #     ax.text(wl_air, 0.8, species, ha='left')


##############################################################

# delta_tb = 0.25
# G = 79
# tb = np.deg2rad(63)
# gamma = np.deg2rad(0.75)
# theta = 0.0
#
# for im, m in enumerate(np.arange(55, 62)):
#    blaze_wave0 = get_blazewave(G, tb, m, 0, 0)
#    fsr = blaze_wave0/m
#    w2 = (np.sin(tb)+1)/m/(G*1e-6)
#    w1 = blaze_wave0 - (w2 - blaze_wave0)
#    wave = np.arange(w1, w2, 0.001)
#    y0 = BF(wave, m, G, tb, 0, 0)
#    intensity0 = intg.simpson(y0, wave)
#
#    blaze_wave = get_blazewave(G, tb, m, theta, gamma)
#    fsr = blaze_wave/m
#    w2 = (np.sin(tb+theta)+1)*np.cos(gamma)/m/(G*1e-6)
#    w1 = blaze_wave - (w2 - blaze_wave)
#    wave = np.arange(w1, w2, 0.001)
#    y2 = BF(wave, m, G, tb, theta, gamma)
#    intensity = intg.simpson(y2, wave)
#    ratio = intensity/intensity0
#
#    if ww1 < blaze_wave0*10 < ww2:
#        ax.text(blaze_wave0*10, 1.02, '{}'.format(m), c='C0', ha='center')
#
#    if im==0:
#        label='UVS, $G$={} l/mm\n$\\theta_{{\\rm B}}={:.1f}\pm{}^\circ \gamma={:.2f}^\circ$'.format(
#                G, np.rad2deg(tb), delta_tb, np.rad2deg(gamma))
#    else:
#        label=None
#    ax.plot(wave*10, y2/ratio, c='C0', label=label)
#
# tb = np.deg2rad(63+delta_tb)
# for im, m in enumerate(np.arange(55, 62)):
#    blaze_wave0 = get_blazewave(G, tb, m, 0, 0)
#    fsr = blaze_wave0/m
#    w2 = (np.sin(tb)+1)/m/(G*1e-6)
#    w1 = blaze_wave0 - (w2 - blaze_wave0)
#    wave = np.arange(w1, w2, 0.001)
#    y0 = BF(wave, m, G, tb, 0, 0)
#    intensity0 = intg.simpson(y0, wave)
#
#    blaze_wave = get_blazewave(G, tb, m, theta, gamma)
#    fsr = blaze_wave/m
#    w2 = (np.sin(tb+theta)+1)*np.cos(gamma)/m/(G*1e-6)
#    w1 = blaze_wave - (w2 - blaze_wave)
#    wave = np.arange(w1, w2, 0.001)
#    y2 = BF(wave, m, G, tb, theta, gamma)
#    intensity = intg.simpson(y2, wave)
#    ratio = intensity/intensity0
#
#    ax.plot(wave*10, y2/ratio, c='C0', alpha=0.4)
#
# tb = np.deg2rad(63-delta_tb)
# for im, m in enumerate(np.arange(55, 62)):
#    blaze_wave0 = get_blazewave(G, tb, m, 0, 0)
#    fsr = blaze_wave0/m
#    w2 = (np.sin(tb)+1)/m/(G*1e-6)
#    w1 = blaze_wave0 - (w2 - blaze_wave0)
#    wave = np.arange(w1, w2, 0.001)
#    y0 = BF(wave, m, G, tb, 0, 0)
#    intensity0 = intg.simpson(y0, wave)
#
#    blaze_wave = get_blazewave(G, tb, m, theta, gamma)
#    fsr = blaze_wave/m
#    w2 = (np.sin(tb+theta)+1)*np.cos(gamma)/m/(G*1e-6)
#    w1 = blaze_wave - (w2 - blaze_wave)
#    wave = np.arange(w1, w2, 0.001)
#    y2 = BF(wave, m, G, tb, theta, gamma)
#    intensity = intg.simpson(y2, wave)
#    ratio = intensity/intensity0
#
#    ax.plot(wave*10, y2/ratio, c='C0', alpha=0.4)

##############################################################

G = 41.59  # 3-slice
# G = 31.6  # 2-slice
tb = np.deg2rad(75.5)
delta_tb = 0.25
gamma = np.deg2rad(1.0)
theta = 0.0

def getw1w2(m, theta, gamma):
    w2 = (np.sin(tb + theta) + 1) * np.cos(gamma) / m / (G * 1e-6)
    blazewave = get_blazewave(G, tb, m, theta, gamma)
    w1 = blazewave - (w2 - blazewave)
    return w1, w2

def get_b(wavelength, m, G, tb, theta, gamma):
    y2 = BF(wavelength, m, G, tb, theta, gamma)
    intensity = intg.simpson(y2, wavelength)
    blaze = y2/max(y2)
    # print('blaze in get_b(): ', blaze)
    return blaze



for im, m in enumerate(np.arange(79, 150)):
   # blaze_wave0 = get_blazewave(G, tb, m, 0, 0)
   # fsr = blaze_wave0/m
   # w1, w2 = getw1w2(m, theta=0, gamma=0)
   # # w2 = (np.sin(tb)+1)/m/(G*1e-6)
   # # w1 = blaze_wave0 - (w2 - blaze_wave0)
   # wave = np.arange(w1, w2, 0.001)
   # # y0 = BF(wave, m, G, tb, 0, 0)
   # # intensity0 = intg.simpson(y0, wave)

   blaze_wave = get_blazewave(G, tb, m, theta, gamma)
   fsr = blaze_wave/m
   w1, w2 = getw1w2(m, theta, gamma)
   # w2 = (np.sin(tb+theta)+1)*np.cos(gamma)/m/(G*1e-6)
   # w1 = blaze_wave - (w2 - blaze_wave)
   wave = np.arange(w1, w2, 0.001)
   # y2 = BF(wave, m, G, tb, theta, gamma)
   # intensity = intg.simpson(y2, wave)
   # ratio = intensity/intensity0
   blaze = get_b(wave, m, G, tb, theta, gamma)

   if ww1 < get_blazewave(G, tb, m, 0, 0)*10 < ww2:
       ax.text(vacuum_to_air(get_blazewave(G, tb, m, 0, 0)*10), 1.02, '{}'.format(m),
               c='C2', ha='center')
   if im==0:
       label='Blaze function of VIS,\n$G$={} l/mm\n$\\theta_{{\\rm B}}={:.1f}^\circ$, $\gamma={:.1f}^\circ$'.format(
               G, np.rad2deg(tb), delta_tb, np.rad2deg(gamma))
   else:
       label=None
   # ax.plot(vacuum_to_air(wave*10), y2/ratio, c='C2', label=label)
   ax.plot(vacuum_to_air(wave * 10), blaze, c='C2', label=label)

#
#tb = np.deg2rad(75.5+delta_tb)
#
#for im, m in enumerate(np.arange(59, 116)):
#    blaze_wave0 = get_blazewave(G, tb, m, 0, 0)
#    fsr = blaze_wave0/m
#    w2 = (np.sin(tb)+1)/m/(G*1e-6)
#    w1 = blaze_wave0 - (w2 - blaze_wave0)
#    wave = np.arange(w1, w2, 0.001)
#    y0 = BF(wave, m, G, tb, 0, 0)
#    intensity0 = intg.simpson(y0, wave)
#
#    blaze_wave = get_blazewave(G, tb, m, theta, gamma)
#    fsr = blaze_wave/m
#    w2 = (np.sin(tb+theta)+1)*np.cos(gamma)/m/(G*1e-6)
#    w1 = blaze_wave - (w2 - blaze_wave)
#    wave = np.arange(w1, w2, 0.001)
#    y2 = BF(wave, m, G, tb, theta, gamma)
#    intensity = intg.simpson(y2, wave)
#    ratio = intensity/intensity0
#
#    ax.plot(vacuum_to_air(wave*10), y2/ratio, c='C2', alpha=0.4)

# tb = np.deg2rad(75.5-delta_tb)
#
# #for im, m in enumerate(np.arange(59, 116)):
# for im, m in enumerate(np.arange(105, 116)):
#     blaze_wave0 = get_blazewave(G, tb, m, 0, 0)
#     fsr = blaze_wave0/m
#     w2 = (np.sin(tb)+1)/m/(G*1e-6)
#     w1 = blaze_wave0 - (w2 - blaze_wave0)
#     wave = np.arange(w1, w2, 0.001)
#     y0 = BF(wave, m, G, tb, 0, 0)
#     intensity0 = intg.simpson(y0, wave)
#
#     blaze_wave = get_blazewave(G, tb, m, theta, gamma)
#     fsr = blaze_wave/m
#     w2 = (np.sin(tb+theta)+1)*np.cos(gamma)/m/(G*1e-6)
#     w1 = blaze_wave - (w2 - blaze_wave)
#     wave = np.arange(w1, w2, 0.001)
#     y2 = BF(wave, m, G, tb, theta, gamma)
#     intensity = intg.simpson(y2, wave)
#     ratio = intensity/intensity0
#
#     ax.plot(vacuum_to_air(wave*10), y2/ratio, c='C2', alpha=0.4)


ax.xaxis.set_major_locator(tck.MultipleLocator(50))
ax.xaxis.set_minor_locator(tck.MultipleLocator(10))
ax.set_xlim(ww1, ww2)
ax.set_ylim(0, 1.1)
ax.grid(True, ls='--', lw=0.5)
ax.set_axisbelow(True)
ax.set_xlabel(u'Air Wavelength (\xc5)')
ax.legend(bbox_to_anchor=(1.02,1), loc=2)
# fig.savefig('UVS_VIS_boundary.png')
# fig.savefig('example.png')
# fig.savefig('blue.png')
# plt.show()
