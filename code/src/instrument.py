"""

This file get instrument setup depending on CHORUS settings.

------- Functions ---------
- CCD setup
+ class Instrument:
    + Blaze function, normalized to 1
        - grating info G (changing with flag), tp, theta, gamma
    - optical throughput, normalized to 1
    + FP lines, including FP interferometer info:
        - finesse: 18 (dimensionless)
        - cavity / mirror spacing: 7.8 mm = 7.8e6 nm
    + get wavelength, Blaze function and optical throughput
+ other inputs, including
    - flag / pupil num: 2 or 3 (3-slice as default)
    # - slit width: 17 pixels
    # - readout noise: 2.5 e-
---------------------------

Created by Chenyang Ji in 2024.11.17.

"""

import os
import json
import itertools
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import namedtuple
from typing import Tuple
from blaze import get_b

# -------------- directory path of CHORUS folder --------------
pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# >>> path of CHORUS folder

# -------------- universal constants --------------
Arg = namedtuple('Arg', ['flag', 'band', 'G', 'tb', 't', 'gamma', 'finesse', 'cavity'])
params = Arg(
    # CCD info, change here if you want to have different settings
    flag=[2, 3],
    band=['red', 'blue'],
    # grating info, this should be matched with slicing info above
    G=[31.6, 41.59], # 31.6 for 2-slice, 41.59 for 3-slice
    tb=np.deg2rad(75.5),
    t=0.0,
    gamma=np.deg2rad(1.0),
    # FP interferometer info
    finesse=18,
    cavity=7.8e6  # nm, 0.0078 m
    )

# -------------- CCD setup --------------
# NOTES in case: avoid repeated data read-in!!!
def CCD_setup(band: str, params: namedtuple, flag=3) -> Tuple[dict, tuple, tuple, tuple]:
    """
    Get CCD setup depending on CHORUS settings.
    Note that this function is developed only for 2- or 3-slice settings.

    Args:
        band (str): 'red' or 'red'
        flag (int): 2 or 3 (3-slice as default)

    Returns:
        ccd_data (dictionary): original CCD setup data, with 7 points per trace
        order_num (tuple, list): Order range of the CCD in this band.
        fiber_num (tuple, list): Fiber range of the CCD in this band.
        trace_num (tuple, list): Trace of the CCD in this band.
    """

    # Validate band input and flag input
    ## if you want to add more CCD settings,
    ## please upload the CCD data to ../data/CCD/ and modify the following code.
    if band not in params.band:
        raise ValueError("The 'band' argument must be in {params.band}.")
    if flag not in params.flag:
        raise ValueError("The 'flag' argument must be in {params.flag}.")

    # Build file paths dynamically
    base_path = os.path.abspath(os.path.join(pwd, './data/CCD/'))
    filename_map = {(f, b): f'V{b[0].upper()}{f}.json' for f, b in itertools.product(params.flag, params.band)}
    ccd_filename = filename_map.get((flag, band), 'default.json')  # personalize your filename
    # if the filename is valid with band and flag fixed
    if not ccd_filename:
        raise FileNotFoundError(f"No CCD file found for {band} band with {flag}-slice setting.")
    ccd_file_path = os.path.join(base_path, ccd_filename)
    # if the file exists on the disk
    if not os.path.isfile(ccd_file_path):
        raise FileNotFoundError(f"CCD file not found on disk: {ccd_file_path}")

    # get CCD setup
    with open(ccd_file_path, 'r') as f:
        ccd_data = json.load(f)
    order_num = list(ccd_data.keys())
    fiber_num = list(ccd_data[order_num[0]].keys())
    trace_num = list(ccd_data[order_num[0]][fiber_num[0]].keys())

    return ccd_data, order_num, fiber_num, trace_num

# -------------- define a class for this instrument input --------------
class Instrument():
    def __init__(self, ccd_data, flag, band, order, fiber, trace, params):
        # input values
        self.ccd_data = ccd_data
        self.flag = flag
        self.band = band
        self.order = order
        self.fiber = fiber
        self.trace = trace
        self.params = params
        # output values
        self.wavelength, self.blaze, self.throughput = self.profile_setup(self.ccd_data, self.order, self.fiber, self.trace, select_blaze=False)
        self.norm_FP = self.FP_lines(self.wavelength, self.params)

    # Blaze function
    @staticmethod
    def __get_blaze(wavelength: np.ndarray | float, order: str, arg: namedtuple, flag=3) -> np.ndarray | float:
        """
        Get Blaze function B(x) for a given order and wavelength.
        The plotting script of the Blaze function is found in '../data/BlazeFunction/plot_wave*.py'.
        Credit to Kai Zhang, Liang Wang from CHORUS team.

        Args:
            wavelength (array): lambda(x) in nm
            order (int): Order number
            arg (namedtuple): universal constants (G, tb, t, gamma, finesse, cavity)
            flag (int): 2 or 3 (3-slice as default) for different G
        
        Returns:
            blaze (array): B(x) normalized to 1
        """

        # Blaze function, select G at givin flag and transfer order to float
        G = arg.G[arg.flag.index(flag)]
        blaze = get_b(wavelength, float(order), G, arg.tb, arg.t, arg.gamma)

        return blaze

    # Optical throughput
    @staticmethod
    def __get_throughput(wavelength: np.ndarray, band: str) -> np.ndarray | float:
        """
        Get optical throughput f_tp(x) for a given order and wavelength.
        The input file is located at '../data/3pupilslicer.mat'.
        Credit to Kai Zhang, Liang Wang from CHORUS team.

        Args:
            wavelength (array): lambda(x) in nm
            band (str): 'red' or 'red'
        
        Returns:
            throughput (array): f_tp(x) normalized to 1
        """

        # load throughput data
        tp_filename = os.path.abspath(os.path.join(pwd, './data/3pupilslicer.mat'))
        tp_data = scipy.io.loadmat(tp_filename)
        keyname_map = {
            ('red', 'wave'): 'wvr',
            ('red', 'tp'): 'opthobsr2',
            ('blue', 'wave'): 'wvb',
            ('blue', 'tp'): 'opthobsb2'
        }
        # get throughput data in this band
        wave_band = tp_data[keyname_map[(band, 'wave')]][0]
        tp_band = tp_data[keyname_map[(band, 'tp')]][0]

        # extrapolation
        f = interp1d(wave_band, tp_band, kind='cubic', fill_value='extrapolate')
        # normalize the throughput by dividing the maximum value in this band
        # red band: 587 - 781 nm, red band: 408 - 545 nm, roughly 8000 points (actually it is not evenly-spaced)
        band_range = {
            'red': np.linspace(587, 781, 8000),
            'blue': np.linspace(408, 545, 8000)
        }
        throughput = f(wavelength) / np.max(f(band_range[band]))

        return throughput

    # wavelength, Blaze function and optical throughput
    def profile_setup(self, ccd_data: dict, order: str, fiber: str, trace: str, select_blaze=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get wavelength, Blaze function and optical throughput from CCD setup.
        This is fitted to 8000 points for one trace.
        Save Blaze above 0.05 to remove the low SNR part. (can be removed)

        Args:
            ccd_data (dictionary): got from function CCD_setup
            order (int): Order number
            fiber (int): Fiber number, '1' for science and '2' for calibration
            trace (int): Trace number, '1' or '2' or '3' by default (3-slice)

        Returns:
            wavelength (array): lambda(x) in nm, 8000 points
            blaze (array): B(x) normalized to 1, 8000 points
            throughput (array): f_tp(x) normalized to 1, 8000 points
        """

        # get CCD setup, change flag if needed
        x = ccd_data[order][fiber][trace]['x']
        wave = [i*1000 for i in ccd_data[order][fiber][trace]['wave']]  # unit: nm

        # do polynomial fitting to get wavelength-pixel relation lambda(x)
        coefficients = np.polyfit(x, wave, deg=2)  # we set deg=2, and it seems the same with deg=3
        x_new = np.linspace(-40, 40, num=8000)
        wavelength = np.polyval(coefficients, x_new)

        # get blaze function B(x) and throughput f_tp(x)
        blaze = self.__get_blaze(wavelength, float(order), self.params, self.flag)
        throughput = self.__get_throughput(wavelength, self.band)

        # select blaze > 0.05 if select_blaze is True
        # CHORUS setting have blaze > 0.4 everywhere, so no very few pixels will be removed and we just skip it.
        if select_blaze:
            mask = blaze <= 0.05
            wavelength[mask] = np.nan
            blaze[mask] = np.nan
            throughput[mask] = np.nan

        return wavelength, blaze, throughput

    # FP lines
    def FP_lines(self, wavelength: np.ndarray | float, arg: namedtuple) -> np.ndarray | float:
        """
        Get FP lines info, including finesse and cavity length.

        Args:
            wavelength (array): lambda(x) in nm
            arg (namedtuple): universal constants (G, tb, t, gamma, finesse, cavity)
        Returns:
            norm_FP (array): normalized flux of FP interferometer theoretically
        """

        # calculate the normalized flux of FP interferometer theoretically
        norm_FP = 1 / (1 + arg.finesse * np.sin(2 * np.pi * arg.cavity / wavelength)**2)  # normalized flux of FP interferometer

        return norm_FP


# -------------- test --------------
test_flag = False
plot_flag = False

if __name__ == '__main__':
    if test_flag:
        # test CCD_setup
        ccd_data, order_num, fiber_num, trace_num = CCD_setup('blue', params, flag=3)

        # see how many pixels with low Blaze are removed
        for order in order_num:
            inst_order = Instrument(ccd_data, 3, 'blue', order, fiber_num[1], trace_num[0], params)
            print(f'Order {order}: {sum(np.isnan(inst_order.blaze))} pixels removed.')

        # test Instrument
        inst = Instrument(ccd_data, 3, 'blue', order_num[27], fiber_num[0], trace_num[0], params)

        if plot_flag:
            # plot wavelength, blaze and throughput
            plt.figure(figsize=(10, 4), dpi=200)
            plt.plot(inst.wavelength, inst.blaze, label='Blaze', zorder=5)
            plt.plot(inst.wavelength, inst.throughput, label='Throughput', zorder=5)
            plt.plot(inst.wavelength, inst.norm_FP, label='FP lines', linewidth=0.1, zorder=0)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized value')
            plt.legend()
            plt.title(f'Order {inst.order}, Fiber {inst.fiber}, Trace {inst.trace}')
            plt.tight_layout()
            plt.savefig(os.path.join(pwd, './output/test_plot/Instrument_test.png'), dpi=300)