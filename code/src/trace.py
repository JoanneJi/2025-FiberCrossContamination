"""

This file offers a way to simulate spectrum in one trace.
Note that this is developed for photon-limited RV precision, so SNR should be chosen > 10.

------- Functions ---------
- template input
+ class Trace:
    - convolution
    + resample with noise
        - Poisson noise
        - Gaussian readout noise
    + cross contamination kernel, imported from './contamination.py'
        - a flag to turn on/off
        - cal-sci: to trace 1
        - sci-sci: to all traces
    + generate RV using least-square method, imported from './least_square.py'.
        - a specific RV result for this trace
+ other inputs, including:
    - this order info: wavelength, Blaze function, throughput
    - template setting: wavelength, normalized flux
    - instrument: resolution (120,000), readout noise (2.5 e-), trace height (17 pixels)
    - observation: SNR (100 to 800) 
---------------------------

Created by Chenyang Ji in 2024.11.18.
---------------------------

===========================
------- Modifications ---------
- modify the FP line generated
- move contamination part from the trace
+ add an external class for contamination process
    - contamination is added in y direction across orders
    - wavelength solution for each trace is different
    + contamination for different traces
        - trace 1: FP + trace 2
        - trace 2: trace 1 + trace 3
        - trace 3: trace 2
- calculate RV only for science fiber

Modified by Chenyang Ji in 2024.11.26.

"""

import os
import numpy as np
import instrument
from collections import namedtuple
from typing import Tuple
from scipy.interpolate import interp1d
from ccf import get_rv
import matplotlib.pyplot as plt
import time


# -------------- directory path of CHORUS folder --------------
pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# >>> path of CHORUS folder

# -------------- other inputs --------------
Arg = namedtuple('Arg', ['temp_file', 'R', 'read_noise', 'trace_height','SNR_cal', 'SNR_sci', 'calsci_frac', 'scisci_frac'])
params = Arg(
    # instrument info based on 3-slice setting
    temp_file='sun.npz',
    R=120000,
    read_noise=2.5,
    trace_height=17,
    # observation info
    SNR_cal=100,
    SNR_sci=100,
    # contamination info, based on order separation of CHORUS
    calsci_frac=1e-5,
    scisci_frac=1e-3
    )

# -------------- flags --------------
Arg_flag = namedtuple('Arg_flag', ['R_flag', 'Poisson_noise_flag', 'read_noise_flag', 'FP_Poisson_noise_flag', 'FP_read_noise_flag', 'weight_flag'])
params_flags = Arg_flag(
    # turn on/off decreasing the spectral resolution to the instrumental setting for testing the code
    R_flag=True,
    # turn on/off adding noise for testing the code
    Poisson_noise_flag=True,
    read_noise_flag=True,
    # turn on/off noise flag of FP lines
    FP_Poisson_noise_flag=True,
    FP_read_noise_flag=True,
    # weight flag for extracting RV
    weight_flag=True
    )

# -------------- template input --------------
def get_template(arg: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the full template spectrum from input folder.
    The default template is in .npz format. Personalize the data loading method if needed.

    Args:
        arg (namedtuple): universal constants (R, read_noise, trace_height, SNR)

    Returns:
        temp_wave (array): the wavelength of the template
        temp_flux (array): the normalized flux of the template
    """

    # load the npz file
    temp_path = os.path.abspath(os.path.join(pwd, './input', arg.temp_file))
    temp_data = np.load(temp_path, allow_pickle=True)
    temp_wave = temp_data['arr_0']
    temp_flux = temp_data['arr_1']

    return temp_wave, temp_flux

# -------------- calibration line input --------------
def get_cal(inst, inst_arg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the calibration line spectrum from theoretical equation according to Fabry-Perot interferometer.
    You can modify this if your calibration line is generated from other methods.

    Args:
        inst (class): the instrument class, imported from ./instrument.py
        inst_arg (namedtuple): universal constants for instrument setup

    Returns:
        cal_wave (array): the wavelength of the calibration line
        cal_flux (array): the normalized flux of the calibration line
    """

    # generate a mock calibration line with high sampling
    trace_wave = inst.wavelength
    padding = 1
    cal_wave = np.linspace(min(trace_wave)-padding, max(trace_wave)+padding, len(trace_wave)*6)  # high sampling
    cal_flux = 1 / (1 + inst_arg.finesse * np.sin(2 * np.pi * inst_arg.cavity / cal_wave)**2)

    return cal_wave, cal_flux


# -------------- class Trace --------------
class Trace():
    def __init__(self, fiber, temp_wave, temp_flux, inst, arg, flags):
        # input info
        self.fiber = fiber
        self.temp_wave = temp_wave
        self.temp_flux = temp_flux
        self.inst = inst
        self.arg = arg
        self.flags = flags
        # output values
        self.syn_wave, self.syn_flux, self.psf, self.sigma = self.generate_spectrum(self.temp_wave, self.temp_flux, self.inst, self.arg, self.flags)


    # convolution process
    @staticmethod
    def __convolve(temp_wave: np.ndarray, temp_flux: np.ndarray, inst: type, arg: namedtuple, flags: namedtuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Convolve the template spectrum (R->inf and continuous) with a Gaussian PSF according to the instrumental setting.
        We can get the first-step synthetic spectrum with the spectral resolution needed, and continuous.

        Args:
            temp_wave (array): full wavelength range of the template
            temp_flux (array): normalized flux of the full template
            inst (class): the instrument class, imported from ./instrument.py
            arg (namedtuple): universal constants (R, read_noise, trace_height, SNR)
            flags (namedtuple): flags to turn on/off some functions

        Returns:
            conv_wave (array): wavelength of the convolved spectrum, with 0.1 nm padding added (will be removed in __resample())
            conv_flux (array): normalized flux of the convolved spectrum
            x_psf (array): x-axis of the PSF  ?? maybe repeated??
            psf (array): PSF depending on the spectral resolution
            sigma (float): sigma of the Gaussian PSF
        """

        if flags.R_flag:
            # add paddings to reach numerically stability
            padding = 0.2
            save_padding = 0.1
            chind = np.where((temp_wave >= min(inst.wavelength)-padding) & (temp_wave <= max(inst.wavelength)+padding))[0]
            delta_lambda = np.median(abs(temp_wave[chind[0:(len(chind)-1)]] - temp_wave[chind[1:]]))
            sigma_lambda = np.median(temp_wave[chind]) / arg.R  # lambda_resolution = sigma_lambda, unit = nm
            sigma = sigma_lambda / delta_lambda / 2.35  # PSF width = 2.35 pixels, unit of resolution = pixel

            # construct PSF
            x_psf = np.arange(-15*5, 15*5, 1.0)
            psf = np.exp(-0.5 * (x_psf / sigma) ** 2)
            psf /= np.trapezoid(psf, x_psf)

            # convolution
            conv_wave_0 = temp_wave[chind]
            conv_flux_0 = np.convolve(temp_flux[chind], psf, mode='same')
            # remove the padding
            used_chind = np.where((conv_wave_0 >= min(inst.wavelength)-save_padding) & (conv_wave_0 <= max(inst.wavelength)+save_padding))[0]
            conv_wave = conv_wave_0[used_chind]
            conv_flux = conv_flux_0[used_chind]

            return conv_wave, conv_flux, psf, sigma
        else:
            return temp_wave, temp_flux, None, None


    # resample to pixel grid
    @staticmethod
    def __resample(conv_wave: np.ndarray, conv_flux: np.ndarray, inst: type, arg: namedtuple, flags: namedtuple, fiber: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the convolved spectrum to 8000 pixels, add noise, and get photon counts in 8000 pixels.

        Args:
            conv_wave (array): wavelength of the convolved spectrum, with 0.1 nm padding added (will be removed in __resample())
            conv_flux (array): normalized flux of the convolved spectrum
            inst (class): the instrument class, imported from ./instrument.py
            arg (namedtuple): universal constants (R, read_noise, trace_height, SNR)
            flags (namedtuple): flags to turn on/off some functions

        Returns:
            res_wave (array): resampled wavelength of this trace
            res_flux (array): photon counts of this trace
        """

        # interpolate the convolved spectrum to the resampled wavelength
        f = interp1d(conv_wave, conv_flux, kind='cubic')  # will cubic interpolation be over-fitted??
        new_flux = [f(wave) for wave in inst.wavelength]

        # choose the SNR for different fiber
        if fiber == '1':
            SNR = arg.SNR_sci
            poisson = flags.Poisson_noise_flag
            read = flags.read_noise_flag
        if fiber =='2':
            SNR = arg.SNR_cal
            poisson = flags.FP_Poisson_noise_flag
            read = flags.FP_read_noise_flag

        # add noise
        if poisson:
            # Poisson noise
            poisson_flux = np.random.poisson(new_flux * inst.blaze * inst.throughput * SNR**2)
            if read:
                # Gaussian readout noise
                noise = np.random.normal(loc=0, scale=arg.read_noise, size=(len(poisson_flux), arg.trace_height)).sum(axis=1)
                rn_flux = poisson_flux + noise
                res_flux = np.maximum(rn_flux, 0)  # remove negative values for numerical stability
            else:
                res_flux = np.maximum(poisson_flux, 0)
        else:
            # no Poisson noise
            no_poisson_flux = new_flux * inst.blaze * inst.throughput * SNR**2
            if read:
                # Gaussian readout noise
                noise = np.random.normal(loc=0, scale=arg.read_noise, size=(len(no_poisson_flux), arg.trace_height)).sum(axis=1)
                rn_flux = no_poisson_flux + noise
                res_flux = np.maximum(rn_flux, 0)
            else:
                res_flux = np.maximum(no_poisson_flux, 0)

        # save the wavelength of this trace
        res_wave = inst.wavelength

        return res_wave, res_flux


    # synthetic wave, synthetic flux and photon counts of the calibration line
    def generate_spectrum(self, temp_wave: np.ndarray, temp_flux: np.ndarray, inst: type, arg: namedtuple, flags: namedtuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        The whole spectrum generation process, including convolution, resampling, and cross-contamination.

        Args:
            temp_wave (array): full wavelength range of the template / FP line
            temp_flux (array): normalized flux of the full template / FP line
            inst (class): the instrument class, imported from ./instrument.py
            arg (namedtuple): universal constants (R, read_noise, trace_height, SNR)
            flags (namedtuple): flags to turn on/off some functions

        Returns:
            syn_wave (array): wavelength of this trace
            syn_flux (array): photon counts of this trace
            psf (array): PSF depending on the spectral resolution
            sigma (float): sigma of the Gaussian PSF
        """

        conv_wave, conv_flux, psf, sigma = self.__convolve(temp_wave, temp_flux, inst, arg, flags)
        syn_wave, syn_flux = self.__resample(conv_wave, conv_flux, inst, arg, flags, self.fiber)

        return syn_wave, syn_flux, psf, sigma


# -------------- RV extraction --------------
def extract_RVmat(temp_wave: np.ndarray, temp_flux: np.ndarray, syn_wave: np.ndarray, syn_flux: np.ndarray, blaze: np.ndarray, throughput: np.ndarray, psf: np.ndarray, arg: namedtuple, flags: namedtuple) -> np.ndarray:
    """
        Extract RV of this trace using least square method, with the main procedure imported from ./ccf.py.
        Note that synthetic spectrum should be normalized before fitting.

        Args:
            temp_wave (array): full wavelength range of the template
            temp_flux (array): normalized flux of the full template
            syn_wave (matrix): synthetic wavelength matrix, with shape of (len(order_num), 3, 8000)
            syn_flux (matrix): photon counts matrix, with shape of (len(order_num), 3, 8000)
            blaze (matrix): Blaze function matrix, with shape of (len(order_num), 3, 8000)
            throughput (matrix): throughput matrix, with shape of (len(order_num), 3, 8000)
            psf (matrix): PSF matrix, with shape of (len(order_num), 3, 15*5*2)
            arg (namedtuple): universal constants (R, read_noise, trace_height, SNR)
            flags (namedtuple): flags to turn on/off some functions

        Returns:
            rv (matrix): RV of each trace in each order, with shape of (len(order_num), 3)
    """

    # initialize weight function
    weight = np.zeros_like(syn_flux)

    # normalize the synthetic spectrum
    norm_syn_flux = syn_flux / (blaze * throughput * arg.SNR_sci**2)
    if flags.weight_flag:
        # weight of normalized error, to decrease the weight of low-SNR part at the edge
        # 1 / normalized error**2
        sigma_mat = np.sqrt(syn_flux) / (blaze * throughput * arg.SNR_sci**2)
        # only consider SNR>1 data points, for CCD counts cannot be below 1
        weight[(sigma_mat!=0) & (np.sqrt(syn_flux) > 1)] = 1 / sigma_mat[(sigma_mat!=0) & (np.sqrt(syn_flux) > 1)]**2
    else:
        weight = np.ones_like(syn_flux)

    # calculate RV
    rv_mat = np.zeros_like(syn_wave[:, :, 0])
    for iorder in range(len(syn_wave)):
        print("RV extraction process: order", iorder)
        for itrace in range(len(syn_wave[0])):
            rv_mat[iorder, itrace] = get_rv(temp_wave, temp_flux, syn_wave[iorder, itrace], norm_syn_flux[iorder, itrace], weight[iorder, itrace], psf[iorder, itrace])

    return rv_mat


# -------------- contamination kernel --------------
def contamination(temp_wave: np.ndarray, temp_flux: np.ndarray, band: str, flag: int, inst_args: namedtuple, args: namedtuple, flags: namedtuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adding cross contamination to the synthetic spectrum.

    Args:
        temp_wave (array): full wavelength range of the template
        temp_flux (array): normalized flux of the full template
        band (str): the band of the instrument
        flag (int): the number of slices for the instrument
        inst_args (namedtuple): universal constants for instrument setup
        args (namedtuple): universal constants for simulation setup
        flags (namedtuple): flags to turn on/off some functions

    Returns:
        syn_wave_mat (matrix): synthetic wavelength matrix, with shape of (len(order_num), 3, 8000)
        syn_flux_mat (matrix): photon counts matrix, with shape of (len(order_num), 3, 8000)
        blaze_mat (matrix): Blaze function matrix, with shape of (len(order_num), 3, 8000)
        throughput_mat (matrix): throughput matrix, with shape of (len(order_num), 3, 8000)
        sigma_mat (matrix): sigma matrix, with shape of (len(order_num), 3)
    """

    # get the instrument
    ccd_data, order_num, fiber_num, trace_num = instrument.CCD_setup(band, inst_args, flag)

    # initialize the matrices
    syn_wave_mat = np.zeros((len(order_num), len(trace_num), 8000))
    syn_flux_mat = np.zeros((len(order_num), len(trace_num), 8000))
    blaze_mat = np.zeros((len(order_num), len(trace_num), 8000))
    throughput_mat = np.zeros((len(order_num), len(trace_num), 8000))
    sigma_mat = np.zeros((len(order_num), len(trace_num)))
    psf_mat = np.zeros((len(order_num), len(trace_num), len(np.arange(-15*5, 15*5, 1.0))))

    # loop through all orders
    for ind, iorder in enumerate(order_num):
        print("spectrum generation process: order", ind)
        # the 3rd trace of calibration line in this order
        inst_cal = instrument.Instrument(ccd_data, flag, band, iorder, '2', '3', inst_args)
        cal_input_wave, cal_input_flux = get_cal(inst_cal, inst_args)
        cal3_trace = Trace('2', cal_input_wave, cal_input_flux, inst_cal, args, flags)
        # the three traces of science line in this order
        sci1_trace = Trace('1', temp_wave, temp_flux, instrument.Instrument(ccd_data, flag, band, iorder, '1', '1', inst_args), args, flags)
        sci2_trace = Trace('1', temp_wave, temp_flux, instrument.Instrument(ccd_data, flag, band, iorder, '1', '2', inst_args), args, flags)
        sci3_trace = Trace('1', temp_wave, temp_flux, instrument.Instrument(ccd_data, flag, band, iorder, '1', '3', inst_args), args, flags)

        # cross contamination issue
        # trace 1
        sci1_wave = sci1_trace.syn_wave
        sci1_flux = sci1_trace.syn_flux * (1 - args.scisci_frac) + args.calsci_frac * cal3_trace.syn_flux + args.scisci_frac * sci2_trace.syn_flux
        syn_wave_mat[ind, 0, :] = sci1_wave
        syn_flux_mat[ind, 0, :] = sci1_flux
        blaze_mat[ind, 0, :] = sci1_trace.inst.blaze
        throughput_mat[ind, 0, :] = sci1_trace.inst.throughput
        sigma_mat[ind, 0] = sci1_trace.sigma
        psf_mat[ind, 0, :] = sci1_trace.psf

        # trace 2
        sci2_wave = sci2_trace.syn_wave
        sci2_flux = sci2_trace.syn_flux * (1 - 2*args.scisci_frac) + args.scisci_frac * sci1_trace.syn_flux + args.scisci_frac * sci3_trace.syn_flux
        syn_wave_mat[ind, 1, :] = sci2_wave
        syn_flux_mat[ind, 1, :] = sci2_flux
        blaze_mat[ind, 1, :] = sci2_trace.inst.blaze
        throughput_mat[ind, 1, :] = sci2_trace.inst.throughput
        sigma_mat[ind, 1] = sci2_trace.sigma
        psf_mat[ind, 1, :] = sci2_trace.psf

        # trace 3
        sci3_wave = sci3_trace.syn_wave
        sci3_flux = sci3_trace.syn_flux * (1 - args.scisci_frac) + args.scisci_frac * sci2_trace.syn_flux
        syn_wave_mat[ind, 2, :] = sci3_wave
        syn_flux_mat[ind, 2, :] = sci3_flux
        blaze_mat[ind, 2, :] = sci3_trace.inst.blaze
        throughput_mat[ind, 2, :] = sci3_trace.inst.throughput
        sigma_mat[ind, 2] = sci3_trace.sigma
        psf_mat[ind, 2, :] = sci3_trace.psf

    return syn_wave_mat, syn_flux_mat, blaze_mat, throughput_mat, psf_mat, sigma_mat


# -------------- test --------------
save_flag = False
test_flag = False
plot_flag = False

if __name__ == '__main__':
    # time the process
    start = time.time()

    # fix the random seed for testing
    if test_flag:
        np.random.seed(20)

    # get the template
    temp_wave, temp_flux = get_template(params)

    # we consider 3-slice setting below
    flag = 3

# -------------- test with saving --------------
    if save_flag:
        for iband in ['blue', 'red']:
            syn_wave_mat, syn_flux_mat, blaze_mat, throughput_mat, psf_mat, sigma_mat = contamination(temp_wave, temp_flux, iband, flag, instrument.params, params, params_flags)
            rv_mat = extract_RVmat(temp_wave, temp_flux, syn_wave_mat, syn_flux_mat, blaze_mat, throughput_mat, psf_mat, params, params_flags)
            # save the results
            syn_filename = os.path.join(pwd, f'./output/synfile_{iband}.npz')
            np.savez(syn_filename, wave=syn_wave_mat, flux=syn_flux_mat, blaze=blaze_mat, throughput=throughput_mat, sigma=sigma_mat)

# -------------- test without saving --------------
    if test_flag:
        band = 'blue'
        order = '113'
        ccd_data, order_num, fiber_num, trace_num = instrument.CCD_setup(band, instrument.params, flag=flag)
        syn_wave_mat = np.zeros((1, len(trace_num), 8000))
        syn_flux_mat = np.zeros((1, len(trace_num), 8000))
        blaze_mat = np.zeros((1, len(trace_num), 8000))
        throughput_mat = np.zeros((1, len(trace_num), 8000))
        sigma_mat = np.zeros((1, len(trace_num)))
        psf_mat = np.zeros((1, len(trace_num), len(np.arange(-15*5, 15*5, 1.0))))

        # the 3rd trace of calibration line in this order
        inst_cal = instrument.Instrument(ccd_data, flag, band, order, '2', '3', instrument.params)
        cal_input_wave, cal_input_flux = get_cal(inst_cal, instrument.params)
        cal3_trace = Trace('2', cal_input_wave, cal_input_flux, inst_cal, params, params_flags)
        # the three traces of science line in this order
        sci1_trace = Trace('1', temp_wave, temp_flux, instrument.Instrument(ccd_data, flag, band, order, '1', '1', instrument.params), params, params_flags)
        sci2_trace = Trace('1', temp_wave, temp_flux, instrument.Instrument(ccd_data, flag, band, order, '1', '2', instrument.params), params, params_flags)
        sci3_trace = Trace('1', temp_wave, temp_flux, instrument.Instrument(ccd_data, flag, band, order, '1', '3', instrument.params), params, params_flags)
        # cross contamination issue
        # trace 1
        sci1_wave = sci1_trace.syn_wave
        sci1_flux = sci1_trace.syn_flux * (1 - params.scisci_frac) + params.calsci_frac * cal3_trace.syn_flux + params.scisci_frac * sci2_trace.syn_flux
        syn_wave_mat[0, 0, :] = sci1_wave
        syn_flux_mat[0, 0, :] = sci1_flux
        blaze_mat[0, 0, :] = sci1_trace.inst.blaze
        throughput_mat[0, 0, :] = sci1_trace.inst.throughput
        sigma_mat[0, 0] = sci1_trace.sigma
        psf_mat[0, 0, :] = sci1_trace.psf
        # trace 2
        sci2_wave = sci2_trace.syn_wave
        sci2_flux = sci2_trace.syn_flux * (1 - 2*params.scisci_frac) + params.scisci_frac * sci1_trace.syn_flux + params.scisci_frac * sci3_trace.syn_flux
        syn_wave_mat[0, 1, :] = sci2_wave
        syn_flux_mat[0, 1, :] = sci2_flux
        blaze_mat[0, 1, :] = sci2_trace.inst.blaze
        throughput_mat[0, 1, :] = sci2_trace.inst.throughput
        sigma_mat[0, 1] = sci2_trace.sigma
        psf_mat[0, 1, :] = sci2_trace.psf
        # trace 3
        sci3_wave = sci3_trace.syn_wave
        sci3_flux = sci3_trace.syn_flux * (1 - params.scisci_frac) + params.scisci_frac * sci2_trace.syn_flux
        syn_wave_mat[0, 2, :] = sci3_wave
        syn_flux_mat[0, 2, :] = sci3_flux
        blaze_mat[0, 2, :] = sci3_trace.inst.blaze
        throughput_mat[0, 2, :] = sci3_trace.inst.throughput
        sigma_mat[0, 2] = sci3_trace.sigma
        psf_mat[0, 2, :] = sci3_trace.psf

        # calculate RV of these tree traces in this test order
        rv_mat = extract_RVmat(temp_wave, temp_flux, syn_wave_mat, syn_flux_mat, blaze_mat, throughput_mat, psf_mat, params, params_flags)
        print(rv_mat)


    # -------------- a test plot --------------
    # plot the three traces of order 110 in the blue band, as well as the calibration line
    if plot_flag:
        iorder = '113'
        plt.figure(figsize=(10, 4), dpi=300)
        # # pixel scale
        # sci1 & sci2 & sci3
        plt.scatter(range(len(sci1_flux)), sci1_flux, label='sci 1', s=1.5, color='c')
        plt.scatter(range(len(sci2_flux)), sci2_flux, label='sci 2', s=1.5, color='m')
        plt.scatter(range(len(sci3_flux)), sci3_flux, label='sci 3', s=1.5, color='b')
        # cal1 & cal2 & cal3
        cal1_input_wave, cal1_input_flux = get_cal(instrument.Instrument(ccd_data, flag, iband, iorder, '2', '1', instrument.params), instrument.params)
        cal1_trace = Trace('2', cal1_input_wave, cal1_input_flux, instrument.Instrument(ccd_data, flag, iband, iorder, '2', '1', instrument.params), params, params_flags)
        cal2_input_wave, cal2_input_flux = get_cal(instrument.Instrument(ccd_data, flag, iband, iorder, '2', '2', instrument.params), instrument.params)
        cal2_trace = Trace('2', cal2_input_wave, cal2_input_flux, instrument.Instrument(ccd_data, flag, iband, iorder, '2', '2', instrument.params), params, params_flags)
        plt.plot(range(len(cal1_trace.syn_flux)), cal1_trace.syn_flux, label='cal 1', linewidth=0.4, color='c')
        plt.plot(range(len(cal2_trace.syn_flux)), cal2_trace.syn_flux, label='cal 2', linewidth=0.4, color='m')
        plt.plot(range(len(cal3_trace.syn_flux)), cal3_trace.syn_flux, label='cal 3', linewidth=0.4, color='b')
        # plt.xlim(4000, 4500)
        plt.xlabel('Pixel')
        # # wavelength scale
        # plt.scatter(sci1_wave, sci1_flux, label='sci 1', s=1.5, color='c')
        # plt.scatter(sci2_wave, sci2_flux, label='sci 2', s=1.5, color='m')
        # plt.scatter(sci3_wave, sci3_flux, label='sci 3', s=1.5, color='b')
        # plt.plot(cal1_trace.syn_wave, cal1_trace.syn_flux, label='cal 1', linewidth=0.4, color='c')
        # plt.plot(cal2_trace.syn_wave, cal2_trace.syn_flux, label='cal 2', linewidth=0.4, color='m')
        # plt.plot(cal3_trace.syn_wave, cal3_trace.syn_flux, label='cal 3', linewidth=0.4, color='b')
        # plt.xlim(423, 424)
        # plt.xlabel('wavelength (nm)')
        plt.ylabel('Photon counts')
        plt.title(f'Order {iorder}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pwd, './output/test_plot/trace_test_113.png'), dpi=300)


# # -------------- a simpler test plot --------------
#     # fiber_num = ['2', '1']
#     inst = instrument.Instrument(ccd_data, 3, 'blue', order_num[0], fiber_num[1], trace_num[0], instrument.params)
#     # get the trace
#     trace = Trace(temp_wave, temp_flux, inst, params, params_flags)
#     if test_flag:
#         print(f"RV of this trace: {trace.rv}m/s")

#     if plot_flag:
#         plt.figure(figsize=(10, 4), dpi=200)
#         plt.plot(trace.syn_wave, trace.syn_flux, label='synthetic spectrum', linewidth=0.5, zorder=5)
#         error = np.sqrt(trace.syn_flux)
#         plt.fill_between(trace.syn_wave, trace.syn_flux - error, trace.syn_flux + error, color='gray', edgecolor=None, alpha=0.5, zorder=4)
#         # plt.plot(trace.syn_wave, inst.blaze * params.SNR_sci**2, label='Blaze * SNR^2')
#         # plt.plot(trace.syn_wave, inst.throughput * params.SNR_sci**2, label='Throughput * SNR^2')
#         plt.plot(trace.syn_wave, inst.blaze * inst.throughput * params.SNR_sci**2, label='Blaze * Throughput * SNR^2', zorder=10)
#         plt.axhline(y=25, color='red', linestyle='--', linewidth=0.4)
#         plt.xlabel('Wavelength (nm)')
#         plt.ylabel('Photon counts')
#         plt.ylim(0, 400)
#         plt.xlim(542, 544)
#         plt.title(f'Order {inst.order}, Fiber {inst.fiber}, Trace {inst.trace}')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(pwd, './output/test_plot/trace_test1.png'), dpi=300)


    # time the process
    end = time.time()
    print(f"Time cost: {end-start:.2f}s")