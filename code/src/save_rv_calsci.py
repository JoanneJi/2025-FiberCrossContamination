"""

This file is used to save the RV calculation results as npz files,
after looping through all traces in one observation with SNR set as 100 and 800, and cal-sci frac ranging from 1e-5 to 1e-1.
Parallel processing for SNR and cal-sci frac.
Only cal-sci cross-contamination is added here.
No Poisson noise and read noise here.

------- Functions ---------
+ save synthetic file, each containing:
    - wavelength: [order, trace, 8000]
    - flux: [order, trace, 8000]
    - blaze: [order, trace, 8000]
    - throughput: [order, trace, 8000]
    - sigma: [order, trace]
    - RV: [order, trace]
+ get RV precision
    - a flag for whether to extract precision for each trace separately
+ a conclusion of inputs, including:
    + params
        - nspec: total number of spectra you want to generate
        - temp_file
        - R (parallel input)
        - read_noise
        - trace_height
        - SNR_cal
        - SNR_sci (parallel input)
        - calsci_frac
        - scisci_frac
    + params_inst
        - flag
        - band
        - G
        - tb
        - t
        - gamma
        - finesse
        - cavity
    + params_flags
        - R_flag
        - Poisson_noise_flag
        - read_noise_flag
        - weight_flag
---------------------------

Created by Chenyang Ji in 2024.11.28.

"""

import os
import numpy as np
import instrument
import trace
from collections import namedtuple
import time
import multiprocessing


# -------------- directory path of CHORUS folder --------------
pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# >>> path of CHORUS folder


# -------------- other inputs --------------
Arg = namedtuple('Arg', ['nspec', 'temp_file', 'R', 'read_noise', 'trace_height', 'SNR_cal', 'SNR_sci', 'calsci_frac', 'scisci_frac'])
Arg_inst = namedtuple('Arg_inst', ['flag', 'band', 'G', 'tb', 't', 'gamma', 'finesse', 'cavity'])
Arg_flag = namedtuple('Arg_flag', ['R_flag', 'Poisson_noise_flag', 'read_noise_flag', 'FP_Poisson_noise_flag', 'FP_read_noise_flag', 'weight_flag'])

# Define ranges
calsci_frac_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SNR_values = [100]

# Prepare inputs for multiprocessing
tasks = []
for calsci_frac in calsci_frac_values:
    for SNR in SNR_values:
        params = Arg(
            # total number of spectra you want to generate
            nspec=1,
            # instrument info based on 3-slice setting
            temp_file='sun.npz',
            R=120000,
            read_noise=2.5,
            trace_height=17,
            # observation info
            SNR_cal=SNR,
            SNR_sci=SNR,
            # contamination info, based on order separation of CHORUS
            # no cross-contamination
            calsci_frac=calsci_frac,
            scisci_frac=0
        )
        params_inst_values = Arg_inst(
            # CCD info, change here if you want to have different settings
            flag=3,
            band=['red', 'blue'],
            # grating info, this should be matched with slicing info above
            G=41.59,  # 31.6 for 2-slice, 41.59 for 3-slice
            tb=np.deg2rad(75.5),
            t=0.0,
            gamma=np.deg2rad(1.0),
            # FP interferometer info
            finesse=18,
            cavity=7.8e6  # nm, 0.0078 m
        )
        params_flags_values = Arg_flag(
            # turn on/off decreasing the spectral resolution to the instrumental setting for testing the code
            R_flag=True,
            # turn on/off adding noise for testing the code
            Poisson_noise_flag=False,
            read_noise_flag=False,
            # turn on/off noise flag of FP lines
            FP_Poisson_noise_flag=False,
            FP_read_noise_flag=False,
            # weight flag for extracting RV
            weight_flag=True
        )
        tasks.append((params, params_inst_values, params_flags_values))


# -------------- save synthetic file --------------
def save_file(params: namedtuple, params_inst: namedtuple, params_flags: namedtuple) -> None:
    """
    Generate nspec synthetic files with different SNR.

    Args:
        params: namedtuple, parameters of the observation
        params_inst: namedtuple, parameters of the instrument
        params_flags: namedtuple, flags of the parameters

    Returns:
        None
    """
    # initialize the template files
    temp_wave, temp_flux = trace.get_template(params)

    for ispec in range(params.nspec):
        for iband in params_inst.band:
            syn_wave_mat, syn_flux_mat, blaze_mat, throughput_mat, psf_mat, sigma_mat = trace.contamination(temp_wave, temp_flux,
                                                                                                            iband,
                                                                                                            params_inst.flag,
                                                                                                            instrument.params,
                                                                                                            params,
                                                                                                            params_flags)
            rv_mat = trace.extract_RVmat(temp_wave, temp_flux,  # template information
                                        syn_wave_mat, syn_flux_mat,  # synthetic spectrum
                                        blaze_mat, throughput_mat, psf_mat,  # instrument setting
                                        params, params_flags)  # params
            
            # save the results
            # example of filename: ./output/synfile_calsci/snr100/1e-5_0_red.npz
            filename = f"{params.calsci_frac}_{ispec}_{iband}.npz"
            filepath = os.path.join(pwd, f'./output/synfile_calsci/snr{params.SNR_sci}/{filename}')
            # create the folder if not exist
            folder_path = os.path.dirname(filepath)
            os.makedirs(folder_path, exist_ok=True)

            # save the synthetic file
            np.savez(filepath,
                    wavelength=syn_wave_mat, flux=syn_flux_mat,  # synthetic spectrum
                    blaze=blaze_mat, throughput=throughput_mat, sigma=sigma_mat,  # instrument setting
                    RV=rv_mat)  # RV calculation results

            print(filename)

    return


# -------------- Multiprocessing Wrapper --------------
def process_combination(args):
    params_values, params_inst, params_flags = args
    save_file(params_values, params_inst, params_flags)


if __name__ == '__main__':
    start = time.time()

    # Run multiprocessing
    num_processes = len(calsci_frac_values) * len(SNR_values)  # Number of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_combination, tasks)

    end = time.time()

    print('Time:', end - start)

