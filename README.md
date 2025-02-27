# **An Easy-to-Use Spectra and RV Simulator**

This `Python` code (for 3.10+) is developed to simulate photon-limited RV precision for RV instruments, especially [GTC-CHORUS](https://www.nao.cas.cn/gtc/). However, for confidentiality reasons, the files in `./code/data` have been processed.

Created by Chenyang Ji in 2024, and the documentation was written in early 2025.

## 1. Description of the Files Contained
- `./2023-EPF`: The poster I presented at the [2023 Exoplanets & Planet Formation Workshop](https://epf2023.github.io/) as well as the appendix.
- `./2024-Zhuhai`: The talk I presented at the [2024 China-Spain Bilateral Workshop](https://china-spain2024.casconf.cn/).
- `./code`: The main part of the code.

### 1.1 Contents of `./code`
- `./code/data`: The instrument setup, including Blaze function, CCD layout, and optical throughput profile.
- `./code/input`: The stellar template spectrum you want to simulate with. I have included a solar template and two other spectra.
- `./code/notebook`: The Jupyter notebooks to create all the plots shown in my paper, including one for the solar template and another two for K and M stars, separately.
- `./code/src`: All of the code.

### 1.2 Format of Inputs
- The input format should be the same as shown in the example.
- **CCD Layout**: A `.json` file with `['order']['cal-or-sci']['trace']['wave', 'x', 'y']` where all keys are strings.
  - For `['order']`: e.g., `['86', '87', ..., '100']`
  - For `['cal-or-sci']`: The default setting is `{'cal': '2', 'sci': '1'}`.
  - For `['trace']`: e.g., for a three-slice setting, it would be `['1', '2', '3']`.
- **Optical Throughput Profile**: A `.mat` file with four keys: `['opthobsr2', 'opthobsb2', 'wvr', 'wvb']`, referring to throughput in the red, throughput in the blue, wave range in the red, and wave range in the blue.
  - Ensure the throughput covers the entire wavelength range of the instrument setting; otherwise, extrapolation errors might occur.
- **Template Spectra**: A `.npz` file containing two arrays: `['arr_0', 'arr_1']`. `['arr_0']` stands for wavelength, and `['arr_1']` stands for the normalized flux.
  - Note: This template spectra should have an infinite spectral resolution for statistical stability. For example, we inserted 5 interpolating points between two neighboring points in the $R=500,000$ Kurucz's solar template.

## 2. How to Use It?

### 2.1 Packages Needed
- `NumPy`
- `Matplotlib`
- `time`
- `multiprocessing`
- `os`
- `SciPy`
- `Collections`
- `typing`

### 2.2 Introduction to the Main Scripts
There are four files in `./src`, each with different settings at the beginning (e.g., flags for adding noise or contamination):
- `save_rv_nocont.py`: Photon noise only.
- `save_rv_calsci.py`: Only cal-sci contamination without photon noise.
- `save_rv_scisci.py`: Only sci-sci contamination without photon noise.
- `save_rv_CHORUS.py`: Combined results, including random photon noise, cal-sci, and sci-sci contamination.

Just follow the comments in the code to see how to modify the settings, including resolution ($R$), SNR, and contamination fraction. It is very simple.

### 2.3 Outputs
I have run `save_rv_nocont.py` with the settings in it as an example. A series of outputs have been stored in the `output` folder. For each output `.npz` file, we have six matrices: `['wavelength', 'flux', 'blaze', 'throughput', 'sigma', 'RV']`. Their shapes and descriptions are:
- Wavelength matrix: `(len(order_num), 3, 8000)`, wavelength information for 8000 points in three science traces in all orders;
- Flux matrix: `(len(order_num), 3, 8000)`, photon counts information for 8000 points in three science traces in all orders;
- Blaze matrix: `(len(order_num), 3, 8000)`, normalized Blaze value (from 0 to 1) for 8000 points in three science traces in all orders;
  > This is the Blaze value at the wavelength points;
- Throughput matrix: `(len(order_num), 3, 8000)`, throughput Blaze value for 8000 points in three science traces in all orders;
- Sigma matrix: `(len(order_num), 3)`: used to create PSF in this trace, calculated equation 2 in my paper;
- RV matrix: `(len(order_num), 3)`: RV calculated from CCF for three science traces in all orders.

### 2.4 Create Your First Synthetic Spectrum
After proper modifications, simply run the following command, and you will have your outputs in the `./output` folder:
```sh
python save_rv_nocont.py
