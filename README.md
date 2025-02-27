# **An Easy-to-Use Spectra and RV simulator**
This $\texttt{Python}$ code (for 3.10+) is developed to simulate photon-limited RV precision for RV instruments, especially [GTC-CHORUS](https://www.nao.cas.cn/gtc/). However, for confidentiality reasons the files in $\texttt{./code/data}$ has been processed. 

Created by Chenyang Ji in 2024, and the documentation was written in early 2025.

## 1. Description for the files contained.
- $\texttt{./2023-EPF}$: The poster I took at [2023 Exoplanets & Planet Formation Workshop](https://epf2023.github.io/) as well as the appendix.
- $\texttt{./2024-Zhuhai}$: The talk I presented at [2024 China-Spain Bilateral Workshop](https://china-spain2024.casconf.cn/).
- $\texttt{./code}$: the main part of the code

### 1.1 Contents of $\texttt{./code}$
- $\texttt{./code/data}$: The instrument setup, including Blaze function, CCD layout, and optical throughput profile.
- $\texttt{./code/input}$: The stellar template spectrum you want to simulate with. I have included a solar template and two other spectra.
- $\texttt{./code/notebook}$: The jupyter notebooks to create all the plots shown in my paper, including one for the solar template and another two for K and M, separately.
- $\texttt{./code/src}$: All of the code.
### 1.2 Format of inputs
- The input format should be the same with what I have showed as an example.
- **CCD layout**: a $\texttt{.json}$ file with $\texttt{['order']['cal-or-sci']['trace']['wave', 'x', 'y']}$ and the keys are all string.
  - For $\texttt{['order']}$: e.g. $\texttt{['86', '87', ..., '100']}$
  - For $\texttt{['cal-or-sci']}$, the default setting is $\texttt{'cal'}='2'$ and $\texttt{'sci'}='1'$.
  - For $\texttt{['trace']}$: e.g. for a three-slice setting it would be $\texttt{['1', '2', '3']}$.
- **Optical throughput profile**: a $\texttt{.mat}$ file with four keys: $\texttt{['opthobsr2', 'opthobsb2', 'wvr', 'wvb']}$, referring to throughput in the red, throughput in the blue, wave range in the red, waverange in the blue.
  - Please make sure you have the throughput covering the whole wavelength range of the instrument setting, otherwise there might be some error during extrapolation.
- **Template spectra**: a $\texttt{.npz}$ file containing two arrays: $\texttt{['arr\_0', 'arr\_1']}$. $\texttt{['arr\_0']}$ stands for wavelength, and $\texttt{['arr\_1']}$ stands for the normalized flux.
  - Note that this template spectra should have an infinite spectral resolution for statistical stability. Instantly, we inserted 5 interpolating points between two neighboring points in the $R=500,000$ Kurucz's solar template.

## 2. How to use it?

### 2.1 Packages needed
- $\texttt{NumPy}$
- $\texttt{Matplotlib}$
- $\texttt{time}$
- $\texttt{multiprocessing}$
- $\texttt{os}$
- $\texttt{SciPy}$
- $\texttt{Collections}$
- $\texttt{typing}$

### 2.2 Introduction to my main scripts
- There are four files in $\texttt{./src}$, with only different settings at the beginning, e.g. flag to add noise or contamination.
  - $\texttt{save\_rv\_nocont.py}$: photon noise only
  - $\texttt{save\_rv\_calsci.py}$: only cal-sci contamination without photon noise
  - $\texttt{save\_rv\_scisci.py}$: only sci-sci contamination without photon noise
  - $\texttt{save\_rv\_CHORUS.py}$: with combined results, including random photon noise, cal-sci and sci-sci contamination
- Just follow the comments in my code to see how to modify the settings, including resolution ($R$), SNR, and contamination fraction. It is very simple.

### 2.3 Outputs
- 

### 2.3 Create your first synthetic spectrum
- After proper modifications, just run this and you will have your outputs in the $\texttt{./output}$ folder.
```sh
python save_rv_nocont.py
```