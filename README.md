# Generative Fourier-based Auto-Encoders

This is a PyTorch implementation for the framework presented in the following paper:
**Generative Fourier-based Auto-Encoders: Preliminary Results paper**

## Requirements
* `torch==1.5.0`
* `torchaudio==0.5.0`
* `scikit-learn==0.22.2.post1`
* `numpy==1.12.0`

Use `requirements.txt` to install all the dependencies. Tested only with `Python 3.6`


## Data
Download the dataset used from [here](https://doi.org/10.5281/zenodo.3833835) and place it in data.

## Run the experiment
* check if all the path are corrects in the settings of the various scripts
* to test the likelihood of the models run `likelihood/run_likelihood.py`
* to obtain the Fig.2a use `likelihood/print_likelihood.py`
* with the number of components obtain from the previous script, run `ppca/experiment_ppca.py`
* with the data saved from the previous script, run `ppca/plot_generation.py` to obtain Fig.2b