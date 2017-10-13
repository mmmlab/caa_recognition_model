# CAA Word Recognition Model

Code implementing the continuously aggregated accumulation (CAA) model of word
recognition.

## Overview

The main modules are `single_process.py` and `dual_process.py`, which define the
functions required to compute the single-process and dual-process versions of 
the CAA model, respectively.

Summary descriptions of the major functions will be added to this document at a
future date. Note, however, that all major functions are well documented and 
include well-formatted docstrings, so that they can be queried using Python's
built-in `help` function. An overview of the model itself appears in 
[NotesOnD2BModel.md](NotesOnD2BModel.md).

- Experimental data (in various formats) are located in the `data` folder.
- Various plots (of data, model predictions, and model schematics) are located
in the `plots` folder.
- Import directives in the various modules assume that the repository has been
cloned into a folder named `caa_model`, which you can do using a shell command
like:

```
git clone https://github.com/mmmlab/caa_recognition_model.git ./caa_model
```

## Requirements

The code makes use of the 'Pylab' stack of scientific libraries for Python. In
particular, the following scientific packages are required:

- `numpy`
- `scipy`
- `matplotlib`

In addition, `pyyaml` is required (for reading in data), and `pyfftw` is 
(optionally) required if you wish to use the fftw library to speed up the
computation of convolutions.