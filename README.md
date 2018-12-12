# Inscopix CNMFe Wrapper

This package contains wrappers for running CNMFe code, both on Python using the [CaImAn](https://github.com/flatironinstitute/CaImAn)
package, and on MATLAB using [MATLAB CNMF_E](https://github.com/zhoupc/CNMF_E).

Both versions of the wrapper take .tiff movie files as input, a file containing parameters, and a string
that represents the output file that will be written. The parameter file format is .yaml for the Python
wrapper, and .mat for the MATLAB wrapper.

For detailed information on the MATLAB wrapper, see [run_cnmfe_wrapper.m](matlab/run_cnmfe_wrapper.m). For
information on the Python wrapper, see [runner.py](isx_cnmfe_wrapper/runner.py). To install the Python wrapper,
clone the repository and execute:

    python setup.py install

