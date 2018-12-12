# Inscopix CNMFe Wrapper

This package contains wrappers for running CNMFe code, both on Python using the [CaImAn](https://github.com/flatironinstitute/CaImAn)
package, and on MATLAB using []MATLAB CNMF_E](https://github.com/zhoupc/CNMF_E).

Both version of the wrapper take .tiff movie files as input, a file containing parameters, and a string
that represents the output file that will be written. The parameter file format is .yaml for the Python
wrapper, and .mat for the MATLAB wrapper.
