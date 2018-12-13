import os
import yaml
import h5py
import argparse

import numpy as np

from caiman import load_memmap, save_memmap
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.source_extraction.cnmf import CNMF
from caiman.source_extraction.cnmf.params import CNMFParams


def _turn_into_array(val):
    """ Turn val into a numpy array with two elements, if it is not already. """
    if val is not None:
        if hasattr(val, '__iter__'):
            val = np.array(val)
        else:
            val = np.array([val, val])
    return val


def _reshape_A(num_rows, num_cols, A):
    """ Turn the sparse scipy.csc_matrix A into a dense matrix with shape (num_rows, num_cols, num_cells) """

    Adense = np.array(A.todense())
    npx, ncells = Adense.shape
    if npx != num_rows * num_cols:
        raise ValueError('A.shape[0] must be equal to num_rows*num_cols')

    Adense = Adense.reshape([num_cols, num_rows, ncells])
    Ars = np.zeros([num_rows, num_cols, ncells])
    for k in range(ncells):
        Ars[:, :, k] = Adense[:, :, k].T
    del Adense
    Ars[np.isnan(Ars)] = 0
    return Ars.astype('float32')


def _get_memmap_name(file_name, num_frames, num_rows, num_cols):
    """ Return the name of a memmap file created by save_memmap in caiman, based on the name of a tiff file. """

    root_dir, fname = os.path.split(file_name)
    base_name, ext = os.path.splitext(fname)

    byte_order = 'C'
    mm_name = '{}_d1_{}_d2_{}_d3_1_order_{}_frames_{}_.mmap'.format(base_name, num_rows, num_cols, byte_order, num_frames)

    return mm_name


def _export_movie_to_memmap(tiff_file, num_frames, num_rows, num_cols, overwrite=False, dest_dir=None):

    if dest_dir is None:
        # use the first movie's directory as the destination directory
        dest_dir, isxd_movie_name = os.path.split(tiff_file)

    # use the first movie's name as the output memmap file
    mmap_name = _get_memmap_name(tiff_file, num_frames, num_rows, num_cols)

    mmap_file = os.path.join(dest_dir, mmap_name)
    if os.path.exists(mmap_file):
        if overwrite:
            os.remove(mmap_file)
        else:
            return mmap_file

    # write a tiff file, use the name of the first movie as the tiff file base name
    root_dir, fname = os.path.split(tiff_file)
    base_name, ext = os.path.splitext(fname)

    save_memmap([tiff_file], base_name=base_name, order='C')

    return mmap_file


def run_cnmfe(tiff_file, param_file, output_file):
    """ Run the CNMFe algorithm through CaImAn.

    :param tiff_file: A .tiff file containing a calcium imaging movie.
    :param param_file: A .yaml parameter file, containing values for the following parameters:
        num_processes : int
            The number of processes to run in parallel. The more parallel processes, the more memory that is used.
        rf : array-like
            An array [half-width, half-height] that specifies the size of a patch.
        stride : int
            The amount of overlap in pixels between patches.
        K : int
            The maximum number of cells per patch.
        gSiz : int
            The expected diameter of a neuron in pixels.
        gSig : int
            The standard deviation a high pass Gaussian filter applied to the movie prior to seed pixel search, roughly
            equal to the half-size of the neuron in pixels.
        min_pnr : float
            The minimum peak-to-noise ratio that is taken into account when searching for seed pixels.
        min_corr : float
            The minimum pixel correlation that is taken into account when searching for seed pixels.
        min_SNR : float
            Cells with an signal-to-noise (SNR) less than this are rejected.
        rval_thr : float
            Cells with a spatial correlation of greater than this are accepted.
        decay_time : float
            The expected decay time of a calcium event in seconds.
        ssub_B : int
            The spatial downsampling factor used on the background term.
        merge_threshold : float
            Cells that are spatially close with a temporal correlation of greater than merge_threshold are automatically merged.
    :param output_file: The path to a .hdf5 file that will be written to contain the traces, footprints, and deconvolved
        events identified by CNMFe.
    """

    if not os.path.exists(tiff_file):
        raise FileNotFoundError(tiff_file)

    if not os.path.exists(param_file):
        raise FileNotFoundError(param_file)

    with open(param_file, 'r') as f:
        params = yaml.load(f)

    expected_params = ['gSiz', 'gSig', 'K', 'min_corr', 'min_pnr', 'rf', 'stride',
                       'decay_time', 'min_SNR', 'rval_thr', 'merge_threshold',
                       'ssub_B', 'frame_rate',
                       'num_rows', 'num_cols', 'num_frames', 'num_processes']

    for pname in expected_params:
        if pname not in params:
            raise ValueError('Missing parameter {} in file {}'.format(pname, param_file))

    gSiz = params['gSiz']
    gSig = params['gSig']
    K = params['K']
    min_corr = params['min_corr']
    min_pnr = params['min_pnr']
    rf = params['rf']
    stride = params['stride']
    decay_time = params['decay_time']
    min_SNR = params['min_SNR']
    rval_thr = params['rval_thr']
    merge_threshold = params['merge_threshold']
    ssub_B = params['ssub_B']
    frame_rate = params['frame_rate']
    num_rows = params['num_rows']
    num_cols = params['num_cols']
    num_frames = params['num_frames']
    num_processes = params['num_processes']

    # write memmapped file
    print('Exporting .isxd to memmap file...')
    mmap_file = _export_movie_to_memmap(tiff_file, num_frames, num_rows, num_cols, overwrite=False)
    print('Wrote .mmap file to: {}'.format(mmap_file))

    # open memmapped file
    Yr, dims, T = load_memmap(mmap_file)
    Y = Yr.T.reshape((T,) + dims, order='F')

    # grab parallel IPython handle
    dview = None
    if num_processes > 1:
        import ipyparallel as ipp
        c = ipp.Client()
        dview = c[:]
        print('Running using parallel IPython, # clusters = {}'.format(len(c.ids)))
        num_processes = len(c.ids)

    # initialize CNMFE parameter object and set user params
    cnmfe_params = CNMFParams()

    if gSiz is None:
        raise ValueError('You must set gSiz to an integer, ideally roughly equal to the expected half-cell width.')
    gSiz = _turn_into_array(gSiz)

    if gSig is None:
        raise ValueError('You must set gSig to a non-zero integer. The default value is 5.')
    gSig = _turn_into_array(gSig)

    cnmfe_params.set('preprocess', {'p':1})

    cnmfe_params.set('init', {'K':K, 'min_corr':min_corr, 'min_pnr':min_pnr,
                              'gSiz':gSiz, 'gSig':gSig})

    if rf is None:
        cnmfe_params.set('patch', {'rf':None, 'stride':1})
    else:
        cnmfe_params.set('patch', {'rf':np.array(rf), 'stride':stride})

    cnmfe_params.set('data', {'decay_time':decay_time})

    cnmfe_params.set('quality', {'min_SNR':min_SNR, 'rval_thr':rval_thr})

    cnmfe_params.set('merging', {'merge_thr':merge_threshold})

    # set parameters that force CNMF into one-photon mode with no temporal or spatial downsampling,
    # except for the background term
    cnmfe_params.set('init', {'center_psf':True, 'method_init':'corr_pnr', 'normalize_init':False, 'nb':-1,
                              'ssub_B':ssub_B, 'tsub':1, 'ssub':1})
    cnmfe_params.set('patch', {'only_init':True, 'low_rank_background':None, 'nb_patch':-1,
                               'p_tsub':1, 'p_ssub':1})
    cnmfe_params.set('spatial', {'nb':-1, 'update_background_components':False})
    cnmfe_params.set('temporal', {'nb':-1, 'p':1})

    # construct and run CNMFE
    print('Running CNMFe...')
    cnmfe = CNMF(num_processes, dview=dview, params=cnmfe_params)
    cnmfe.fit(Y)

    # run auto accept/reject
    print('Estimating component quality...')
    idx_components, idx_components_bad, comp_SNR, r_values, pred_CNN = estimate_components_quality_auto(
        Y, cnmfe.estimates.A, cnmfe.estimates.C, cnmfe.estimates.b, cnmfe.estimates.f, cnmfe.estimates.YrA,
        frame_rate, cnmfe_params.get('data', 'decay_time'), cnmfe_params.get('init', 'gSiz'), cnmfe.dims, dview=None,
        min_SNR=cnmfe_params.get('quality', 'min_SNR'), use_cnn=False)

    save_cnmfe(cnmfe, output_file, good_idx=idx_components)


def save_cnmfe(cnmfe, output_file, good_idx=None):
    """ Save the essential components of a CNMF object to an hdf5 file. """

    if not isinstance(cnmfe, CNMF):
        raise ValueError('cnmfe input must be type CNMF')

    if good_idx is None:
        good_idx = list(range(cnmfe.estimates.A.shape[-1]))

    # retrieve estimates from the CNMF object and reshape/retype them as necessary
    A = _reshape_A(cnmfe.dims[0], cnmfe.dims[1], cnmfe.estimates.A)
    A = A[:, :, good_idx]

    C = cnmfe.estimates.C.astype('float32')
    C = C[good_idx, :]

    S = cnmfe.estimates.S.astype('float32')
    S = S[good_idx, :]

    with h5py.File(output_file, 'w') as hf:
        hf['A'] = A
        hf['C'] = C
        hf['S'] = S


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='Run CaImAn CNMFe')
    _parser.add_argument('--input_file', type=str, required=True, help='The full path to a .tiff movie.')
    _parser.add_argument('--params_file', type=str, required=True, help='The full path to a .yaml file containing parameters.')
    _parser.add_argument('--output_file', type=str, required=True, help='The path to dump an .hdf5 file to.')

    _args = _parser.parse_args()

    run_cnmfe(_args.input_file.strip('"'), _args.params_file.strip('"'), _args.output_file.strip('"'))
