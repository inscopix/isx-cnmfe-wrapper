function run_cnmfe_wrapper(tiff_file, params_file, output_file)
% Run CNMFe on an .isxd movie file. This code is adapted from the CNMF_E demo script available at
% https://github.com/zhoupc/CNMF_E/blob/master/demos/demo_batch_1p.m.
%
% Arguments
% ---------
% tiff_file: The input movie file
%
% params_file: A .mat file containing a structure named "params", that
% contains the parameters used to run CNMFe.
%
% output_file: The path to an output .hdf5 file that contains CNMFe
% footprints, traces, and deconvolved spikes.
%

    params_to_check = {'K', 'min_corr', 'min_pnr', 'patch_dims', 'gSiz', ...
                       'gSig', 'max_tau', 'memory_size_to_use', 'memory_size_per_patch', ...
                       'merge_threshold', 'frame_rate'};
                   
    % load parameters and check them
    pstruct = load(params_file);
    user_params = pstruct.user_params
    for k = 1:length(params_to_check)       
        if ~isfield(user_params, params_to_check{k})
            error('Missing parameter %s', params_to_check{k});
        end        
    end
    
    %% Set free parameters
    Fs = user_params.frame_rate;
    K = user_params.K;                      % maximum number of neurons per patch. when K=[], take as many as possible.
    min_corr = user_params.min_corr;        % minimum local correlation for a seeding pixel
    min_pnr = user_params.min_pnr;          % minimum peak-to-noise ratio for a seeding pixel
    patch_dims = user_params.patch_dims;    % dimensions of patches
    gSiz = user_params.gSiz;                % pixel, neuron diameter
    maxTau = user_params.max_tau;           % maximum decay time in seconds
    gSig = user_params.gSig;                % pixel, gaussian width of a gaussian kernel for filtering the data. 0 means no filtering
    
    min_corr_res = 0.7;                           % minimum correlation in residual image
    min_pnr_res = max(2, round(0.66 * min_pnr));  % minimum PNR in residual image

    %% choose data
    neuron = Sources2D();
    movie_file_name = get_fullname(tiff_file);          % this demo data is very small, here we just use it as an example
    movie_file_name = neuron.select_data(movie_file_name);  %if nam is [], then select data interactively

    %% parameters
    % -------------------------    COMPUTATION    -------------------------  %
    pars_envs = struct('memory_size_to_use', user_params.memory_size_to_use, ...   % GB, memory space you allow to use in MATLAB
        'memory_size_per_patch', 0.6, ...   % GB, space for loading data within one patch
        'patch_dims', patch_dims);  %GB, patch size

    % -------------------------      SPATIAL      -------------------------  %
    ssub = 1;           % spatial downsampling factor
    with_dendrites = true;   % with dendrites or not
    if with_dendrites
        % determine the search locations by dilating the current neuron shapes
        updateA_search_method = 'dilate';
        updateA_bSiz = 5;
        updateA_dist = neuron.options.dist;
    else
        % determine the search locations by selecting a round area
        updateA_search_method = 'ellipse'; %#ok<UNRCH>
        updateA_dist = 5;
        updateA_bSiz = neuron.options.dist;
    end
    spatial_constraints = struct('connected', true, 'circular', false);  % you can include following constraints: 'circular'
    spatial_algorithm = 'hals_thresh';

    % -------------------------      TEMPORAL     -------------------------  %
    tsub = 1;           % temporal downsampling factor
    deconv_options = struct('type', 'ar1', ... % model of the calcium traces. {'ar1', 'ar2'}
        'method', 'foopsi', ... % method for running deconvolution {'foopsi', 'constrained', 'thresholded'}
        'smin', -5, ...         % minimum spike size. When the value is negative, the actual threshold is abs(smin)*noise level
        'optimize_pars', true, ...  % optimize AR coefficients
        'optimize_b', true, ...% optimize the baseline);
        'max_tau', round(maxTau * Fs));    % maximum decay time (unit: frame);

    nk = 3;             % detrending the slow fluctuation. usually 1 is fine (no detrending)
    % when changed, try some integers smaller than total_frame/(Fs*30)
    detrend_method = 'spline';  % compute the local minimum as an estimation of trend.

    % -------------------------     BACKGROUND    -------------------------  %
    bg_model = 'ring';  % model of the background {'ring', 'svd'(default), 'nmf'}
    nb = 1;             % number of background sources for each patch (only be used in SVD and NMF model)
    ring_radius = round(1.5 * gSiz);  % when the ring model used, it is the radius of the ring used in the background model.
    %otherwise, it's just the width of the overlapping area
    num_neighbors = []; % number of neighbors for each neuron

    % -------------------------      MERGING      -------------------------  %
    show_merge = false;  % if true, manually verify the merging step
    merge_thr = user_params.merge_threshold;     % thresholds for merging neurons; [spatial overlap ratio, temporal correlation of calcium traces, spike correlation]
    method_dist = 'max';   % method for computing neuron distances {'mean', 'max'}
    dmin = 5;       % minimum distances between two neurons. it is used together with merge_thr
    dmin_only = 2;  % merge neurons if their distances are smaller than dmin_only.
    merge_thr_spatial = [0.8, 0.1, -inf];  % merge components with highly correlated spatial shapes (corr=0.8) and small temporal correlations (corr=0.1)

    % -------------------------  INITIALIZATION   -------------------------  %
    min_pixel = max(4, round(gSiz*0.1));      % minimum number of nonzero pixels for each neuron
    bd = 0;             % number of rows/columns to be ignored in the boundary (mainly for motion corrected data)
    frame_range = [];   % when [], uses all frames
    save_initialization = false;    % save the initialization procedure as a video.
    use_parallel = true;    % use parallel computation for parallel computing
    show_init = false;   % show initialization results
    choose_params = false; % manually choose parameters
    center_psf = true;  % set the value as true when the background fluctuation is large (usually 1p data)
    % set the value as false when the background fluctuation is small (2p)

    % -------------------------  Residual   -------------------------  %
    seed_method_res = 'auto';  % method for initializing neurons from the residual
    update_sn = true;

    % ----------------------  WITH MANUAL INTERVENTION  --------------------  %
    with_manual_intervention = false;

    % -------------------------  FINAL RESULTS   -------------------------  %
    save_demixed = false;    % save the demixed file or not
    kt = 3;                 % frame intervals

    % -------------------------    UPDATE ALL    -------------------------  %
    neuron.updateParams('gSig', gSig, ...       % -------- spatial --------
        'gSiz', gSiz, ...
        'ring_radius', ring_radius, ...
        'ssub', ssub, ...
        'search_method', updateA_search_method, ...
        'bSiz', updateA_bSiz, ...
        'dist', updateA_bSiz, ...
        'spatial_constraints', spatial_constraints, ...
        'spatial_algorithm', spatial_algorithm, ...
        'tsub', tsub, ...                       % -------- temporal --------
        'deconv_options', deconv_options, ...
        'nk', nk, ...
        'detrend_method', detrend_method, ...
        'background_model', bg_model, ...       % -------- background --------
        'nb', nb, ...
        'ring_radius', ring_radius, ...
        'num_neighbors', num_neighbors, ...
        'merge_thr', merge_thr, ...             % -------- merging ---------
        'dmin', dmin, ...
        'method_dist', method_dist, ...
        'min_corr', min_corr, ...               % ----- initialization -----
        'min_pnr', min_pnr, ...
        'min_pixel', min_pixel, ...
        'bd', bd, ...
        'center_psf', center_psf);
    neuron.Fs = Fs;

    %% distribute data and be ready to run source extraction
    neuron.getReady(pars_envs);

    %% initialize neurons from the video data within a selected temporal range
    if choose_params
        % change parameters for optimized initialization
        [gSig, gSiz, ring_radius, min_corr, min_pnr] = neuron.set_parameters();
    end

    [center, Cn, PNR] = neuron.initComponents_parallel(K, frame_range, save_initialization, use_parallel);
    neuron.compactSpatial();

    %% estimate the background components
    neuron.update_background_parallel(use_parallel);
    neuron_init = neuron.copy();

    %%  merge neurons and update spatial/temporal components
    neuron.merge_neurons_dist_corr(show_merge);
    neuron.merge_high_corr(show_merge, merge_thr_spatial);

    %% update spatial components

    %% pick neurons from the residual
    [center_res, Cn_res, PNR_res] = neuron.initComponents_residual_parallel([], save_initialization, use_parallel, min_corr_res, min_pnr_res, seed_method_res);
    if show_init
        axes(ax_init);
        plot(center_res(:, 2), center_res(:, 1), '.g', 'markersize', 10);
    end
    neuron_init_res = neuron.copy();

    %% udpate spatial&temporal components, delete false positives and merge neurons
    % update spatial
    if update_sn
        neuron.update_spatial_parallel(use_parallel, true);
        udpate_sn = false;
    else
        neuron.update_spatial_parallel(use_parallel);
    end
    % merge neurons based on correlations
    neuron.merge_high_corr(show_merge, merge_thr_spatial);

    for m=1:2
        % update temporal
        neuron.update_temporal_parallel(use_parallel);

        % delete bad neurons
        neuron.remove_false_positives();

        % merge neurons based on temporal correlation + distances
        neuron.merge_neurons_dist_corr(show_merge);
    end

    %% run the whole procedure for a second time
    neuron.options.spatial_algorithm = 'nnls';

    %% run more iterations
    neuron.update_background_parallel(use_parallel);
    neuron.update_spatial_parallel(use_parallel);
    neuron.update_temporal_parallel(use_parallel);

    K = size(neuron.A,2);
    tags = neuron.tag_neurons_parallel();  % find neurons with fewer nonzero pixels than min_pixel and silent calcium transients
    neuron.remove_false_positives();
    neuron.merge_neurons_dist_corr(show_merge);
    neuron.merge_high_corr(show_merge, merge_thr_spatial);

    if K~=size(neuron.A,2)
        neuron.update_spatial_parallel(use_parallel);
        neuron.update_temporal_parallel(use_parallel);
        neuron.remove_false_positives();
    end

    neuron.orderROIs('snr');
    
    % save the footprints, traces, and events to an hdf5 file
    Asz = size(neuron.A);
    Csz = size(neuron.C);
    Ssz = size(neuron.S);
    
    if exist(output_file, 'file')
        delete(output_file);
    end
    
    h5create(output_file, '/A', Asz);
    h5write(output_file, '/A', neuron.A);
    
    h5create(output_file, '/C', Csz);
    h5write(output_file, '/C', neuron.C);
    
    h5create(output_file, '/S', Ssz);
    h5write(output_file, '/S', neuron.S);
    
    clear neuron;
    
end