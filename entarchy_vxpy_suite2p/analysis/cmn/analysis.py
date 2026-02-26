import numpy as np
import pandas as pd
import tifffile
import alive_progress

import entarchy

# Import schema and types
from ... import *

from . import functions
from . import helper


def create(analysis_path: str, recreate: bool = False, **kwargs):

    from entarchy import backend

    # Delete analysis
    if recreate:
        Suite2PVxPy(analysis_path).delete()

    # Create analysis
    dbhost = kwargs.get('dbhost')
    dbname = kwargs.get('dbname')
    dbuser = kwargs.get('dbuser')
    dbpassword = kwargs.get('dbpassword')

    _backend = entarchy.backend.MySQLBackend(dbhost=dbhost, dbname=dbname, dbuser=dbuser, dbpassword=dbpassword)

    Suite2PVxPy.create(analysis_path, _backend)


# def digest_new_data(analysis_path: str, data_root_path: str, **kwargs):
#
#     # Open and digest
#     analysis = caload.analysis.open_analysis(analysis_path)
#
#     sync_type = kwargs.get('sync_type')
#     sync_signal = kwargs.get('sync_signal')
#     sync_signal_time = kwargs.get('sync_signal_time')
#     frame_avg_num = kwargs.get('frame_avg_num', 1)
#     caload.analysis.digest_data(analysis, digest, data_root_path,
#                                 sync_type=sync_type, sync_signal=sync_signal, sync_signal_time=sync_signal_time,
#                                 frame_avg_num=frame_avg_num)


def run(analysis_path: str):

    ent = Suite2PVxPy(analysis_path)

    # Process recordings
    recordings = ent.get(Recording, 'NOT(EXIST(time_resampled))')

    recordings.map_async(functions.process_recording, sample_rate=10, radial_bin_num=16)

    # Process ROIs
    (ent.get(Roi, 'NOT(EXIST(dff))')
     .map_async(functions.calculate_dff, window_size=120, percentile=10))

    (ent.get(Roi, 'NOT(EXIST(rotation_autocorrelation))')
     .map_async(functions.calculate_autocorrelations))

    (ent.get(Roi, 'NOT(EXIST(signal_dff_mean))')
     .map_async(functions.detect_events_with_derivative))

    (ent.get(Roi, 'NOT(EXIST(bs_radial_bin_etas))')
     .map_async(functions.calculate_reverse_correlations, worker_num=5, bootstrap_num=1000, use_gpu=True))

    (ent.get(Roi).where('NOT(EXIST(has_receptive_field))')
     .map_async(functions.run_2step_ncb_test, bernoulli_alpha=0.05))

    # Select all ROIs that have RF and calculate similarity to egomotion templates
    (ent.get(Roi, '(has_receptive_field == True) AND NOT(EXIST(is_rotation_selective))')
     .map_async(functions.calculate_egomotion_similarities))


def postprocessing(ent: entarchy.Entarchy):

    with (alive_progress.alive_bar(force_tty=True, total=8) as bar):

        cmn_rois = ent.get(Roi, 'has_receptive_field == True')

        # # Calculate indices
        cmn_rois['motion_selectivity_index'] = (
                (cmn_rois['translation_best_similarity'] - cmn_rois['rotation_best_similarity']) /
                (cmn_rois['translation_best_similarity'] + cmn_rois['rotation_best_similarity']))
        bar()

        # Calculate other stuff
        cmn_rois['1_minus_rotation_similarity'] = 1 - cmn_rois['rotation_best_similarity']
        cmn_rois['1_minus_translation_similarity'] = 1 - cmn_rois['translation_best_similarity']
        bar()

        cmn_rois['cluster_significant_num'] = cmn_rois['cluster_significant_indices'].apply(lambda x: len(x))
        bar()

        # Add convencience access to properties
        def _unpack_indices(series: pd.Series):
            _patch_idcs = []
            for _cluster_idx in series['cluster_significant_indices']:
                _patch_idcs.extend(series['cluster_unique_patch_indices'][_cluster_idx])
            return len(_patch_idcs)
            # return _patch_idcs

        patch_indices = cmn_rois[['cluster_significant_indices', 'cluster_unique_patch_indices']].apply(_unpack_indices, axis=1)
        cmn_rois['rf_significant_patch_num'] = patch_indices  # [len(idcs) for idcs in patch_indices]
        bar()

        # Convert patch number to approx estimate of area
        patch_centers = cmn_rois[0].recording['positions']
        patch_num = patch_centers.shape[0]
        cmn_rois['rf_size_sr'] = cmn_rois['rf_significant_patch_num'] * 4 * np.pi / patch_num
        bar()

        # Calculate response type based on mode number
        cmn_rois['roi_response_type'] = cmn_rois['cluster_significant_num'].apply(lambda x: 'simple' if x == 1 else 'complex')
        bar()

        # Calculate egomotion axes for all
        cmn_rois['translation_axis_sph'] = cmn_rois['translation_axis'].apply(lambda v: helper.cart2sph(*v))
        cmn_rois['translation_azimuth'] = cmn_rois[f'translation_axis_sph'].apply(lambda v: v[0])
        cmn_rois['translation_elevation'] = cmn_rois[f'translation_axis_sph'].apply(lambda v: v[1])
        bar()

        cmn_rois['rotation_axis_sph'] = cmn_rois['rotation_axis'].apply(lambda v: helper.cart2sph(*v))
        cmn_rois['rotation_azimuth'] = cmn_rois[f'rotation_axis_sph'].apply(lambda v: v[0])
        cmn_rois['rotation_elevation'] = cmn_rois[f'rotation_axis_sph'].apply(lambda v: v[1])

        bar()


def are_rois_in_region(rois: RoiCollection, region_map: str):

    def coordinates_in(region_data: np.ndarray):

        def _coordinates_in(row: pd.Series):
            return region_data[tuple(row[['ants/x', 'ants/y', 'ants/z']].astype(int).values)] > 0

        return _coordinates_in

    # Load from tiff
    region_data = np.swapaxes(np.moveaxis(tifffile.imread(region_map), 0, 2), 0, 1)

    # Write region flags
    return rois[['ants/x', 'ants/y', 'ants/z']].apply(coordinates_in(region_data), axis=1)


if __name__ == '__main__':
    pass
