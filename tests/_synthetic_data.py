"""Builds a minimal but realistic vxpy + suite2p dataset on disk for ingest tests.

Layout produced::

    <root>/<animal_id>/
        animal_metadata.yaml
        zstack_1.tif
        <recording_id>/
            recording_metadata.yaml
            Io.hdf5              galvo mirror trace, record group ids
            Display.hdf5         stimulus phases and CMN base data
            suite2p/
                plane0/ ops.npy F.npy spks.npy stat.npy iscell.npy
                plane1/ ...
"""
import os

import h5py
import numpy as np
import tifffile
import yaml

MIRROR_FREQUENCY = 10.0     # Hz, one imaging frame per half cycle
SAMPLE_RATE = 1000.0        # Hz of the analog io recording
DURATION = 20.0             # s

PHASE_WINDOWS = {0: (2.0, 6.0), 1: (8.0, 12.0)}


def mirror_trace():
    """Triangle wave starting on a falling flank, so a trough precedes the first peak."""
    from scipy.signal import sawtooth

    times = np.arange(0, DURATION, 1 / SAMPLE_RATE)
    position = sawtooth(2 * np.pi * MIRROR_FREQUENCY * times - np.pi / 2, width=0.5)
    return position, times


def expected_frame_times():
    """Frame times the ingest is expected to reconstruct from the mirror trace."""
    from entarchy_vxpy_suite2p.schema import ca_frame_times_from_y_mirror

    position, times = mirror_trace()
    return ca_frame_times_from_y_mirror(position, times)[1]


def record_group_trace(times):
    """Stimulus phase id per analog sample (-1 between phases)."""
    ids = np.full(times.shape, -1, dtype=np.int64)
    for phase_index, (start, end) in PHASE_WINDOWS.items():
        ids[(times >= start) & (times < end)] = phase_index
    return ids


def write_io_file(path):
    position, times = mirror_trace()

    with h5py.File(os.path.join(path, 'Io.hdf5'), 'w') as f:
        f.create_dataset('ai_y_mirror_in', data=position[:, None])
        f.create_dataset('ai_y_mirror_in_time', data=times[:, None])
        f.create_dataset('__record_group_id', data=record_group_trace(times)[:, None])
        f.create_dataset('__time', data=times[:, None])


def write_display_file(path, patch_num=12, cmn_frames=30):
    """Stimulus log with two phases plus the CMN base data they reference."""
    rng = np.random.default_rng(4)

    with h5py.File(os.path.join(path, 'Display.hdf5'), 'w') as f:
        f.attrs['__display_fps'] = 60.0
        f.attrs['__protocol_name'] = 'SyntheticProtocol'

        # Datasets at root level land directly on the Recording entity
        f.create_dataset('__time', data=np.arange(cmn_frames, dtype=float)[:, None])

        for phase_index, (start, end) in PHASE_WINDOWS.items():
            group = f.create_group(f'phase{phase_index}')
            group.attrs['__visual_name'] = 'CMN' if phase_index == 0 else 'TranslationGrating'
            group.attrs['__start_time'] = start
            group.attrs['__target_duration'] = end - start
            group.create_dataset('frame_index', data=np.arange(cmn_frames)[:, None])
            group.create_dataset('__time',
                                 data=np.linspace(start, end, cmn_frames)[:, None])

        # A non-phase group becomes namespaced attributes on the Recording
        cmn = f.create_group('CMN')
        cmn.attrs['seed'] = 42
        centers = rng.normal(size=(patch_num, 3))
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        cmn.create_dataset('centers_0', data=centers)
        cmn.create_dataset('motion_vectors_0', data=rng.normal(size=(cmn_frames, patch_num, 3)))


def write_suite2p_plane(path, plane_index, roi_num, frame_num):
    plane_path = os.path.join(path, 'suite2p', f'plane{plane_index}')
    os.makedirs(plane_path, exist_ok=True)
    rng = np.random.default_rng(100 + plane_index)

    ops = {
        'fs': 5.0,
        'nframes': frame_num,
        'meanImg': rng.random((8, 8)),
        'nested': {'do_registration': 1},
    }
    np.save(os.path.join(plane_path, 'ops.npy'), ops)

    fluorescence = rng.random((roi_num, frame_num)) * 100 + 500
    np.save(os.path.join(plane_path, 'F.npy'), fluorescence)
    np.save(os.path.join(plane_path, 'spks.npy'), rng.random((roi_num, frame_num)))

    stats = np.array([{'npix': 40 + i, 'skew': float(i) / 10, 'med': [i, i * 2]}
                      for i in range(roi_num)], dtype=object)
    np.save(os.path.join(plane_path, 'stat.npy'), stats, allow_pickle=True)

    iscell = np.zeros((roi_num, 2))
    iscell[:, 0] = np.arange(roi_num) % 2
    np.save(os.path.join(plane_path, 'iscell.npy'), iscell)

    return fluorescence


def build_dataset(root, animal_id='animal_01', recording_id='rec_01',
                  roi_num=4, plane_num=2, frames_per_plane=None, with_zstack=True):
    """Create the folder tree and return a description of what was written."""
    animal_path = os.path.join(root, animal_id)
    recording_path = os.path.join(animal_path, recording_id)
    os.makedirs(recording_path, exist_ok=True)

    with open(os.path.join(animal_path, 'animal_metadata.yaml'), 'w') as f:
        yaml.safe_dump({'strain': 'wildtype', 'age_dpf': 6, 'nested': {'dish': 3}}, f)

    zstack = None
    if with_zstack:
        zstack = np.random.default_rng(7).integers(0, 255, (3, 8, 8), dtype=np.uint8)
        tifffile.imwrite(os.path.join(animal_path, 'zstack_1.tif'), zstack,
                         photometric='minisblack')

    with open(os.path.join(recording_path, 'recording_metadata.yaml'), 'w') as f:
        yaml.safe_dump({'repeat_num': 2, 'experimenter': 'tester'}, f)

    write_io_file(recording_path)
    write_display_file(recording_path)

    # By default give every plane the same frame count, as suite2p does for a
    # volumetric acquisition. The ingest truncates its reconstructed frame times
    # to match.
    frame_times = expected_frame_times()
    if frames_per_plane is None:
        shortest = len(frame_times[plane_num - 1::plane_num])
        frames_per_plane = [shortest] * plane_num

    fluorescence = {}
    for plane_index in range(plane_num):
        fluorescence[plane_index] = write_suite2p_plane(
            recording_path, plane_index, roi_num, frames_per_plane[plane_index])

    return {
        'root': root,
        'animal_id': animal_id,
        'recording_id': recording_id,
        'animal_path': animal_path,
        'recording_path': recording_path,
        'roi_num': roi_num,
        'plane_num': plane_num,
        'frames_per_plane': frames_per_plane,
        'frame_times': frame_times,
        'fluorescence': fluorescence,
        'zstack': zstack,
        'phase_indices': sorted(PHASE_WINDOWS),
    }
