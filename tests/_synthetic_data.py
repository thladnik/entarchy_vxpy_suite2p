"""Builds a minimal but realistic vxpy + suite2p dataset on disk for ingest tests.

Layout produced::

    <root>/<animal_id>/
        animal_metadata.yaml
        zstack_1.tif
        <recording_id>/
            recording_metadata.yaml
            Io.hdf5              galvo mirror trace, record group ids
            Display.hdf5         stimulus phases and CMN base data
            Camera.hdf5          behaviour camera frame times and tail tracking
            fish_embedded_frame.avi   the frames those times belong to

The number of stimulation phases is configurable: `build_dataset(phase_num=6)`,
or `phase_windows={0: (2.0, 5.0), ...}` to place them by hand.
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

# Start and end time of each stimulation phase, keyed by phase index. The io file
#  and the display file have to agree on these: the ingest derives a phase's
#  calcium frame window by looking for its index in the record group trace, so a
#  phase present in only one of the two files gets no frames and the ingest fails
#  on an empty index array.
PHASE_WINDOWS = {0: (2.0, 6.0), 1: (8.0, 12.0)}

# Cycled over the phases, so a dataset with more phases keeps varied stimuli
VISUAL_NAMES = ['CMN', 'TranslationGrating']


def make_phase_windows(count, start=2.0, end=DURATION - 2.0, duty=0.7):
    """Evenly spaced phase windows with a gap between them.

    `duty` is the fraction of each slot the phase occupies; the remainder is the
    inter-phase interval, which the record group trace marks as -1. Windows must
    stay wide enough to contain several imaging frames, so counts beyond about
    twenty leave phases with too few frames for the ingest to place them.
    """
    if count < 1:
        raise ValueError('need at least one phase')

    span = (end - start) / count

    return {index: (start + index * span, start + index * span + span * duty)
            for index in range(count)}


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


def record_group_trace(times, phase_windows=None):
    """Stimulus phase id per analog sample (-1 between phases)."""
    phase_windows = PHASE_WINDOWS if phase_windows is None else phase_windows

    ids = np.full(times.shape, -1, dtype=np.int64)
    for phase_index, (start, end) in phase_windows.items():
        ids[(times >= start) & (times < end)] = phase_index

    return ids


def write_io_file(path, phase_windows=None):
    position, times = mirror_trace()

    with h5py.File(os.path.join(path, 'Io.hdf5'), 'w') as f:
        f.create_dataset('ai_y_mirror_in', data=position[:, None])
        f.create_dataset('ai_y_mirror_in_time', data=times[:, None])
        f.create_dataset('__record_group_id',
                         data=record_group_trace(times, phase_windows)[:, None])
        f.create_dataset('__time', data=times[:, None])


def write_phases(h5file, phase_windows=None, frames_per_phase=30, visual_names=None):
    """Write one `phaseN` group per stimulation phase into an open stimulus log.

    The ingest turns each of these into a Phase entity: group attributes become
    `<file>/<attr>` attributes on it, and datasets inside the group become
    attributes under their own name.

    Kept separate from the rest of the display file so a test can build a
    stimulus log with a different number of phases, or vary what they contain,
    without rewriting the CMN base data alongside it.
    """
    phase_windows = PHASE_WINDOWS if phase_windows is None else phase_windows
    visual_names = VISUAL_NAMES if visual_names is None else visual_names

    for phase_index, (start, end) in sorted(phase_windows.items()):
        group = h5file.create_group(f'phase{phase_index}')

        group.attrs['__visual_name'] = visual_names[phase_index % len(visual_names)]
        group.attrs['__start_time'] = start
        group.attrs['__target_duration'] = end - start

        group.create_dataset('frame_index', data=np.arange(frames_per_phase)[:, None])
        group.create_dataset('__time',
                             data=np.linspace(start, end, frames_per_phase)[:, None])

    return sorted(phase_windows)


def write_display_file(path, patch_num=12, cmn_frames=30, phase_windows=None,
                       visual_names=None):
    """Stimulus log with the stimulation phases plus the CMN base data."""
    rng = np.random.default_rng(4)

    with h5py.File(os.path.join(path, 'Display.hdf5'), 'w') as f:
        f.attrs['__display_fps'] = 60.0
        f.attrs['__protocol_name'] = 'SyntheticProtocol'

        # Datasets at root level land directly on the Recording entity
        f.create_dataset('__time', data=np.arange(cmn_frames, dtype=float)[:, None])

        write_phases(f, phase_windows, frames_per_phase=cmn_frames,
                     visual_names=visual_names)

        # A non-phase group becomes namespaced attributes on the Recording
        cmn = f.create_group('CMN')
        cmn.attrs['seed'] = 42
        centers = rng.normal(size=(patch_num, 3))
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        cmn.create_dataset('centers_0', data=centers)
        cmn.create_dataset('motion_vectors_0', data=rng.normal(size=(cmn_frames, patch_num, 3)))


def write_camera_file(path, device='fish_embedded', frame_rate=160, frame_num=400,
                      with_video=True):
    """Camera.hdf5 plus the video the camera wrote beside it.

    The pair is what the ingest has to reconcile: the HDF5 carries the frame
    times and whatever tracking ran on the frames, the file beside it carries
    the frames. Named as vxpy names them - `<device>_frame.avi` for the device
    listed in `__camera_device_list`.

    The video is not a real one. Nothing in the ingest decodes it, which is the
    point: entarchy stores it and never looks inside.
    """
    times = np.arange(frame_num) / frame_rate

    with h5py.File(os.path.join(path, 'Camera.hdf5'), 'w') as f:
        f.attrs['__camera_device_list'] = np.array([device], dtype=object)
        f.attrs[f'__{device}_frame_rate'] = frame_rate
        f.attrs[f'__{device}_height'] = 304
        f.attrs[f'__{device}_model'] = 'a2A1920-160umBAS'

        f.create_dataset('__time', data=times[:, None])
        f.create_dataset(f'{device}_frame_time', data=times[:, None])
        # Nine tail keypoints, x/y/confidence - the shape the tracking writes
        rng = np.random.default_rng(11)
        f.create_dataset('tail_pose_data', data=rng.normal(size=(frame_num, 9, 3)))
        f.create_dataset('tail_pose_data_time', data=times[:, None])

    if with_video:
        with open(os.path.join(path, f'{device}_frame.avi'), 'wb') as f:
            f.write(b'RIFF----AVI ' + bytes(range(256)) * 8)


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
                  roi_num=4, plane_num=2, frames_per_plane=None, with_zstack=True,
                  phase_num=None, phase_windows=None, visual_names=None,
                  with_camera=True, with_video=True):
    """Create the folder tree and return a description of what was written.

    Phases default to PHASE_WINDOWS. Pass `phase_num` for that many evenly
    spaced ones, or `phase_windows` to place them yourself. Either way the io
    file and the stimulus log are written from the same windows, which they have
    to be: the ingest locates a phase's calcium frames by its index in the record
    group trace.
    """
    if phase_windows is None and phase_num is not None:
        phase_windows = make_phase_windows(phase_num)
    if phase_windows is None:
        phase_windows = PHASE_WINDOWS

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

    write_io_file(recording_path, phase_windows)
    write_display_file(recording_path, phase_windows=phase_windows,
                       visual_names=visual_names)
    if with_camera:
        write_camera_file(recording_path, with_video=with_video)

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
        'phase_indices': sorted(phase_windows),
        'phase_windows': dict(phase_windows),
        'camera_device': 'fish_embedded' if with_camera else None,
        'video_path': (os.path.join(recording_path, 'fish_embedded_frame.avi')
                       if with_camera and with_video else None),
    }
