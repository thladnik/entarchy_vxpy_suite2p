from __future__ import annotations
import os
import pathlib
from typing import Callable, TypeVar, cast

import h5py
import numpy as np
import pandas as pd
import scipy
import tifffile
import tqdm
import yaml

import entarchy


__all__ = ['Suite2PVxPy',
           'Animal', 'Recording', 'Layer', 'Roi', 'Phase',
           'AnimalCollection', 'RecordingCollection', 'LayerCollection', 'RoiCollection', 'PhaseCollection']


# C = TypeVar("C", bound=schema.Collection)
# def get_collection_as(ent: schema.Entarchy, entity_type: type[schema.Entity], *expr, **kw) -> C:
#     return cast(C, ent.get(entity_type, *expr, **kw))


class AnimalCollection(entarchy.Collection):
    pass


class RecordingCollection(entarchy.Collection):
    pass


class LayerCollection(entarchy.Collection):
    pass


class RoiCollection(entarchy.Collection):
    pass


class PhaseCollection(entarchy.Collection):
    pass


class Animal(entarchy.Entity):
    collection_type = AnimalCollection

    @property
    def recordings(self) -> RecordingCollection:
        return self.entarchy.get(Recording, f'[Animal]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.entarchy.get(Layer, f'[Animal]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Animal]uuid == "{self.uuid}"')  # type: ignore[return-value]


class Recording(entarchy.Entity):
    collection_type = RecordingCollection
    # # Child entity types may be added by name meta // Does not work yet
    # _child_entity_types = [Roi, Phase]

    @property
    def animal(self) -> Animal:
        return self.parent  # type: ignore[return-value]

    @property
    def phases(self) -> PhaseCollection:
        return self.entarchy.get(Phase, f'[Recording]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.entarchy.get(Layer, f'[Recording]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Recording]uuid == "{self.uuid}"')  # type: ignore[return-value]


class Phase(entarchy.Entity):
    collection_type = PhaseCollection

    @property
    def recording(self) -> Recording:
        return self.parent  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.parent  # type: ignore[return-value]


class Layer(entarchy.Entity):
    collection_type = LayerCollection

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Layer]uuid == "{self.uuid}"')  # type: ignore[return-value]
        # return get_collection_as[RoiCollection](self.schema, Roi, f'[Recording]uuid == "{self.uuid}"')

    @property
    def recording(self) -> Recording:
        return self.parent  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.animal  # type: ignore[return-value]


class Roi(entarchy.Entity):
    collection_type = RoiCollection

    @property
    def layer(self) -> Layer:
        return self.parent  # type: ignore[return-value]

    @property
    def recording(self) -> Recording:
        return self.layer.recording  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.animal  # type: ignore[return-value]


# Establish hierarchy
Animal.add_child_entity_type(Recording)
Recording.add_child_entity_type(Layer)
Layer.add_child_entity_type(Roi)
Recording.add_child_entity_type(Phase)


class Suite2PVxPy(entarchy.Entarchy):

    _implementation_compat_version_list = ['0.2']
    _implementation_version = '0.2'

    _hierarchy_root_type = Animal

    @entarchy.digest_method
    def add_animal(self, path: str, use_anatomy_reference: str = None) -> Animal:

        path = pathlib.Path(path).as_posix()

        print(f'> Add animal from path {path}')

        # Create animal
        path_parts = path.split('/')
        animal_id = path_parts[-1]

        animal_collection = (self.get(Animal) & f'id == "{animal_id}"')

        if len(animal_collection) > 0:
            print(f'WARNING: recording with id {animal_id} already exists. Skipping.')
            return animal_collection[0]

        with self:

            # Create new animal entity
            print(f'>> Create new entity for animal {animal_id}')
            animal = Animal(self, _id=animal_id, _parent=self.root)
            self.add_new_entity(animal)

            # Search for zstacks
            zstack_names = []
            for fn in os.listdir(path):
                _p = os.path.join(path, fn)
                if os.path.isdir(_p):
                    continue
                if 'zstack' in fn:
                    if fn.lower().endswith(('.tif', '.tiff')):
                        zstack_names.append(fn)

            # Add first stack that was detected
            if len(zstack_names) > 0:
                if len(zstack_names) > 1:
                    print(f'WARNING: multiple zstacks detected, using {zstack_names[0]}')

                # Load zstack
                zstack_data = tifffile.imread(os.path.join(path, zstack_names[0]))

                print(f'>> Add zstack {zstack_names[0]} of shape {zstack_data.shape}')
                animal['zstack_fn'] = zstack_names[0]
                animal['zstack'] = zstack_data

        # Add metadata
        add_metadata(animal, path)

        # Search for valid registration path in animal folder
        valid_reg_path = None
        if 'ants_registration' in os.listdir(path):
            for mov_folder in os.listdir(os.path.join(path, 'ants_registration')):
                for ref_folder in os.listdir(os.path.join(path, 'ants_registration', mov_folder)):

                    # Skip if user specified a particular reference name
                    if use_anatomy_reference is not None and use_anatomy_reference not in ref_folder:
                        continue

                    reg_path = os.path.join(path, 'ants_registration', mov_folder, ref_folder)

                    # If there is a transform file, we'll take it
                    if 'Composite.h5' in os.listdir(reg_path):
                        valid_reg_path = reg_path
                        break

        # Write registration metadata to animal entity
        if valid_reg_path is not None:
            print(f'Loading ANTs registration metadata at {valid_reg_path}')
            ants_metadata = yaml.safe_load(open(os.path.join(valid_reg_path, 'metadata.yaml'), 'r'))
            animal.update({f'ants/{n}': v for n, v in ants_metadata.items()})

        self.commit()

        return animal

    # Behaviour cameras write one video per device beside the HDF5 files, named
    #  after the device that produced it: fish_embedded -> fish_embedded_frame.avi
    VIDEO_SUFFIXES = ('.avi', '.mp4', '.mkv', '.mov')

    def _camera_devices(self, path: str) -> list[str]:
        """The camera devices Camera.hdf5 says this recording used."""
        camera_path = next((os.path.join(path, name) for name in os.listdir(path)
                            if name.lower() == 'camera.hdf5'), None)
        if camera_path is None:
            return []

        with h5py.File(camera_path, 'r') as h5file:
            devices = h5file.attrs.get('__camera_device_list', [])

        return [str(device) for device in devices]

    def _add_recording_videos(self, recording: Recording, path: str) -> list[str]:
        """Take the behaviour videos of a recording into the entarchy.

        Camera.hdf5 already carries the frame times and whatever tracking ran on
        the video; what it does not carry is the video. Without it the frames
        behind `tail_pose_data` cannot be looked at again, and an entarchy is
        meant to hold everything it needs.

        Videos are copied, never moved - an ingest must not consume the raw data
        it was pointed at. Files are matched to the camera devices the recording
        declares, so a video that is an output rather than an acquisition (a
        stimulus animation dropped in the folder, say) is left alone.

        Returns:
            list of str: the attribute names written.
        """
        entries = {name.lower(): name for name in os.listdir(path)}
        written = []

        for device in self._camera_devices(path):
            for suffix in self.VIDEO_SUFFIXES:
                candidate = entries.get(f'{device}_frame{suffix}'.lower())
                if candidate is None:
                    continue

                name = f'camera/{device}/video'
                source = os.path.join(path, candidate)
                print(f'> Take in {candidate} '
                      f'({os.path.getsize(source) / 1024 ** 2:.1f} MB)')
                recording.set_media(name, source)
                written.append(name)
                break
            else:
                print(f'WARNING: camera device "{device}" has no video file in {path}')

        return written

    # What a Roi must carry whatever produced it. A source's own vocabulary stays
    #  namespaced beside these - s2p/npix, caiman/SNR_comp - so nothing is lost;
    #  what these names buy is that analysis can read a Roi without knowing which
    #  software segmented it.
    ROI_REQUIRED = ('index', 'fluorescence')
    ROI_OPTIONAL = ('spikes', 'is_unit', 'unit_probability')

    def _suite2p_layers(self, path: str) -> list[str]:
        """The plane directories suite2p wrote, in index order."""
        suite2p_path = os.path.join(path, 'suite2p')
        if not os.path.isdir(suite2p_path):
            return []

        names = [name for name in os.listdir(suite2p_path)
                 if name.startswith('plane') and os.path.isdir(os.path.join(suite2p_path, name))]

        return sorted(names, key=lambda name: int(name.replace('plane', '')))

    def _read_record_groups(self, path: str):
        """The stimulation phase id per analog sample, on the io timebase.

        This is what says when each phase actually ran, and it needs no
        microscope - which is why it is read here rather than inside the imaging
        ingest, where it used to be resampled onto calcium frames.
        """
        io_path = os.path.join(path, 'Io.hdf5')
        if not os.path.exists(io_path):
            print('WARNING: no Io.hdf5; phases will have no time windows')
            return None

        with h5py.File(io_path, 'r') as io_file:
            if '__record_group_id' in io_file:
                ids = io_file['__record_group_id'][:].squeeze()
                times = io_file['__time'][:].squeeze()
            else:
                ids = io_file['record_group_id'][:].squeeze()
                times = io_file['global_time'][:].squeeze()

        return ids, times

    def _imaging_timing(self, path: str, layer_num: int, frame_avg_num, animal: Animal,
                        recording: Recording, sync_signal: str, sync_signal_time: str,
                        sync_type: str) -> dict:
        """When each imaging frame was acquired, per layer.

        Timing is a property of how the microscope scanned, not of the software
        that segmented the result: the demultiplexing below describes a scanner
        that visits planes in turn, and would be wrong for a light sheet stack
        whichever program found the ROIs.
        """
        with h5py.File(os.path.join(path, 'Io.hdf5'), 'r') as io_file:
            sync_data = np.squeeze(io_file[sync_signal])[:]
            sync_data_times = np.squeeze(io_file[sync_signal_time])[:]

        _, frame_times_all = frame_time_methods[sync_type](sync_data, sync_data_times)

        if isinstance(frame_avg_num, int):
            frame_avg_num_cur = frame_avg_num
        else:
            if not callable(frame_avg_num):
                raise Exception('frame_avg_num must be int or callable')
            frame_avg_num_cur = frame_avg_num(animal.id, recording.id)

        frame_times_by_layer = []
        for layer_idx in range(layer_num):
            _f_times = frame_times_all[int(layer_idx + frame_avg_num_cur // 2)::(layer_num * frame_avg_num_cur)]
            frame_times_by_layer.append(_f_times)

        # For now, use frame timing data of first layer for recording-level timing data and phase assignment
        frame_times = frame_times_by_layer[0].squeeze()
        # TODO: improve this in future? There is a time offset between layers due to sequential acquisition
        #  Doing this properly would require LinkEntities between layers and stimulation phases, which is not implemented yet

        dt_frames = np.diff(frame_times).mean()  # seconds

        return {'frame_times_by_layer': frame_times_by_layer,
                'frame_times': frame_times,
                'dt_frames': dt_frames,
                'rate': 1. / dt_frames}

    def _ingest_hdf5_files(self, recording: Recording, path: str, record_groups) -> dict:
        """Everything the acquisition software wrote, imaging or not.

        Phases are created here rather than in the imaging ingest: a stimulation
        phase is a fact about what was shown, and a recording with no microscope
        still has them.

        Returns:
            dict: the Phase entities, keyed by index.
        """
        phase_data = {}

        for data_fn in os.listdir(path):
            if not any([data_fn.lower().endswith(fn) for fn in ['.h5', 'hdf5']]):
                continue

            # Get short name for attribute names
            fn_short = data_fn.split('.')[0].lower()
            with h5py.File(os.path.join(path, data_fn), 'r') as h5file:

                print(f'> {data_fn}')
                # Get attributes
                recording.update({f'{fn_short}/attrs/{k}': v for k, v in h5file.attrs.items()})
                for key1, member1 in tqdm.tqdm(h5file.items()):

                    # If dataset, save to recording directly
                    if isinstance(member1, h5py.Dataset):
                        recording[f'{fn_short}/{key1}'] = np.squeeze(member1[:])
                        continue

                    # Otherwise it's a group -> keep going

                    # Add phase
                    if key1.startswith('phase'):
                        phase_index = int(key1.replace('phase', ''))

                        # Shared across files: two logs describing the same phase
                        #  contribute attributes to one entity rather than
                        #  colliding on a second one with the same id
                        if phase_index in phase_data:
                            phase = phase_data[phase_index]
                        else:
                            phase = Phase(self, _id=key1, _parent=recording)
                            self.add_new_entity(phase)
                            phase_data[phase_index] = phase
                            phase['index'] = phase_index

                            window = _phase_time_window(record_groups, phase_index)
                            if window is not None:
                                phase['start_time'], phase['end_time'] = window

                        # Write attributes
                        for attr_key, attr_value in member1.attrs.items():
                            phase[f'{fn_short}/{attr_key}'] = attr_value

                        # Write datasets
                        for key2, member2 in member1.items():
                            if isinstance(member2, h5py.Dataset):
                                phase[key2] = np.squeeze(member2[:])
                            else:
                                print('WARNING: nested groups in phase not supported yet')

                    # Add other data
                    else:
                        # Write attributes
                        for k, v in member1.attrs.items():
                            recording[f'{fn_short}/{key1}/{k}'] = v

                        # Write datasets
                        for key2, member2 in member1.items():
                            if isinstance(member2, h5py.Dataset):
                                recording[f'{fn_short}/{key1}/{key2}'] = np.squeeze(member2[:])

        return phase_data

    def _add_phase_frame_windows(self, phases: dict, frame_times: np.ndarray,
                                 record_group_ids: np.ndarray) -> None:
        """Which imaging frames fall inside each phase.

        Only meaningful once there are imaging frames, and only unambiguous
        while there is one source of them - see
        docs/proposals/imaging-sources.md for where this goes next.
        """
        for phase in phases.values():
            in_phase_indices = np.where(record_group_ids == phase['index'])[0]
            if len(in_phase_indices) == 0:
                print(f'WARNING: phase {phase["index"]} covers no imaging frames')
                continue

            start_index = np.argmin(np.abs(frame_times - frame_times[in_phase_indices[0]]))
            end_index = np.argmin(np.abs(frame_times - frame_times[in_phase_indices[-1]]))
            phase['ca_start_index'] = start_index
            phase['ca_end_index'] = end_index

    def _ingest_suite2p(self, recording: Recording, path: str, layers: list[str],
                        timing: dict) -> None:
        """Layers and ROIs from suite2p's output."""
        layer_num = len(layers)
        frame_times = timing['frame_times']

        for layer_str in layers:

            # Add layer
            layer = Layer(self, _id=layer_str, _parent=recording)
            self.add_new_entity(layer)
            print(f'> Process {layer}')

            # Get path to plane data
            s2p_path = os.path.join(path, 'suite2p', layer_str)

            # Get plane index
            layer_idx = int(layer_str.replace('plane', ''))

            # Get frame times for this layer
            frame_times = timing['frame_times_by_layer'][layer_idx]

            # Load suite2p's analysis options
            print('>> Include suite2p ops')
            ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()
            unravel_dict(ops, layer, 's2p')

            print('>> Load ROI data')
            fluorescence = np.load(os.path.join(s2p_path, 'F.npy'), allow_pickle=True)
            spikes_all = np.load(os.path.join(s2p_path, 'spks.npy'), allow_pickle=True)
            roi_stats_all = np.load(os.path.join(s2p_path, 'stat.npy'), allow_pickle=True)
            # In some suite2p versions the iscell file may be missing?
            try:
                iscell_all = np.load(os.path.join(s2p_path, 'iscell.npy'), allow_pickle=True)
            except:
                iscell_all = None

            # Check if frame times and signal match
            if frame_times.shape[0] != fluorescence.shape[1]:
                print(f'Detected frame times length does not match frame count. '
                      f'Detected frame times: {frame_times.shape[0]} / Frames: {fluorescence.shape[1]}')

                # Shorten signal
                if frame_times.shape[0] < fluorescence.shape[1]:
                    fluorescence = fluorescence[:, :frame_times.shape[0]]
                    print('Truncated signal at end to resolve mismatch. Check debug output to verify')

                # Shorten frame times
                else:
                    frame_times = frame_times[:fluorescence.shape[1]]
                    print('Truncated detected frame times at end to resolve mismatch. Check debug output to verify')

            # Save to recording
            layer['roi_num'] = fluorescence.shape[0]
            layer['t_offset'] = layer_idx * timing['dt_frames'] / layer_num

            print('Load anatomical registration data')
            roi_coordinates = None
            if 'ants_registration' in os.listdir(os.path.join(path, 'suite2p')):
                # Check for registration data in each registration subfolder for current plane
                for fld in os.listdir(os.path.join(path, 'suite2p', 'ants_registration', layer_str)):
                    registration_path = os.path.join(path, 'suite2p', 'ants_registration', layer_str, fld)

                    # Read coordinates of available
                    if 'mapped_points.h5' in os.listdir(registration_path):
                        roi_coordinates = pd.read_hdf(os.path.join(registration_path, 'mapped_points.h5'),
                                                      key='coordinates')

                        print(f'Found ANTs registration data for  ROI coordinates: {registration_path}')
                        break

            if roi_coordinates is None:
                print('WARNING: no ANTs registration data found')

            # Add suite2p's analysis ROI stats
            print('>> Add ROI stats and signals')
            for roi_idx in tqdm.tqdm(range(fluorescence.shape[0])):
                # Create ROI
                roi = Roi(self, _id=f'Roi_{roi_idx}', _parent=layer)
                self.add_new_entity(roi)
                roi['index'] = roi_idx

                roi_stats = roi_stats_all[roi_idx]

                # Write ROI stats
                roi.update({f's2p/{k}': v for k, v in roi_stats.items()})

                # Write ROI coordinates
                if roi_coordinates is not None:
                    coords = roi_coordinates.iloc[roi_idx]
                    roi.update({'ants/x': float(coords.x), 'ants/y': float(coords.y), 'ants/z': float(coords.z)})

                # Write data
                roi['fluorescence'] = fluorescence[roi_idx]
                roi['spikes'] = spikes_all[roi_idx]

                # suite2p packs the classifier's verdict and its confidence into
                #  one two element row. Split, so that a source which classifies
                #  differently - or not at all - writes the same names.
                if iscell_all is not None:
                    roi['is_unit'] = bool(iscell_all[roi_idx][0])
                    roi['unit_probability'] = float(iscell_all[roi_idx][1])

            self._check_roi_contract(layer)

        # Add recording-level timing data after layers have been processed
        #  (frame_times may be truncated to match signal length, so we need to add them after processing layers)
        recording['signal_length'] = frame_times.shape[0]
        recording['ca_times'] = frame_times

    def _check_roi_contract(self, layer: Layer) -> None:
        """Whether a layer's ROIs carry what analysis is entitled to expect.

        Checked on the first ROI of each layer rather than all of them: this
        catches a source that does not honour the contract, which is a mistake
        in the source rather than in one ROI, and reading every ROI of a plane
        to check would cost more than the ingest that wrote them.
        """
        rois = layer.rois
        if len(rois) == 0:
            return

        missing = [name for name in self.ROI_REQUIRED if name not in rois[0]]
        if len(missing) > 0:
            raise RuntimeError(
                f'{layer} produced ROIs without {missing}. Every imaging source '
                f'must write {list(self.ROI_REQUIRED)} so that analysis can read '
                f'an ROI without knowing what segmented it.')

    @entarchy.digest_method
    def add_recording(self, animal: Animal, path: str,
                      sync_signal: str = None, sync_signal_time: str = None,
                      sync_type = None, frame_avg_num: int | Callable = 1,
                      with_video: bool = True, imaging: str = 'auto') -> Recording | None:
        """Ingest one vxpy recording folder.

        Args:
            imaging: which imaging source to take signals from. 'auto' ingests
                suite2p output when the folder has it and skips it otherwise;
                'suite2p' requires it and fails if it is missing; None ingests
                no imaging at all, leaving a recording of stimulus, io and
                behaviour data. See docs/proposals/imaging-sources.md.
            with_video: take the behaviour videos into the entarchy as media.
        """

        sync_type = 'y_mirror' if sync_type is None else sync_type

        sync_signal = 'ai_y_mirror_in' if sync_signal is None else sync_signal

        sync_signal_time = f'{sync_signal}_time' if sync_signal_time is None else sync_signal_time

        path = pathlib.Path(path).as_posix()

        # Create recording
        path_parts = path.split('/')
        recording_id = path_parts[-1]

        # Check if recording with same id already exists for this animal
        if len(self.get(Recording, f'id == "{recording_id}" and [Animal]uuid == "{animal.uuid}"')) > 0:
            print(f'WARNING: recording with id {recording_id} already exists for animal {animal.id}. Skipping.')
            return None

        # Check if path appears to be a recording path by looking for expected files
        is_rec_path = any(n in [n1.lower() for n1 in os.listdir(path)]
                          for n in ['io.hdf5', 'camera.hdf5', 'display.hdf5', 'gui.hdf5'])
        if not is_rec_path:
            print(f'WARNING: {path} does not appear to be vxpy recording folder. Skipping.')
            return None

        if imaging not in (None, 'auto', 'suite2p'):
            raise ValueError(f'Unknown imaging source "{imaging}". '
                             f'Use "suite2p", "auto" or None.')

        layers = self._suite2p_layers(path) if imaging is not None else []
        if imaging == 'suite2p' and len(layers) == 0:
            raise FileNotFoundError(
                f'imaging="suite2p" but {path} has no suite2p plane directories. '
                f'Use imaging="auto" to ingest whatever is there, or None to skip it.')

        with_imaging = len(layers) > 0
        if with_imaging and not os.path.exists(os.path.join(path, 'Io.hdf5')):
            raise FileNotFoundError(
                f'{path} has suite2p output but no Io.hdf5, so its frames cannot '
                f'be timed. Pass imaging=None to ingest the rest of it.')

        print(f'Process recording folder {path}')

        with self:

            # When each stimulation phase ran, on the io timebase. Needed whether
            #  or not there is imaging, so it is read before deciding.
            record_groups = self._read_record_groups(path)

            # Create new recording entity
            recording = Recording(self, _id=recording_id, _parent=animal)
            self.add_new_entity(recording)
            print(f'> Create {recording}')

            # Add metadata
            add_metadata(recording, path)
            recording['has_imaging'] = with_imaging

            timing = None
            if with_imaging:
                print('> Calculate frame timing of signal')
                timing = self._imaging_timing(path, len(layers), frame_avg_num, animal,
                                              recording, sync_signal, sync_signal_time,
                                              sync_type)
                recording['imaging_rate'] = timing['rate']
                print(f'> Estimated, effective imaging rate {timing["rate"]:.2f}Hz')

                # Interpolate record_group_ids to frame times
                ids, times = record_groups
                on_frames = scipy.interpolate.interp1d(times, ids, kind='nearest')(
                    timing['frame_times'])
                recording['record_group_ids'] = on_frames

            phases = self._ingest_hdf5_files(recording, path, record_groups)

            if with_imaging:
                self._add_phase_frame_windows(phases, timing['frame_times'],
                                              recording['record_group_ids'])

            if with_video:
                self._add_recording_videos(recording, path)

            if with_imaging:
                self._ingest_suite2p(recording, path, layers, timing)

        return recording

    @entarchy.digest_method
    def update_roi_coordinates_from_registration(self, recording_path: str):
        """
        Function to update coordinates of the ROIs from a given recording path
        """

        # Find recordings based on path
        parts = pathlib.Path(recording_path).as_posix().split('/')
        recording = self.get(Recording, f'[Animal]id == "{parts[-2]}" AND id == "{parts[-1]}"')[0]

        # Find all layers in suite2p folder
        layer_names = []
        for _name in os.listdir(os.path.join(recording_path, 'suite2p')):
            if (not os.path.isdir(os.path.join(recording_path, 'suite2p', _name))
                    or not _name.startswith('plane')):
                continue
            layer_names.append(_name)

        for layer_n in layer_names:

            layer = self.get(Layer, f'id == "{layer_n}" and [Recording]uuid == "{recording.uuid}"')

            roi_coordinates = None
            if 'ants_registration' in os.listdir(os.path.join(recording_path, 'suite2p')):
                # Check for registration data in each registration subfolder for current plane
                for fld in os.listdir(os.path.join(recording_path, 'suite2p', 'ants_registration', layer_n)):
                    registration_path = os.path.join(recording_path, 'suite2p', 'ants_registration', layer_n, fld)

                    # Read coordinates of available
                    if 'mapped_points.h5' in os.listdir(registration_path):
                        roi_coordinates = pd.read_hdf(os.path.join(registration_path, 'mapped_points.h5'),
                                                      key='coordinates')

                        print(f'Found ANTs registration data for  ROI coordinates in {layer}: {registration_path}')
                        break

            if roi_coordinates is None:
                print('WARNING: no ANTs registration data found')
                return

            # Update ROI coordiantes
            rois = layer.rois
            for k in ['x', 'y', 'z']:
                rois[f'ants/{k}'] = rois['index'].apply(lambda idx: roi_coordinates.iloc[idx][k])

    # @schema.digest_method
    # def test_digest(self) -> None:
    #
    #     import random
    #
    #     # Use in context to control commits
    #     #  and speed up adding multiple entities and their attributes
    #     with self:
    #
    #         # for i in tqdm.tqdm(range(3), desc='Animals', position=0):
    #         for i in range(3):
    #             animal = Animal(self, _id=f'Animal_{i}', _parent=self.root)
    #             print(f'Add {animal}')
    #             for k, v in {'age': random.randint(24, 96),
    #                          'weight': random.randint(20, 100) / 10,
    #                          'strain': random.choice(['jf1', 'mpn400', 'jf7']),
    #                          'zstack': np.random.randint(0, 255, size=(50, 512, 512), dtype=np.uint8)}.items():
    #                 animal[k] = v
    #
    #             self.add_new_entity(animal)
    #
    #             # for j in tqdm.tqdm(range(7), desc='Recordings', position=1, leave=False):
    #             for j in range(random.randint(2, 4)):
    #                 recording = Recording(self, _id=f'Recording_{j}', _parent=animal)
    #                 print(f'> Add {recording}')
    #                 self.add_new_entity(recording)
    #
    #                 for jj in range(5):
    #                     recording[f'rec_param_int_{jj}'] = random.randint(0, 1000)
    #                 for jj in range(5):
    #                     recording[f'rec_param_float_{jj}'] = random.randint(0, 10000) / 10
    #                 for jj in range(5):
    #                     recording[f'rec_param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
    #                 for jj in range(5):
    #                     recording[f'rec_param_array_{jj}'] = np.random.rand(random.randint(10, 100))
    #                 for jj in range(5):
    #                     recording[f'rec_param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]
    #                 for jj in range(2):
    #                     recording[f'rec_param_largelist_{jj}'] = ['abc'] * 20_000_000
    #                 for jj in range(2):
    #                     recording[f'rec_param_largearray_{jj}'] = np.random.rand(*np.random.randint(50, 150, size=(3,)))
    #
    #                 # for p in tqdm.tqdm(range(300), desc='Phases', position=2, leave=False):
    #                 p_num = 300
    #                 print(f'>> Add {p_num} Phases')
    #                 for p in range(p_num):
    #                     phase = Phase(self, _id=f'Phase_{p}', _parent=recording)
    #                     self.add_new_entity(phase)
    #
    #                     for jj in range(5):
    #                         phase[f'phase_param_int_{jj}'] = random.randint(0, 1000)
    #                     for jj in range(5):
    #                         phase[f'phase_param_float_{jj}'] = random.randint(0, 10000) / 10
    #                     for jj in range(5):
    #                         phase[f'phase_param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
    #                     for jj in range(5):
    #                         phase[f'phase_param_array_{jj}'] = np.random.rand(random.randint(10, 100))
    #                     for jj in range(5):
    #                         phase[f'phase_param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]
    #
    #                 for li in range(5):
    #
    #                     print('>> Add Layer', li)
    #                     layer = Layer(self, _id=f'Layer_{li}', _parent=recording)
    #                     self.add_new_entity(layer)
    #
    #                     # for r in tqdm.tqdm(range(random.randint(400, 800)), desc='Rois', position=2, leave=False):
    #                     r_num = random.randint(200, 400)
    #                     print(f'>>> Add {r_num} Rois')
    #                     for r in range(r_num):
    #                         roi = Roi(self, _id=f'Roi_{r}', _parent=layer)
    #                         self.add_new_entity(roi)
    #
    #                         for jj in range(5):
    #                             roi[f'roi_param_int_{jj}'] = random.randint(0, 1000)
    #                         for jj in range(5):
    #                             roi[f'roi_param_float_{jj}'] = random.randint(0, 10000) / 10
    #                         for jj in range(5):
    #                             roi[f'roi_param_string_{jj}'] = random.choice(['foo', 'bar', 'baz', 'lorem', 'ipsum', 'dolor'])
    #                         for jj in range(5):
    #                             roi[f'roi_param_array_{jj}'] = np.random.rand(random.randint(10, 100))
    #                         for jj in range(5):
    #                             roi[f'roi_param_list_{jj}'] = [random.randint(0, 100) for _ in range(random.randint(10, 100))]
    #                         for jj in range(2):
    #                             roi[f'roi_param_largelist_{jj}'] = ['abc'] * 2_000_000
    #                         for jj in range(2):
    #                             roi[f'roi_param_largearray_{jj}'] = np.random.rand(*np.random.randint(1, 70, size=(3,)))
    #
    #                     # Commit after each layer
    #                     self.commit()


def add_metadata(entity: entarchy.Entity, folder_path: str):
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (ending in `meta.yaml`)
    and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path) if f.endswith('metadata.yaml')]

    print(f'Found {len(meta_files)} metadata files in {folder_path}.')

    metadata = {}
    for f in meta_files:
        with open(os.path.join(folder_path, f), 'r') as stream:
            try:
                metadata.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    # Add metadata
    unravel_dict(metadata, entity, 'metadata')


def _phase_time_window(record_groups, index: int):
    """When a stimulation phase started and ended, in seconds.

    Read off the record group trace, so it says when the phase actually ran
    rather than what the protocol asked for - and needs no microscope, which is
    the point: a phase is a fact about the stimulus.
    """
    if record_groups is None:
        return None

    ids, times = record_groups
    inside = np.where(ids == index)[0]
    if len(inside) == 0:
        return None

    return float(times[inside[0]]), float(times[inside[-1]])


def unravel_dict(dict_data: dict, entity: entarchy.Entity, path: str):
    for key, item in dict_data.items():
        if isinstance(item, dict):
            unravel_dict(item, entity, f'{path}/{key}')
            continue
        entity[f'{path}/{key}'] = item


# TODO: replace scipy here, it's overkill and only used twice

# Frame time calculation methods

def ca_frame_times_from_y_mirror(mirror_position: np.ndarray, mirror_time: np.ndarray):
    peak_prominence = (mirror_position.max() - mirror_position.min()) / 4
    peak_idcs, _ = scipy.signal.find_peaks(mirror_position, prominence=peak_prominence)
    trough_idcs, _ = scipy.signal.find_peaks(-mirror_position, prominence=peak_prominence)

    # Find first trough
    first_peak = peak_idcs[0]
    first_trough = trough_idcs[trough_idcs < first_peak][-1]

    # Discard all before (and including) first trough
    trough_idcs = trough_idcs[first_trough < trough_idcs]
    frame_idcs = np.sort(np.concatenate([trough_idcs, peak_idcs]))

    # Get corresponding times
    frame_times = mirror_time[frame_idcs]

    return frame_idcs, frame_times


def ca_frame_times_from_sync_toggle(sync_signal: np.ndarray, sync_time: np.ndarray):
    frame_indices = np.where(np.diff(sync_signal) > 0)
    frame_times = sync_time[frame_indices]

    return frame_indices, frame_times


frame_time_methods = {
    'y_mirror': ca_frame_times_from_y_mirror,
    'frame_sync_toggle': ca_frame_times_from_sync_toggle,
}


if __name__ == '__main__':
    pass