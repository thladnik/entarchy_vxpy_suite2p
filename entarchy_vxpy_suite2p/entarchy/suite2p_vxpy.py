from __future__ import annotations
import os
import pathlib
from typing import Callable

import h5py
import numpy as np
import pandas as pd
import scipy
import tifffile
import tqdm
import yaml

import entarchy


__all__ = ['Animal', 'Recording', 'Layer', 'Roi', 'Phase', 'Suite2PVxPy']


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
        return self.entarchy.get(Recording, f'[Animal]uuid == "{self.uuid}"')  # type: ignore

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Animal]uuid == "{self.uuid}"')  # type: ignore


class Recording(entarchy.Entity):
    collection_type = RecordingCollection
    # # Child entity types may be added by name meta // Does not work yet
    # _child_entity_types = ['Roi', 'Phase']

    @property
    def animal(self) -> Animal:
        return self.parent # type: ignore

    @property
    def phases(self) -> PhaseCollection:
        return self.entarchy.get(Phase, f'[Recording]uuid == "{self.uuid}"')  # type: ignore

    @property
    def layers(self) -> LayerCollection:
        return self.entarchy.get(Layer, f'[Recording]uuid == "{self.uuid}"')  # type: ignore

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Recording]uuid == "{self.uuid}"')  # type: ignore


class Phase(entarchy.Entity):
    collection_type = PhaseCollection

    @property
    def recording(self) -> Recording:
        return self.parent # type: ignore

    @property
    def animal(self) -> Animal:
        return self.recording.parent # type: ignore


class Layer(entarchy.Entity):
    collection_type = LayerCollection

    @property
    def rois(self):
        return self.entarchy.get(Roi, f'[Recording]uuid == "{self.uuid}"')  # type: ignore

    @property
    def recording(self) -> Recording:
        return self.parent  # type: ignore

    @property
    def animal(self) -> Animal:
        return self.recording.animal  # type: ignore


class Roi(entarchy.Entity):
    collection_type = RoiCollection

    @property
    def layer(self) -> Layer:
        return self.parent  # type: ignore

    @property
    def recording(self) -> Recording:
        return self.layer.recording  # type: ignore


# Child entity types may be added by method calls
Animal.add_child_entity_type(Recording)
Recording.add_child_entity_type(Layer)
Layer.add_child_entity_type(Roi)
Recording.add_child_entity_type(Phase)


class Suite2PVxPy(entarchy.Entarchy):

    implementation_version = '0.2'

    _hierarchy_root = Animal

    @entarchy.digest_method
    def add_animal(self, path: str) -> Animal:

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

    @entarchy.digest_method
    def add_recording(self, animal: Animal, path: str,
                      sync_signal: str = None, sync_signal_time: str = None,
                      sync_type = None, frame_avg_num: int | Callable = 1) -> Recording | None:

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

        print(f'Process recording folder {path}')

        with self:

            print('> Calculate frame timing of signal')
            with h5py.File(os.path.join(path, 'Io.hdf5'), 'r') as io_file:

                sync_data = np.squeeze(io_file[sync_signal])[:]
                sync_data_times = np.squeeze(io_file[sync_signal_time])[:]

                # Calculate frame timing
                frame_idcs_all, frame_times_all = frame_time_methods[sync_type](sync_data, sync_data_times)

                # Interpolate record group IDs to imaging frame time
                try:
                    record_group_ids = io_file['__record_group_id'][:].squeeze()
                    record_group_ids_time = io_file['__time'][:].squeeze()
                except KeyError as _:
                    # For backwards compatibility to pre-2023 vxpy data
                    record_group_ids = io_file['record_group_id'][:].squeeze()
                    record_group_ids_time = io_file['global_time'][:].squeeze()

                ca_rec_group_id_fun = scipy.interpolate.interp1d(record_group_ids_time,
                                                                 record_group_ids,
                                                                 kind='nearest')

            # Find all layers in suite2p folder
            layers = []
            for _name in os.listdir(os.path.join(path, 'suite2p')):
                if (not os.path.isdir(os.path.join(path, 'suite2p', _name))
                        or not _name.startswith('plane')):
                    continue
                layers.append(_name)
            layer_num = len(layers)

            # Create new recording entity
            recording = Recording(self, _id=recording_id, _parent=animal)
            self.add_new_entity(recording)
            print(f'> Create {recording}')

            # Add metadata
            add_metadata(recording, path)

            # Calculate layer times
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

            # Get imaging rate from sync signal
            dt_frames = np.diff(frame_times).mean()  # seconds
            imaging_rate = 1. / dt_frames  # Hz
            recording['imaging_rate'] = imaging_rate
            print(f'> Estimated, effective imaging rate {imaging_rate:.2f}Hz')

            # Interpolate record_group_ids to frame times
            record_group_ids = ca_rec_group_id_fun(frame_times)
            recording['record_group_ids'] = record_group_ids

            for data_fn in os.listdir(path):
                if not any([data_fn.lower().endswith(fn) for fn in ['.h5', 'hdf5']]):
                    continue

                # Get short name for attribute names
                fn_short = data_fn.split('.')[0].lower()
                phase_data = {}
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

                            # Get phase entity
                            if key1 in phase_data:
                                phase = phase_data[key1]
                            else:
                                # Add new phase entity
                                phase = Phase(self, _id=key1, _parent=recording)
                                self.add_new_entity(phase)
                                phase_data[key1] = phase
                                phase['index'] = int(key1.replace('phase', ''))

                                # Add calcium start/end indices
                                in_phase_indices = np.where(record_group_ids == phase['index'])[0]
                                start_index = np.argmin(np.abs(frame_times - frame_times[in_phase_indices[0]]))
                                end_index = np.argmin(np.abs(frame_times - frame_times[in_phase_indices[-1]]))
                                phase['ca_start_index'] = start_index
                                phase['ca_end_index'] = end_index

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
                frame_times = frame_times_by_layer[layer_idx]

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
                layer['t_offset'] = layer_idx * dt_frames / layer_num

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

                    if iscell_all is not None:
                        roi['iscell'] = iscell_all[roi_idx]

            # Add recording-level timing data after layers have been processed
            #  (frame_times may be truncated to match signal length, so we need to add them after processing layers)
            recording['signal_length'] = frame_times.shape[0]
            recording['ca_times'] = frame_times

        return recording

    # @entarchy.digest_method
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


def unravel_dict(dict_data: dict, entity: entarchy.Entity, path: str):
    for key, item in dict_data.items():
        if isinstance(item, dict):
            unravel_dict(item, entity, f'{path}/{key}')
            continue
        entity[f'{path}/{key}'] = item


if __name__ == '__main__':
    pass