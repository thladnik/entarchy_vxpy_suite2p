from __future__ import annotations
import os
import pathlib
import time
import traceback
from typing import Callable, TypeVar, cast, overload

import h5py
import numpy as np
import pandas as pd
import scipy
import tifffile
import tqdm
import yaml

import entarchy


__all__ = ['Suite2PVxPy',
           'Experiment', 'Animal', 'Recording', 'Imaging', 'Layer', 'Roi', 'Phase',
           'ExperimentCollection', 'AnimalCollection', 'RecordingCollection',
           'ImagingCollection', 'LayerCollection', 'RoiCollection', 'PhaseCollection',
           'ImagingSource', 'Suite2pSource', 'ImagingSpec', 'imaging_sources',
           'FrameTiming', 'SyncSignalTiming', 'ClockDivisionTiming', 'CameraTiming',
           'is_recording_folder', 'scan_experiment']


# Each names the entity it holds, so that indexing and iterating one gives that
#  entity rather than the base class - recording.phases[0] is a Phase, and a
#  Phase's own properties complete. The names are quoted because every
#  collection here is declared before the entity it holds.


class ExperimentCollection(entarchy.Collection['Experiment']):
    pass


class AnimalCollection(entarchy.Collection['Animal']):
    pass


class RecordingCollection(entarchy.Collection['Recording']):
    pass


class LayerCollection(entarchy.Collection['Layer']):
    pass


class RoiCollection(entarchy.Collection['Roi']):
    pass


class PhaseCollection(entarchy.Collection['Phase']):
    pass


class ImagingCollection(entarchy.Collection['Imaging']):

    @overload
    def __getitem__(self, item: str) -> 'Imaging': ...

    @overload
    def __getitem__(self, item: int) -> 'Imaging': ...

    @overload
    def __getitem__(self, item: slice) -> list['Imaging']: ...

    @overload
    def __getitem__(self, item: list) -> pd.DataFrame: ...

    def __getitem__(self, item):  # type: ignore[override]
        """Also addressable by source name, since that is what an Imaging is.

        Which is why the overloads are spelled out here rather than inherited:
        on any other collection a string names an attribute and gives its
        values, and on this one it names a source and gives that source.

        A type checker is right to call that an incompatible override, and the
        ignore above is the record of it being deliberate. The way to be rid of
        it would be a named method - imaging.by_source('ca') - leaving the
        brackets meaning what they mean everywhere else.
        """
        if isinstance(item, str):
            matching = [imaging for imaging in self if imaging.id == item]
            if len(matching) == 0:
                available = ', '.join(sorted(imaging.id for imaging in self)) or 'none'
                raise KeyError(f'No imaging source "{item}". Available: {available}.')
            return matching[0]

        return super().__getitem__(item)


class Experiment(entarchy.Entity):
    """One experiment: a set of animals recorded under one protocol.

    The raw tree has always had this level - `<experiment>/<animal>/<recording>`
    - and the ingest used to flatten it onto a string attribute. Two things
    make it worth an entity of its own. Which animals belong together is a
    fact about the data rather than a label, so `[Experiment]id == "cmn"`
    reads like every other ancestor query here. And how frames were timed is
    chosen per experiment and cannot be detected, so an Experiment is where
    that choice is recorded, under `imaging/`.
    """

    collection_type = ExperimentCollection

    @property
    def animals(self) -> AnimalCollection:
        return self.entarchy.get(Animal, f'[Experiment]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def recordings(self) -> RecordingCollection:
        return self.entarchy.get(Recording, f'[Experiment]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def phases(self) -> PhaseCollection:
        return self.entarchy.get(Phase, f'[Experiment]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.entarchy.get(Layer, f'[Experiment]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Experiment]uuid == "{self.uuid}"')  # type: ignore[return-value]


class Animal(entarchy.Entity):
    collection_type = AnimalCollection

    @property
    def experiment(self) -> Experiment:
        return self.parent  # type: ignore[return-value]

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
    def imaging(self) -> ImagingCollection:
        """The imaging sources of this recording. May be empty."""
        return self.entarchy.get(Imaging, f'[Recording]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.entarchy.get(Layer, f'[Recording]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Recording]uuid == "{self.uuid}"')  # type: ignore[return-value]

    def sole_imaging(self) -> 'Imaging':
        """The one imaging source, when asking without naming one is unambiguous."""
        sources = list(self.imaging)
        if len(sources) == 1:
            return sources[0]

        if len(sources) == 0:
            raise LookupError(f'{self} has no imaging data.')

        names = ', '.join(sorted(source.id for source in sources))
        raise LookupError(f'{self} has several imaging sources ({names}); '
                          f'say which one.')


class Imaging(entarchy.Entity):
    """One imaging source of a recording: an acquisition and what read it.

    Its id is the source name - `suite2p`, `caiman`, `widefield` - so a
    recording processed twice holds two of these rather than two sets of
    colliding plane ids. Everything about when frames happened and what
    produced the signals lives here rather than on the Recording, which knows
    only what the acquisition software wrote.
    """
    collection_type = ImagingCollection

    @property
    def recording(self) -> 'Recording':
        return self.parent  # type: ignore[return-value]

    @property
    def animal(self) -> 'Animal':
        return self.recording.animal  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.entarchy.get(Layer, f'[Imaging]uuid == "{self.uuid}"')  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Imaging]uuid == "{self.uuid}"')  # type: ignore[return-value]


class Phase(entarchy.Entity):
    collection_type = PhaseCollection

    @property
    def recording(self) -> Recording:
        return self.parent  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.parent  # type: ignore[return-value]

    def frames_in(self, imaging: 'Imaging' = None):
        """Which frames of an imaging source fall inside this phase.

        The window belongs to the pair, not to the phase: two sources of one
        recording sample it differently. It is stored as a link for that reason,
        and this hides that. Omit `imaging` when the recording has one source.

        Returns:
            (start_index, end_index), or None if the phase covers no frames.
        """
        imaging = self.recording.sole_imaging() if imaging is None else imaging

        link = self.entarchy.get_link(self, imaging, PHASE_FRAMES_LINK)
        if link is None:
            return None

        return int(link['start_index']), int(link['end_index'])


class Layer(entarchy.Entity):
    collection_type = LayerCollection

    @property
    def rois(self) -> RoiCollection:
        return self.entarchy.get(Roi, f'[Layer]uuid == "{self.uuid}"')  # type: ignore[return-value]
        # return get_collection_as[RoiCollection](self.schema, Roi, f'[Recording]uuid == "{self.uuid}"')

    @property
    def imaging(self) -> Imaging:
        return self.parent  # type: ignore[return-value]

    @property
    def recording(self) -> Recording:
        return self.imaging.recording  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.animal  # type: ignore[return-value]


class Roi(entarchy.Entity):
    collection_type = RoiCollection

    @property
    def layer(self) -> Layer:
        return self.parent  # type: ignore[return-value]

    @property
    def imaging(self) -> Imaging:
        return self.layer.imaging  # type: ignore[return-value]

    @property
    def recording(self) -> Recording:
        return self.layer.recording  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.animal  # type: ignore[return-value]


# Establish hierarchy
Experiment.add_child_entity_type(Animal)
Animal.add_child_entity_type(Recording)
Recording.add_child_entity_type(Imaging)
Imaging.add_child_entity_type(Layer)
Layer.add_child_entity_type(Roi)
Recording.add_child_entity_type(Phase)

# Which frames of an imaging source a stimulation phase covers. A link, because
#  the window belongs to the pair: the same phase falls on different frames of
#  two sources that sampled it differently.
PHASE_FRAMES_LINK = 'phase_frames' 


class FrameTiming:
    """When each imaging frame was acquired, on the recording's timebase.

    Separate from the source that reads the signals, because the two are
    independent choices: suite2p output may be timed from a galvo mirror, a
    frame sync toggle or a timestamps file, and the same timing may serve
    suite2p and CaImAn alike.
    """
    name: str = 'timing'

    def frame_times(self, path: str, layer_num: int) -> list:
        """One array of frame times per layer, in acquisition order."""
        raise NotImplementedError

    @property
    def config(self) -> dict:
        """What this timing was configured with.

        Frame timing is a choice rather than something readable off the data -
        the ratio a divided clock is read at was measured by hand - so an
        ingest writes this onto the Experiment. Every subclass keeps its
        configuration in plain attributes named after its arguments, which is
        all this collects.
        """
        return {name: value for name, value in vars(self).items()
                if not name.startswith('_')}


class SyncSignalTiming(FrameTiming):
    """Frame times recovered from an analog channel of Io.hdf5.

    A scanner that visits planes in turn puts every layer_num-th frame in the
    same plane, which is what the slicing below undoes. `frame_avg_num` is how
    many frames the scanner averaged into one.
    """
    name = 'sync_signal'

    def __init__(self, method: str = 'y_mirror', signal: str = 'ai_y_mirror_in',
                 signal_time: str = None, frame_avg_num: int = 1):
        if method not in frame_time_methods:
            raise ValueError(f'Unknown frame timing method "{method}". '
                             f'Known: {", ".join(sorted(frame_time_methods))}.')

        self.method = method
        self.signal = signal
        self.signal_time = f'{signal}_time' if signal_time is None else signal_time
        self.frame_avg_num = frame_avg_num

    def all_frame_times(self, path: str) -> np.ndarray:
        with h5py.File(os.path.join(path, 'Io.hdf5'), 'r') as io_file:
            sync_data = np.squeeze(io_file[self.signal])[:]
            sync_data_times = np.squeeze(io_file[self.signal_time])[:]

        return frame_time_methods[self.method](sync_data, sync_data_times)[1]

    def frame_times(self, path: str, layer_num: int) -> list:
        all_times = self.all_frame_times(path)
        step = layer_num * self.frame_avg_num

        return [all_times[int(layer_idx + self.frame_avg_num // 2)::step]
                for layer_idx in range(layer_num)]


class ClockDivisionTiming(FrameTiming):
    """Frame times from a sync line that ticks at a fixed multiple of the rate.

    Not every frame sync pulses once per acquired slice. A scanner whose slow
    axis free-runs keeps emitting its frame clock through the piezo flyback, so
    the line ticks evenly for exactly as long as the acquisition lasts while
    only some of those ticks are frames that were kept. Such a line still times
    the frames - it has to be divided rather than counted.

    `edges_per_volume` is that ratio and need not be a whole number, which is
    the point: it is the volume period measured in clock ticks, and a volume
    that is not phase-locked to the clock gives a fraction. Measure it as
    (edges / volumes) on a recording where both are known, then check that it
    reproduces the frame count of the others.
    """
    name = 'clock_division'

    def __init__(self, edges_per_volume: float, signal: str = 'di_frame_sync',
                 signal_time: str = None):
        if edges_per_volume <= 0:
            raise ValueError('edges_per_volume must be positive, '
                             f'got {edges_per_volume}.')

        self.edges_per_volume = float(edges_per_volume)
        self.signal = signal
        self.signal_time = f'{signal}_time' if signal_time is None else signal_time

    def edge_times(self, path: str) -> np.ndarray:
        with h5py.File(os.path.join(path, 'Io.hdf5'), 'r') as io_file:
            sync_data = np.squeeze(io_file[self.signal])[:]
            sync_data_times = np.squeeze(io_file[self.signal_time])[:]

        return ca_frame_times_from_sync_toggle(sync_data, sync_data_times)[1]

    def frame_times(self, path: str, layer_num: int) -> list:
        edges = self.edge_times(path)
        if len(edges) < 2:
            raise ValueError(f'The sync line "{self.signal}" in {path} has '
                             f'{len(edges)} edges, so it cannot time anything.')

        # Ride the clock rather than a nominal rate, so that a clock which is
        #  slightly off nominal does not accumulate drift across the recording
        volume_num = int((len(edges) - 1) / self.edges_per_volume) + 1
        positions = np.arange(volume_num) * self.edges_per_volume
        times = np.interp(positions, np.arange(len(edges)), edges)

        # Planes are visited in turn within a volume
        slice_dt = np.median(np.diff(times)) / layer_num

        return [times + layer_idx * slice_dt for layer_idx in range(layer_num)]


class CameraTiming(FrameTiming):
    """Frame times the camera itself recorded, for a source that is not scanned.

    One plane, so no demultiplexing: the times are the times.
    """
    name = 'camera'

    def __init__(self, device: str, file_name: str = 'Camera.hdf5'):
        self.device = device
        self.file_name = file_name

    def frame_times(self, path: str, layer_num: int) -> list:
        if layer_num != 1:
            raise ValueError(f'{type(self).__name__} times a single plane, '
                             f'but {layer_num} layers were asked for.')

        with h5py.File(os.path.join(path, self.file_name), 'r') as h5file:
            times = np.squeeze(h5file[f'{self.device}_frame_time'][:])

        return [times]


class ImagingSource:
    """Where signals come from: what is on disk and how to read it.

    An implementation must write the ROI contract - Suite2PVxPy.ROI_REQUIRED -
    whatever its own vocabulary is. Anything else it knows stays namespaced
    under its own name, as suite2p's stats do under `s2p/`.
    """
    name: str = 'source'
    default_timing: str = 'sync_signal'

    def detect(self, path: str) -> bool:
        """Whether this source has output in a recording folder."""
        raise NotImplementedError

    def layer_names(self, path: str) -> list:
        """The planes this source produced, in index order."""
        raise NotImplementedError

    def ingest(self, imaging: Imaging, path: str, frame_times_by_layer: list,
               options: dict) -> None:
        """Create the layers and ROIs of this source under `imaging`."""
        raise NotImplementedError


class Suite2pSource(ImagingSource):
    """suite2p's plane directories: ops, F, spks, stat, iscell."""
    name = 'suite2p'

    def detect(self, path: str) -> bool:
        return len(self.layer_names(path)) > 0

    def layer_names(self, path: str) -> list:
        suite2p_path = os.path.join(path, 'suite2p')
        if not os.path.isdir(suite2p_path):
            return []

        names = [name for name in os.listdir(suite2p_path)
                 if name.startswith('plane')
                 and os.path.isdir(os.path.join(suite2p_path, name))]

        return sorted(names, key=lambda name: int(name.replace('plane', '')))

    def ingest(self, imaging: Imaging, path: str, frame_times_by_layer: list,
               options: dict) -> None:
        ent = imaging.entarchy
        layer_names = self.layer_names(path)
        layer_num = len(layer_names)
        dt_frames = np.diff(frame_times_by_layer[0].squeeze()).mean()

        frame_times = frame_times_by_layer[0].squeeze()

        for layer_str in layer_names:

            layer = Layer(ent, _id=layer_str, _parent=imaging)
            ent.add_new_entity(layer)
            print(f'> Process {layer}')

            s2p_path = os.path.join(path, 'suite2p', layer_str)
            layer_idx = int(layer_str.replace('plane', ''))
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
            except Exception:
                iscell_all = None

            # Check if frame times and signal match
            if frame_times.shape[0] != fluorescence.shape[1]:
                print(f'Detected frame times length does not match frame count. '
                      f'Detected frame times: {frame_times.shape[0]} / Frames: {fluorescence.shape[1]}')

                if frame_times.shape[0] < fluorescence.shape[1]:
                    fluorescence = fluorescence[:, :frame_times.shape[0]]
                    print('Truncated signal at end to resolve mismatch. Check debug output to verify')
                else:
                    frame_times = frame_times[:fluorescence.shape[1]]
                    print('Truncated detected frame times at end to resolve mismatch. Check debug output to verify')

            layer['index'] = layer_idx
            layer['roi_num'] = fluorescence.shape[0]
            layer['t_offset'] = layer_idx * dt_frames / layer_num

            roi_coordinates = self._registration_coordinates(path, layer_str)

            print('>> Add ROI stats and signals')
            for roi_idx in tqdm.tqdm(range(fluorescence.shape[0])):
                roi = Roi(ent, _id=f'Roi_{roi_idx}', _parent=layer)
                ent.add_new_entity(roi)
                roi['index'] = roi_idx

                # suite2p's own vocabulary, kept where it cannot be mistaken for
                #  the shared contract
                roi.update({f's2p/{k}': v for k, v in roi_stats_all[roi_idx].items()})

                if roi_coordinates is not None:
                    coords = roi_coordinates.iloc[roi_idx]
                    roi.update({'ants/x': float(coords.x), 'ants/y': float(coords.y),
                                'ants/z': float(coords.z)})

                roi['fluorescence'] = fluorescence[roi_idx]
                roi['spikes'] = spikes_all[roi_idx]

                # suite2p packs the classifier's verdict and its confidence into
                #  one two element row. Split, so that a source which classifies
                #  differently - or not at all - writes the same names.
                if iscell_all is not None:
                    roi['is_unit'] = bool(iscell_all[roi_idx][0])
                    roi['unit_probability'] = float(iscell_all[roi_idx][1])

        imaging['frame_times'] = frame_times
        imaging['frame_num'] = frame_times.shape[0]

    def _registration_coordinates(self, path: str, layer_str: str):
        """ANTs-registered ROI coordinates for a plane, if any were computed.

        Two layouts are in use. Registration run per plane names the plane in
        the path; registration run once over the whole extraction names what was
        registered instead, which is the suite2p folder:

            <rec>/suite2p/ants_registration/<plane>/<reference>/mapped_points.h5
            <rec>/ants_registration/suite2p/<reference>/mapped_points.h5

        The second says nothing about planes, so it is only read when there is
        one - otherwise every plane would be handed the same coordinates.
        """
        roots = [os.path.join(path, 'suite2p', 'ants_registration', layer_str)]
        if len(self.layer_names(path)) == 1:
            roots.append(os.path.join(path, 'ants_registration', 'suite2p'))

        for root in roots:
            if not os.path.isdir(long_path(root)):
                continue

            for reference in sorted(os.listdir(long_path(root))):
                mapped_path = long_path(os.path.join(root, reference,
                                                     'mapped_points.h5'))
                if os.path.exists(mapped_path):
                    print(f'>> ANTs ROI coordinates from {mapped_path}')
                    return pd.read_hdf(mapped_path, key='coordinates')

        print(f'>> No ANTs ROI coordinates for {layer_str}')
        return None


class ImagingSpec:
    """One imaging source to ingest, and how to time it.

    `source` and `timing` may be names from the registries or instances, so a
    caller with an unregistered source can pass it directly.
    """

    def __init__(self, source='suite2p', name: str = None, timing=None,
                 frame_avg_num=1, **options):
        self.source = imaging_sources[source] if isinstance(source, str) else source
        self.name = self.source.name if name is None else name
        self.frame_avg_num = frame_avg_num
        self.options = options
        self._timing = timing

    def timing_for(self, animal: Animal, recording: Recording) -> FrameTiming:
        if isinstance(self._timing, FrameTiming):
            return self._timing

        frame_avg_num = self.frame_avg_num
        if callable(frame_avg_num):
            frame_avg_num = frame_avg_num(animal.id, recording.id)

        name = self.source.default_timing if self._timing is None else self._timing
        if name == 'sync_signal':
            return SyncSignalTiming(frame_avg_num=frame_avg_num, **self.options)

        raise ValueError(f'Unknown frame timing "{name}". Pass a FrameTiming instance.')

    def __repr__(self):
        return f'ImagingSpec({self.name}, source={self.source.name})'


imaging_sources = {
    'suite2p': Suite2pSource(),
}


# The files vxpy writes beside one another into a recording folder. Any one of
#  them is enough to tell a recording from whatever else sits in the tree -
#  ants_registration, a zstack, a notes file.
RECORDING_FILE_NAMES = frozenset({'io.hdf5', 'camera.hdf5', 'display.hdf5',
                                  'gui.hdf5'})


def is_recording_folder(path: str) -> bool:
    """Whether `path` looks like a vxpy recording folder."""
    if not os.path.isdir(path):
        return False

    return any(name.lower() in RECORDING_FILE_NAMES for name in os.listdir(path))


def scan_experiment(path: str) -> list[tuple[str, list[str]]]:
    """The animal folders of an experiment folder, and the recordings in each.

    A folder is an animal if it holds at least one recording, which is what
    leaves out anything else sitting at that level. Reading the tree is separate
    from ingesting it, so that a caller can see what a run would do before
    starting one that takes hours:

        for animal_path, recordings in scan_experiment('/data/cmn'):
            print(os.path.basename(animal_path), len(recordings))

    Returns:
        list: one (animal_path, [recording_path, ...]) per animal, in name
            order, holding only animals that have at least one recording.
    """
    contents = []
    for animal_name in sorted(os.listdir(path)):
        animal_path = os.path.join(path, animal_name)
        if not os.path.isdir(animal_path):
            continue

        recordings = [os.path.join(animal_path, name)
                      for name in sorted(os.listdir(animal_path))
                      if is_recording_folder(os.path.join(animal_path, name))]

        if len(recordings) > 0:
            contents.append((animal_path, recordings))

    return contents


class Suite2PVxPy(entarchy.Entarchy):

    # 0.3 put an Imaging entity between Recording and Layer, so that imaging is
    #  one source among possibly several rather than a property of the recording
    # 0.4 put an Experiment above Animal, so that which animals belong together
    #  is a level of the hierarchy rather than a string written on each of them
    _implementation_compat_version_list = ['0.4']
    _implementation_version = '0.4'

    _hierarchy_root_type = Experiment

    def add_experiment(self, path: str, imaging='auto', name: str = None,
                       with_video: bool = True, skip_broken: bool = True,
                       limit: int = None) -> Experiment:
        """Ingest one experiment folder: `<experiment>/<animal>/<recording>`.

        This is the unit the ingest was missing. `add_animal` and `add_recording`
        each take one folder, and everything above them - which folders are
        animals, which of those hold recordings, what to do when one of
        thirty-eight fails halfway through - was left to whoever wrote the loop.

            ent.add_experiment('/data/cmn')
            ent.add_experiment('/data/rot_trans', imaging=ImagingSpec(
                timing=ClockDivisionTiming(7.5, signal='di_frame_sync')))

        The experiment is the right scope for the imaging choice, because that
        choice cannot be detected and does not vary within one: a rig that
        recorded no galvo mirror trace has to be timed from a divided clock whose
        ratio was measured by hand. Whatever is passed here is written onto the
        Experiment under `imaging/`, so the entarchy records how it read its own
        data rather than leaving that in the script that happened to run.

        Re-running continues rather than duplicating, since `add_animal` and
        `add_recording` both skip what is already there. An ingest that stopped
        is resumed by running it again.

        Args:
            path: the experiment folder. Its name becomes the experiment id
                unless `name` says otherwise.
            imaging: as in `add_recording`, applied to every recording here.
            with_video: take the behaviour videos in as media.
            skip_broken: report a folder that fails and carry on rather than
                ending the run. False re-raises, which is what a test wants.
            limit: at most this many recordings per animal. A trial run over
                the whole tree before committing hours to it, since a later
                full run picks up the rest.

        Returns:
            Experiment: the experiment, newly created or already present.
        """
        path = pathlib.Path(path).as_posix()
        if not os.path.isdir(path):
            raise FileNotFoundError(f'No experiment folder {path}')

        if limit is not None and limit < 1:
            raise ValueError(f'limit is a number of recordings per animal and '
                             f'is at least one, got {limit}.')

        contents = scan_experiment(path)
        if limit is not None:
            contents = [(animal_path, recordings[:limit])
                        for animal_path, recordings in contents]

        if len(contents) == 0:
            raise FileNotFoundError(
                f'{path} holds no animal folder with a vxpy recording in it. An '
                f'experiment folder is <experiment>/<animal>/<recording>; one '
                f'animal folder goes to add_animal instead.')

        experiment = self.experiment(os.path.basename(path) if name is None else name)
        self._record_imaging_choice(experiment, imaging, path)

        total = sum(len(recordings) for _, recordings in contents)
        print(f'\nIngest experiment {experiment.id} from {path}')
        print(f'{len(contents)} animals, {total} recordings')

        started = time.time()
        added, skipped, failed = 0, 0, []

        for animal_path, recording_paths in contents:
            animal_id = os.path.basename(animal_path)
            print(f'\n{"#" * 78}\n# {experiment.id} / {animal_id}\n{"#" * 78}')

            try:
                animal = self.add_animal(experiment, animal_path)
            except Exception:
                if not skip_broken:
                    raise
                failed.append((animal_path, traceback.format_exc()))
                print(f'FAILED to add animal {animal_id}:\n{failed[-1][1]}')
                continue

            for recording_path in recording_paths:
                done = added + skipped + len(failed)
                print(f'\n----- {os.path.basename(recording_path)}  '
                      f'[{done}/{total}, {(time.time() - started) / 60:.0f} min] -----')

                try:
                    recording = self.add_recording(animal, recording_path,
                                                   imaging=imaging,
                                                   with_video=with_video)
                except Exception:
                    if not skip_broken:
                        raise
                    failed.append((recording_path, traceback.format_exc()))
                    print(f'FAILED {os.path.basename(recording_path)}:\n{failed[-1][1]}')
                    continue

                if recording is None:
                    skipped += 1
                else:
                    added += 1

        print(f'\n{"=" * 78}')
        print(f'{experiment.id}: added {added}, skipped {skipped} already present, '
              f'{len(failed)} failed, in {(time.time() - started) / 60:.1f} min')

        for failed_path, error in failed:
            print(f'\n--- {failed_path}\n{error}')

        return experiment

    def _record_imaging_choice(self, experiment: Experiment, imaging,
                               path: str) -> None:
        """Write how this experiment's frames were timed onto the experiment.

        Without it the entarchy holds data it cannot say how it interpreted. The
        ratio a divided clock is read at is a measurement, and it used to exist
        only in whichever script happened to run the ingest.
        """
        with self:
            if imaging is None:
                experiment['imaging'] = 'none'
                return

            if isinstance(imaging, str) and imaging == 'auto':
                experiment['imaging'] = 'auto'
                return

            specs = self._imaging_specs(imaging, path)
            experiment['imaging'] = ', '.join(spec.name for spec in specs)

            for spec in specs:
                prefix = f'imaging/{spec.name}'
                experiment[f'{prefix}/source'] = spec.source.name

                # A per-recording frame_avg_num is a function of the animal and
                #  recording ids, so there is no one value to record
                if not callable(spec.frame_avg_num):
                    experiment[f'{prefix}/frame_avg_num'] = spec.frame_avg_num

                timing = spec._timing
                if isinstance(timing, FrameTiming):
                    experiment[f'{prefix}/timing/type'] = type(timing).__name__
                    for key, value in timing.config.items():
                        experiment[f'{prefix}/timing/{key}'] = value
                elif timing is not None:
                    experiment[f'{prefix}/timing/type'] = str(timing)

    def experiment(self, experiment: Experiment | str) -> Experiment:
        """The Experiment of that name, created if there is none yet.

        Idempotent by id, the way `set_current_analysis` is, so an ingest that
        runs twice adds animals to the experiment it made the first time rather
        than to a second one.
        """
        if isinstance(experiment, Experiment):
            return experiment

        if not isinstance(experiment, str):
            raise TypeError(f'An experiment is named by an Experiment or a '
                            f'string, not by {type(experiment).__name__}.')

        existing = self.get(Experiment) & f'id == "{experiment}"'
        if len(existing) > 0:
            return existing[0]

        with self:
            print(f'> Create new entity for experiment {experiment}')
            entity = Experiment(self, _id=experiment, _parent=self.root)
            self.add_new_entity(entity)

        return entity

    @entarchy.digest_method
    def add_animal(self, experiment: Experiment | str, path: str,
                   use_anatomy_reference: str = None) -> Animal:
        """Add one animal folder to an experiment.

        The experiment comes first, as the animal does in `add_recording`: an
        animal belongs to one, and naming it by string creates it if it is new.

            ent.add_animal('cmn', '/data/cmn/2024-08-02_fish1')
        """

        experiment = self.experiment(experiment)
        path = pathlib.Path(path).as_posix()

        print(f'> Add animal from path {path}')

        # Create animal
        path_parts = path.split('/')
        animal_id = path_parts[-1]

        # Scoped to the experiment, so that the same animal id under two of them
        #  is two animals rather than a collision
        animal_collection = self.get(
            Animal, f'id == "{animal_id}" and [Experiment]uuid == "{experiment.uuid}"')

        if len(animal_collection) > 0:
            print(f'WARNING: animal with id {animal_id} already exists in '
                  f'experiment {experiment.id}. Skipping.')
            return animal_collection[0]

        with self:

            # Create new animal entity
            print(f'>> Create new entity for animal {animal_id}')
            animal = Animal(self, _id=animal_id, _parent=experiment)
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
                    if 'Composite.h5' in os.listdir(long_path(reg_path)):
                        valid_reg_path = reg_path
                        break

        # Write registration metadata to animal entity
        if valid_reg_path is not None:
            print(f'Loading ANTs registration metadata at {valid_reg_path}')
            with open(long_path(os.path.join(valid_reg_path, 'metadata.yaml')), 'r') as f:
                ants_metadata = yaml.safe_load(f)
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

    def _read_record_groups(self, path: str):
        """The stimulation phase id per analog sample, on the io timebase.

        This is what says when each phase actually ran, and it needs no
        microscope - which is why it is read here rather than inside an imaging
        source, where it used to be resampled onto calcium frames.
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

    def _ingest_hdf5_files(self, recording: Recording, path: str, record_groups) -> dict:
        """Everything the acquisition software wrote, imaging or not.

        Phases are created here rather than by an imaging source: a stimulation
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

    def _link_phase_frames(self, imaging: Imaging, record_groups) -> int:
        """Record which frames of this source each phase covers.

        A link rather than an attribute on the Phase, because the window belongs
        to the pair - two sources of one recording sample the same phase
        differently. Phase.frames_in() reads it back.
        """
        if record_groups is None:
            return 0

        frame_times = imaging['frame_times']
        ids, times = record_groups
        on_frames = scipy.interpolate.interp1d(times, ids, kind='nearest')(frame_times)

        linked = 0
        for phase in imaging.recording.phases:
            inside = np.where(on_frames == phase['index'])[0]
            if len(inside) == 0:
                print(f'WARNING: phase {phase["index"]} covers no frames of {imaging.id}')
                continue

            self.link(phase, imaging, PHASE_FRAMES_LINK,
                      start_index=int(inside[0]), end_index=int(inside[-1]))
            linked += 1

        return linked

    def _check_roi_contract(self, imaging: Imaging) -> None:
        """Whether a source wrote what analysis is entitled to expect.

        Checked on the first ROI of each layer rather than all of them: a source
        that does not honour the contract fails for every ROI it wrote, and
        reading them all to find that out would cost more than the ingest did.

        Must run after a commit - entities queued in a write context cannot be
        queried, so this would otherwise inspect nothing and pass.
        """
        layers = list(imaging.layers)
        if len(layers) == 0:
            raise RuntimeError(
                f'Imaging source "{imaging.id}" produced no layers.')

        for layer in layers:
            rois = layer.rois
            if len(rois) == 0:
                continue

            missing = [name for name in self.ROI_REQUIRED if name not in rois[0]]
            if len(missing) > 0:
                raise RuntimeError(
                    f'Imaging source "{imaging.id}" produced ROIs without {missing}. '
                    f'Every source must write {list(self.ROI_REQUIRED)} so that '
                    f'analysis can read an ROI without knowing what segmented it.')

    def _imaging_specs(self, imaging, path: str) -> list:
        """Normalise the `imaging` argument into a list of ImagingSpec."""
        if imaging is None:
            return []

        if imaging == 'auto':
            return [ImagingSpec(source=source)
                    for source in imaging_sources.values() if source.detect(path)]

        if isinstance(imaging, (str, ImagingSpec, ImagingSource)):
            imaging = [imaging]

        specs = []
        for item in imaging:
            if isinstance(item, ImagingSpec):
                specs.append(item)
            elif isinstance(item, str):
                if item not in imaging_sources:
                    raise ValueError(
                        f'Unknown imaging source "{item}". '
                        f'Known: {", ".join(sorted(imaging_sources))}.')
                specs.append(ImagingSpec(source=item))
            elif isinstance(item, ImagingSource):
                specs.append(ImagingSpec(source=item))
            else:
                raise TypeError(f'Cannot read {item!r} as an imaging source.')

        return specs

    @entarchy.digest_method
    def add_imaging(self, recording: Recording, source='suite2p', path: str = None,
                    name: str = None, timing=None, frame_avg_num=1,
                    **options) -> Imaging:
        """Add one imaging source to a recording that already exists.

        The recording folder has to be given again rather than remembered: an
        entarchy is self-contained and must not depend on a path outside itself
        for reading, only for provenance.

            ent.add_imaging(recording, 'suite2p', path='/data/fish1/rec_01')
            ent.add_imaging(recording, caiman_source, path=..., name='caiman_v2')

        Returns:
            Imaging: the new source entity.
        """
        if path is None:
            raise ValueError('add_imaging needs the recording folder; an entarchy '
                             'does not keep a path it can read from.')

        spec = source if isinstance(source, ImagingSpec) else ImagingSpec(
            source=source, name=name, timing=timing, frame_avg_num=frame_avg_num,
            **options)

        with self:
            imaging = self._add_imaging(recording, spec, path,
                                        self._read_record_groups(path))

        return imaging

    def _add_imaging(self, recording: Recording, spec: 'ImagingSpec', path: str,
                     record_groups) -> Imaging:
        """Ingest one source. Assumes an open write context."""
        existing = [source.id for source in recording.imaging]
        if spec.name in existing:
            raise ValueError(f'{recording} already has an imaging source named '
                             f'"{spec.name}".')

        layer_names = spec.source.layer_names(path)
        if len(layer_names) == 0:
            raise FileNotFoundError(
                f'Imaging source "{spec.source.name}" found no data in {path}. '
                f'Use imaging="auto" to ingest whatever is there, or None to skip it.')

        print(f'> Add imaging source {spec.name} ({spec.source.name})')
        imaging = Imaging(self, _id=spec.name, _parent=recording)
        self.add_new_entity(imaging)
        imaging['method'] = spec.source.name
        imaging['layer_num'] = len(layer_names)

        timing = spec.timing_for(recording.animal, recording)
        imaging['timing'] = timing.name
        frame_times_by_layer = timing.frame_times(path, len(layer_names))

        spec.source.ingest(imaging, path, frame_times_by_layer, spec.options)

        frame_times = imaging['frame_times']
        imaging['rate'] = 1. / np.diff(frame_times).mean()
        print(f'> Estimated, effective imaging rate {imaging["rate"]:.2f}Hz')

        # Both of the following ask the database what was just created, and
        #  entities queued inside a write context are not visible to a query
        #  until they are written
        self.commit()

        self._check_roi_contract(imaging)
        self._link_phase_frames(imaging, record_groups)

        recording['has_imaging'] = True

        return imaging

    @entarchy.digest_method
    def add_recording(self, animal: Animal, path: str,
                      sync_signal: str = None, sync_signal_time: str = None,
                      sync_type = None, frame_avg_num: int | Callable = 1,
                      with_video: bool = True, imaging='auto') -> Recording | None:
        """Ingest one vxpy recording folder.

        Args:
            imaging: which imaging sources to take signals from. 'auto' ingests
                every registered source that has data in the folder and skips
                imaging entirely when none does; None ingests none, leaving a
                recording of stimulus, io and behaviour data; a name, an
                ImagingSpec, or a list of either ingests those.
            with_video: take the behaviour videos into the entarchy as media.
            sync_signal, sync_signal_time, sync_type, frame_avg_num: how frames
                are timed, for sources that read an analog sync channel. Given
                per source by an ImagingSpec when they differ.
        """

        path = pathlib.Path(path).as_posix()

        # Create recording
        path_parts = path.split('/')
        recording_id = path_parts[-1]

        # Check if recording with same id already exists for this animal
        if len(self.get(Recording, f'id == "{recording_id}" and [Animal]uuid == "{animal.uuid}"')) > 0:
            print(f'WARNING: recording with id {recording_id} already exists for animal {animal.id}. Skipping.')
            return None

        if not is_recording_folder(path):
            print(f'WARNING: {path} does not appear to be vxpy recording folder. Skipping.')
            return None

        specs = self._imaging_specs(imaging, path)

        # The old signature named one sync channel for the whole recording; keep
        #  it working by folding it into the specs that did not say otherwise
        for spec in specs:
            if spec._timing is None and (sync_signal is not None or sync_type is not None):
                spec._timing = SyncSignalTiming(
                    method='y_mirror' if sync_type is None else sync_type,
                    signal='ai_y_mirror_in' if sync_signal is None else sync_signal,
                    signal_time=sync_signal_time,
                    frame_avg_num=frame_avg_num if spec.frame_avg_num == 1 else spec.frame_avg_num)
            elif spec.frame_avg_num == 1 and frame_avg_num != 1:
                spec.frame_avg_num = frame_avg_num

        if len(specs) > 0 and not os.path.exists(os.path.join(path, 'Io.hdf5')):
            raise FileNotFoundError(
                f'{path} has imaging output but no Io.hdf5, so its frames cannot '
                f'be timed. Pass imaging=None to ingest the rest of it.')

        print(f'Process recording folder {path}')

        with self:

            # When each stimulation phase ran, on the io timebase. Needed whether
            #  or not there is imaging, so it is read before any source is.
            record_groups = self._read_record_groups(path)

            # Create new recording entity
            recording = Recording(self, _id=recording_id, _parent=animal)
            self.add_new_entity(recording)
            print(f'> Create {recording}')

            # Add metadata
            add_metadata(recording, path)
            recording['has_imaging'] = False

            self._ingest_hdf5_files(recording, path, record_groups)

            if with_video:
                self._add_recording_videos(recording, path)

            for spec in specs:
                self._add_imaging(recording, spec, path, record_groups)

        return recording

    @entarchy.digest_method
    def update_roi_coordinates_from_registration(self, recording_path: str,
                                                 imaging: str = None):
        """Refresh ROI coordinates from ANTs registration output.

        Registration is computed about an extraction rather than being part of
        one, so it can be re-run and written back afterwards. `imaging` names
        which source's ROIs to update; with one source it can be left out.
        """
        parts = pathlib.Path(recording_path).as_posix().split('/')
        recording = self.get(Recording,
                             f'[Animal]id == "{parts[-2]}" AND id == "{parts[-1]}"')[0]

        source_entity = (recording.sole_imaging() if imaging is None
                         else recording.imaging[imaging])
        source = imaging_sources.get(source_entity['method'])
        if source is None:
            raise ValueError(
                f'No registered source named "{source_entity["method"]}", so its '
                f'registration output cannot be located.')

        for layer in source_entity.layers:
            roi_coordinates = source._registration_coordinates(recording_path, layer.id)

            if roi_coordinates is None:
                continue

            rois = layer.rois
            for k in ['x', 'y', 'z']:
                rois[f'ants/{k}'] = rois['index'].apply(
                    lambda idx: roi_coordinates.iloc[idx][k])

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


# Where Windows starts refusing paths. Directories cap lower than files, so one
#  threshold below both is what decides when to ask for the long form.
_PATH_LIMIT = 240


def long_path(path: str) -> str:
    """A path Windows will open past its 260 character limit.

    ANTs registration output nests a reference image name inside a moving image
    name, both long and descriptive, so what sits beside them can land a
    character or two over: `2024-08-02_fish1` has its registration metadata at
    261. The failure is quiet and misleading - listing the folder still works,
    while opening a file inside it raises a bare FileNotFoundError - so this is
    applied wherever the ingest reaches into registration output.

    Returns the path unchanged off Windows, and where it is short enough that
    the plain form is the better-trodden one.
    """
    if os.name != 'nt':
        return path

    absolute = os.path.abspath(path)
    if len(absolute) < _PATH_LIMIT or absolute.startswith('\\\\'):
        return absolute

    return f'\\\\?\\{absolute}'


# What a folder may say about itself. Setups differ in what they call the
#  sidecar they write - a recording depth is under `info.yaml` in one and
#  `metadata.yaml` in another - so both are read and neither is lost.
METADATA_FILE_NAMES = ('metadata.yaml', 'info.yaml')


def add_metadata(entity: entarchy.Entity, folder_path: str):
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (see
    `METADATA_FILE_NAMES`) and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path)
                  if f.endswith(METADATA_FILE_NAMES)]

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
    """Frame times from the rising edges of a digital frame sync line.

    A digital line is recorded as bool, and np.diff of a bool array is XOR: a
    fall reads as True just as a rise does, so every frame would be counted
    twice. Widening to a signed type first makes a fall negative again.

    np.diff reports the gap between samples i and i+1, so the rising sample -
    the first one recorded while the frame was being acquired - is i+1.
    """
    frame_indices = np.where(np.diff(sync_signal.astype(np.int8)) > 0)[0] + 1
    frame_times = sync_time[frame_indices]

    return frame_indices, frame_times


frame_time_methods = {
    'y_mirror': ca_frame_times_from_y_mirror,
    'frame_sync_toggle': ca_frame_times_from_sync_toggle,
}


if __name__ == '__main__':
    pass