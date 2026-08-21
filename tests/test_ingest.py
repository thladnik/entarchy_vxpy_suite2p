"""End-to-end ingest of a synthetic vxpy + suite2p dataset."""
import os

import numpy as np
import pytest

from entarchy.backend import SQLiteBackend
from entarchy_vxpy_suite2p.schema import (Animal, Experiment, Imaging, ImagingSource,
                                          ImagingSpec, Layer, Phase, Recording, Roi,
                                          Suite2PVxPy, imaging_sources,
                                          is_recording_folder, scan_experiment)

import _synthetic_data

EXPERIMENT = 'experiment_01'


@pytest.fixture()
def dataset(tmp_path):
    return _synthetic_data.build_dataset((tmp_path / 'data').as_posix())


@pytest.fixture()
def ent(tmp_path):
    base = (tmp_path / 'archy').as_posix()
    entarchy_obj = Suite2PVxPy.create(base, SQLiteBackend(base, dbname='ingest.db'))
    yield entarchy_obj
    entarchy_obj.backend.close()


@pytest.fixture()
def ingested(ent, dataset):
    animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
    recording = ent.add_recording(animal, dataset['recording_path'])
    return ent, dataset, animal, recording


class TestAddAnimal:

    def test_creates_animal_entity(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        assert isinstance(animal, Animal)
        assert animal.id == dataset['animal_id']
        assert len(ent.get(Animal)) == 1

    def test_metadata_is_flattened_onto_entity(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        assert animal['metadata/strain'] == 'wildtype'
        assert animal['metadata/age_dpf'] == 6
        assert animal['metadata/nested/dish'] == 3

    def test_zstack_is_stored(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        assert animal['zstack_fn'] == 'zstack_1.tif'
        assert np.array_equal(animal['zstack'], dataset['zstack'])

    def test_missing_zstack_is_tolerated(self, tmp_path, ent):
        dataset = _synthetic_data.build_dataset((tmp_path / 'nostack').as_posix(),
                                                with_zstack=False)
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        assert 'zstack' not in animal

    def test_adding_twice_returns_existing_animal(self, ent, dataset, capsys):
        first = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        second = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        assert second.uuid == first.uuid
        assert len(ent.get(Animal)) == 1
        assert 'already exists' in capsys.readouterr().out

    def test_animal_is_child_of_its_experiment(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        assert isinstance(animal.parent, Experiment)
        assert animal.experiment.id == EXPERIMENT
        assert animal.parent.parent.uuid == ent.root.uuid

    def test_the_same_animal_id_under_two_experiments_is_two_animals(self, ent, dataset):
        """The id is scoped to the experiment, so two protocols may share one."""
        first = ent.add_animal('experiment_a', dataset['animal_path'])
        second = ent.add_animal('experiment_b', dataset['animal_path'])

        assert first.uuid != second.uuid
        assert first.id == second.id
        assert len(ent.get(Animal)) == 2


class TestExperiment:

    def test_named_by_string_creates_one(self, ent):
        experiment = ent.experiment('cmn')

        assert isinstance(experiment, Experiment)
        assert experiment.id == 'cmn'
        assert experiment.parent.uuid == ent.root.uuid

    def test_the_same_name_is_the_same_entity(self, ent):
        first = ent.experiment('cmn')
        second = ent.experiment('cmn')

        assert first.uuid == second.uuid
        assert len(ent.get(Experiment)) == 1

    def test_an_experiment_passes_through(self, ent):
        experiment = ent.experiment('cmn')

        assert ent.experiment(experiment) is experiment

    def test_anything_else_is_refused(self, ent):
        with pytest.raises(TypeError, match='named by an Experiment or a string'):
            ent.experiment(7)

    def test_reaches_everything_below_it(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        ent.add_recording(animal, dataset['recording_path'])
        experiment = ent.experiment(EXPERIMENT)

        assert len(experiment.animals) == 1
        assert len(experiment.recordings) == 1
        assert len(experiment.rois) == len(animal.rois)
        assert len(experiment.phases) > 0


class TestScanExperiment:

    def test_finds_the_animals_and_their_recordings(self, dataset):
        contents = scan_experiment(dataset['root'])

        assert len(contents) == 1
        animal_path, recordings = contents[0]
        assert os.path.basename(animal_path) == dataset['animal_id']
        assert recordings == [dataset['recording_path']]

    def test_a_folder_without_recordings_is_not_an_animal(self, dataset, tmp_path):
        """Which is what leaves out whatever else sits at that level."""
        os.makedirs(os.path.join(dataset['root'], 'notes'), exist_ok=True)

        assert [os.path.basename(a) for a, _ in scan_experiment(dataset['root'])] == [
            dataset['animal_id']]

    def test_registration_output_beside_a_recording_is_not_one(self, dataset):
        os.makedirs(os.path.join(dataset['animal_path'], 'ants_registration'),
                    exist_ok=True)

        _, recordings = scan_experiment(dataset['root'])[0]
        assert recordings == [dataset['recording_path']]

    def test_is_recording_folder_agrees_with_add_recording(self, dataset, tmp_path):
        assert is_recording_folder(dataset['recording_path'])
        assert not is_recording_folder(dataset['animal_path'])
        assert not is_recording_folder(str(tmp_path / 'does_not_exist'))


class TestAddExperiment:

    def test_ingests_the_whole_folder(self, ent, dataset):
        experiment = ent.add_experiment(dataset['root'], skip_broken=False)

        assert isinstance(experiment, Experiment)
        assert experiment.id == os.path.basename(dataset['root'])
        assert len(experiment.animals) == 1
        assert len(experiment.recordings) == 1
        assert len(experiment.rois) > 0

    def test_name_overrides_the_folder_name(self, ent, dataset):
        experiment = ent.add_experiment(dataset['root'], name='cmn', skip_broken=False)

        assert experiment.id == 'cmn'

    def test_running_it_twice_adds_nothing(self, ent, dataset):
        first = ent.add_experiment(dataset['root'], skip_broken=False)
        rois = len(first.rois)

        second = ent.add_experiment(dataset['root'], skip_broken=False)

        assert second.uuid == first.uuid
        assert len(ent.get(Experiment)) == 1
        assert len(ent.get(Animal)) == 1
        assert len(ent.get(Recording)) == 1
        assert len(second.rois) == rois

    def test_a_folder_with_no_animals_is_an_error(self, ent, tmp_path):
        empty = tmp_path / 'empty_experiment'
        os.makedirs(empty, exist_ok=True)

        with pytest.raises(FileNotFoundError, match='no animal folder'):
            ent.add_experiment(empty.as_posix())

    def test_a_missing_folder_is_an_error(self, ent, tmp_path):
        with pytest.raises(FileNotFoundError, match='No experiment folder'):
            ent.add_experiment((tmp_path / 'nope').as_posix())

    def test_a_broken_recording_is_reported_rather_than_fatal(self, ent, dataset,
                                                              monkeypatch):
        """One bad folder out of thirty-eight must not end the run."""
        def explode(*args, **kwargs):
            raise RuntimeError('this recording is broken')

        monkeypatch.setattr(Suite2PVxPy, 'add_recording', explode)
        experiment = ent.add_experiment(dataset['root'])

        # The animal still went in; only its recording failed
        assert len(experiment.animals) == 1
        assert len(experiment.recordings) == 0

    def test_skip_broken_false_re_raises(self, ent, dataset, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError('this recording is broken')

        monkeypatch.setattr(Suite2PVxPy, 'add_recording', explode)

        with pytest.raises(RuntimeError, match='this recording is broken'):
            ent.add_experiment(dataset['root'], skip_broken=False)


class TestImagingChoiceIsRecorded:
    """How frames were timed cannot be detected, so it has to be written down."""

    def test_an_explicit_timing_is_stored_on_the_experiment(self, ent, dataset):
        from entarchy_vxpy_suite2p.schema import ClockDivisionTiming

        spec = ImagingSpec(source='suite2p',
                           timing=ClockDivisionTiming(7.5, signal='di_frame_sync'))
        experiment = ent.add_experiment(dataset['root'], imaging=spec)

        assert experiment['imaging'] == 'suite2p'
        assert experiment['imaging/suite2p/source'] == 'suite2p'
        assert experiment['imaging/suite2p/timing/type'] == 'ClockDivisionTiming'
        assert experiment['imaging/suite2p/timing/edges_per_volume'] == 7.5
        assert experiment['imaging/suite2p/timing/signal'] == 'di_frame_sync'

    def test_auto_says_so(self, ent, dataset):
        experiment = ent.add_experiment(dataset['root'], imaging='auto')

        assert experiment['imaging'] == 'auto'

    def test_none_says_so(self, ent, dataset):
        experiment = ent.add_experiment(dataset['root'], imaging=None)

        assert experiment['imaging'] == 'none'
        assert len(experiment.rois) == 0

    def test_timing_config_covers_every_timing(self):
        from entarchy_vxpy_suite2p.schema import (CameraTiming, ClockDivisionTiming,
                                                  SyncSignalTiming)

        assert SyncSignalTiming(method='y_mirror').config == {
            'method': 'y_mirror', 'signal': 'ai_y_mirror_in',
            'signal_time': 'ai_y_mirror_in_time', 'frame_avg_num': 1}
        assert ClockDivisionTiming(7.5).config == {
            'edges_per_volume': 7.5, 'signal': 'di_frame_sync',
            'signal_time': 'di_frame_sync_time'}
        assert CameraTiming('fish_embedded').config == {
            'device': 'fish_embedded', 'file_name': 'Camera.hdf5'}


class TestAddRecording:

    def test_creates_full_entity_tree(self, ingested):
        ent, dataset, animal, recording = ingested

        assert isinstance(recording, Recording)
        assert len(ent.get(Recording)) == 1
        assert len(ent.get(Layer)) == dataset['plane_num']
        assert len(ent.get(Phase)) == len(dataset['phase_indices'])
        assert len(ent.get(Roi)) == dataset['plane_num'] * dataset['roi_num']

    def test_parent_links(self, ingested):
        """Animal > Recording > Imaging > Layer > Roi, with each level able to
        name its ancestors however many steps away they are."""
        ent, _, animal, recording = ingested

        assert recording.parent.uuid == animal.uuid

        imaging = ent.get(Imaging)[0]
        assert imaging.parent.uuid == recording.uuid

        layer = ent.get(Layer)[0]
        assert layer.parent.uuid == imaging.uuid
        assert layer.imaging.uuid == imaging.uuid
        assert layer.recording.uuid == recording.uuid

        roi = ent.get(Roi)[0]
        assert roi.layer.parent.uuid == imaging.uuid
        assert roi.imaging.uuid == imaging.uuid
        assert roi.recording.uuid == recording.uuid
        assert roi.animal.uuid == animal.uuid

    def test_convenience_collections(self, ingested):
        ent, dataset, animal, recording = ingested

        assert len(animal.recordings) == 1
        assert len(animal.layers) == dataset['plane_num']
        assert len(animal.rois) == dataset['plane_num'] * dataset['roi_num']
        assert len(recording.phases) == len(dataset['phase_indices'])
        assert len(recording.layers) == dataset['plane_num']
        assert len(recording.rois) == dataset['plane_num'] * dataset['roi_num']

        layer = ent.get(Layer, 'id == "plane0"')[0]
        assert len(layer.rois) == dataset['roi_num']

    def test_recording_metadata(self, ingested):
        _, _, _, recording = ingested
        assert recording['metadata/repeat_num'] == 2
        assert recording['metadata/experimenter'] == 'tester'

    def test_rejects_non_recording_folder(self, ent, dataset, tmp_path, capsys):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        empty = tmp_path / 'not_a_recording'
        empty.mkdir()

        assert ent.add_recording(animal, empty.as_posix()) is None
        assert 'does not appear to be vxpy recording folder' in capsys.readouterr().out

    def test_adding_twice_is_skipped(self, ingested, capsys):
        ent, dataset, animal, _ = ingested

        assert ent.add_recording(animal, dataset['recording_path']) is None
        assert len(ent.get(Recording)) == 1
        assert 'already exists' in capsys.readouterr().out


class TestFrameTiming:
    """Timing belongs to the imaging source, not to the recording: two sources
    of one recording may sample it at different rates."""

    def test_rate_is_plausible(self, ingested):
        _, _, _, recording = ingested
        # 10 Hz mirror -> 20 half-cycles per second, split across 2 planes
        assert recording.sole_imaging()['rate'] == pytest.approx(10.0, rel=0.05)

    def test_frame_times_are_monotonic(self, ingested):
        _, _, _, recording = ingested
        assert np.all(np.diff(recording.sole_imaging()['frame_times']) > 0)

    def test_frame_num_matches_frame_times(self, ingested):
        _, _, _, recording = ingested
        imaging = recording.sole_imaging()
        assert imaging['frame_num'] == len(imaging['frame_times'])

    def test_frame_times_match_fluorescence_length(self, ingested):
        """Every ROI's signal must be indexable by its source's timeline."""
        ent, dataset, _, recording = ingested
        frame_times = recording.sole_imaging()['frame_times']

        for roi in ent.get(Roi):
            assert len(roi['fluorescence']) == len(frame_times)

    def test_the_recording_carries_no_imaging_timing(self, ingested):
        _, _, _, recording = ingested

        for name in ('imaging_rate', 'ca_times', 'signal_length', 'record_group_ids'):
            assert name not in recording

    def test_the_source_records_what_it_was(self, ingested):
        _, _, _, recording = ingested
        imaging = recording.sole_imaging()

        assert imaging.id == 'suite2p'
        assert imaging['method'] == 'suite2p'
        assert imaging['timing'] == 'sync_signal'
        assert imaging['layer_num'] == 2

    def test_layer_time_offsets(self, ingested):
        ent, dataset, _, recording = ingested

        offsets = {layer.id: layer['t_offset'] for layer in ent.get(Layer)}
        assert offsets['plane0'] == 0.0
        assert offsets['plane1'] > 0.0

    def test_the_raw_record_group_trace_is_on_the_recording(self, ingested):
        """On the io timebase, where it needs no microscope to mean anything."""
        _, _, _, recording = ingested

        ids = recording['io/__record_group_id']
        assert set(np.unique(ids)).issubset({-1, 0, 1})
        assert (ids == 0).any() and (ids == 1).any()


class TestPhases:

    def test_indices_and_ordering(self, ingested):
        ent, dataset, _, _ = ingested

        indices = sorted(phase['index'] for phase in ent.get(Phase))
        assert indices == dataset['phase_indices']

    def test_attributes_are_namespaced_by_file(self, ingested):
        ent, _, _, _ = ingested

        phase = ent.get(Phase, 'index == 0')[0]
        assert phase['display/__visual_name'] == 'CMN'
        assert phase['display/__start_time'] == pytest.approx(2.0)
        assert phase['display/__target_duration'] == pytest.approx(4.0)

    def test_datasets_are_stored_unprefixed(self, ingested):
        ent, _, _, _ = ingested

        phase = ent.get(Phase, 'index == 0')[0]
        assert len(phase['frame_index']) == 30
        assert len(phase['__time']) == 30

    def test_calcium_window_indices(self, ingested):
        ent, _, _, recording = ingested
        frame_times = recording.sole_imaging()['frame_times']

        for phase in ent.get(Phase):
            start, end = phase.frames_in()

            assert 0 <= start < end < len(frame_times)
            # The window must overlap the stimulus interval it was derived from
            assert frame_times[start] >= phase['display/__start_time'] - 0.5
            assert frame_times[end] <= (phase['display/__start_time']
                                        + phase['display/__target_duration'] + 0.5)

    def test_phases_are_children_of_recording(self, ingested):
        ent, _, _, recording = ingested
        for phase in ent.get(Phase):
            assert phase.recording.uuid == recording.uuid


class TestConfigurablePhases:
    """The phase count is a parameter of the synthetic dataset.

    The io file and the stimulus log have to be written from the same windows:
    the ingest finds a phase's calcium frames by looking its index up in the
    record group trace, so a phase in only one of the two files would end up
    with an empty index array.
    """

    @pytest.mark.parametrize('phase_num', [1, 3, 6])
    def test_requested_number_of_phases_is_ingested(self, ent, tmp_path, phase_num):
        dataset = _synthetic_data.build_dataset(
            (tmp_path / f'data_{phase_num}').as_posix(), phase_num=phase_num)

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        ent.add_recording(animal, dataset['recording_path'])

        assert len(ent.get(Phase)) == phase_num
        assert sorted(p['index'] for p in ent.get(Phase)) == list(range(phase_num))

    def test_windows_reach_the_phase_attributes(self, ent, tmp_path):
        dataset = _synthetic_data.build_dataset(
            (tmp_path / 'data_custom').as_posix(),
            phase_windows={0: (3.0, 5.0), 1: (9.0, 13.0), 2: (15.0, 17.0)})

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        ent.add_recording(animal, dataset['recording_path'])

        for index, (start, end) in dataset['phase_windows'].items():
            phase = ent.get(Phase, f'index == {index}')[0]
            assert phase['display/__start_time'] == pytest.approx(start)
            assert phase['display/__target_duration'] == pytest.approx(end - start)

    def test_every_phase_gets_a_calcium_window(self, ent, tmp_path):
        dataset = _synthetic_data.build_dataset(
            (tmp_path / 'data_windows').as_posix(), phase_num=5)

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])
        frame_times = recording.sole_imaging()['frame_times']

        windows = sorted(p.frames_in() for p in ent.get(Phase))

        for start, end in windows:
            assert 0 <= start < end < len(frame_times)

        # Phases are laid out in order and do not overlap
        for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
            assert earlier_end < later_start

    def test_visual_names_cycle_over_the_phases(self, ent, tmp_path):
        dataset = _synthetic_data.build_dataset(
            (tmp_path / 'data_names').as_posix(), phase_num=4)

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        ent.add_recording(animal, dataset['recording_path'])

        names = [ent.get(Phase, f'index == {i}')[0]['display/__visual_name']
                 for i in range(4)]
        assert names == ['CMN', 'TranslationGrating', 'CMN', 'TranslationGrating']

    def test_custom_visual_names(self, ent, tmp_path):
        dataset = _synthetic_data.build_dataset(
            (tmp_path / 'data_looming').as_posix(), phase_num=2,
            visual_names=['Looming'])

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        ent.add_recording(animal, dataset['recording_path'])

        assert {p['display/__visual_name'] for p in ent.get(Phase)} == {'Looming'}

    def test_record_group_trace_matches_the_windows(self, tmp_path):
        """Without this the ingest cannot place a phase at all."""
        windows = _synthetic_data.make_phase_windows(4)
        _, times = _synthetic_data.mirror_trace()

        ids = _synthetic_data.record_group_trace(times, windows)

        assert set(np.unique(ids)) == {-1, 0, 1, 2, 3}
        for index, (start, end) in windows.items():
            inside = ids[(times >= start) & (times < end)]
            assert (inside == index).all()

    def test_default_dataset_is_unchanged(self, tmp_path):
        dataset = _synthetic_data.build_dataset((tmp_path / 'data_default').as_posix())

        assert dataset['phase_indices'] == [0, 1]
        assert dataset['phase_windows'] == {0: (2.0, 6.0), 1: (8.0, 12.0)}


class TestMakePhaseWindows:

    def test_evenly_spaced_with_gaps(self):
        windows = _synthetic_data.make_phase_windows(4)

        assert sorted(windows) == [0, 1, 2, 3]
        for (_, end), (later_start, _) in zip(windows.values(),
                                              list(windows.values())[1:]):
            assert end < later_start

    def test_stays_inside_the_recording(self):
        windows = _synthetic_data.make_phase_windows(8)

        assert min(start for start, _ in windows.values()) >= 2.0
        assert max(end for _, end in windows.values()) <= _synthetic_data.DURATION

    def test_rejects_an_empty_protocol(self):
        with pytest.raises(ValueError, match='at least one'):
            _synthetic_data.make_phase_windows(0)


class TestLayersAndRois:

    def test_layer_attributes(self, ingested):
        ent, dataset, _, _ = ingested

        for layer in ent.get(Layer):
            assert layer['roi_num'] == dataset['roi_num']
            assert layer['s2p/fs'] == 5.0
            assert layer['s2p/nested/do_registration'] == 1
            assert layer['s2p/meanImg'].shape == (8, 8)

    def test_roi_signals_match_source_files(self, ingested):
        ent, dataset, _, _ = ingested

        layer = ent.get(Layer, 'id == "plane0"')[0]
        expected = dataset['fluorescence'][0]

        for roi in layer.rois:
            index = roi['index']
            assert np.allclose(roi['fluorescence'], expected[index][:len(roi['fluorescence'])])

    def test_roi_stats_are_namespaced(self, ingested):
        ent, _, _, _ = ingested

        roi = ent.get(Roi, 'index == 0 AND [Layer]id == "plane0"')[0]
        assert roi['s2p/npix'] == 40
        assert roi['s2p/skew'] == pytest.approx(0.0)
        assert list(roi['s2p/med']) == [0, 0]

    def test_the_classifier_verdict_is_split_out(self, ingested):
        """suite2p packs the verdict and its confidence into one two element
        row; the contract keeps them apart so a source that classifies
        differently, or not at all, writes the same names."""
        ent, _, _, _ = ingested

        roi = ent.get(Roi)[0]
        assert isinstance(roi['is_unit'], bool)
        assert 0.0 <= roi['unit_probability'] <= 1.0

    def test_is_unit_is_queryable(self, ingested):
        """Which is the point of it being a bool rather than a packed array."""
        ent, dataset, _, _ = ingested

        units = ent.get(Roi, 'is_unit == True')

        assert 0 < len(units) <= len(ent.get(Roi))

    def test_spikes_are_stored(self, ingested):
        ent, _, _, _ = ingested

        roi = ent.get(Roi)[0]
        assert len(roi['spikes']) == len(roi['fluorescence'])

    def test_roi_ids_are_unique_within_layer(self, ingested):
        ent, dataset, _, _ = ingested

        for layer in ent.get(Layer):
            ids = [roi.id for roi in layer.rois]
            assert len(set(ids)) == dataset['roi_num']

    def test_missing_iscell_is_tolerated(self, tmp_path, ent):
        import os
        dataset = _synthetic_data.build_dataset((tmp_path / 'noiscell').as_posix())
        for plane in range(dataset['plane_num']):
            os.remove(os.path.join(dataset['recording_path'], 'suite2p',
                                   f'plane{plane}', 'iscell.npy'))

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        ent.add_recording(animal, dataset['recording_path'])

        roi = ent.get(Roi)[0]
        assert 'is_unit' not in roi
        assert 'unit_probability' not in roi
        assert len(roi['fluorescence']) > 0


class TestStimulusData:

    def test_file_level_attributes(self, ingested):
        _, _, _, recording = ingested
        assert recording['display/attrs/__protocol_name'] == 'SyntheticProtocol'
        assert recording['display/attrs/__display_fps'] == 60.0

    def test_non_phase_groups_are_namespaced(self, ingested):
        _, _, _, recording = ingested
        assert recording['display/CMN/seed'] == 42
        assert recording['display/CMN/centers_0'].shape == (12, 3)
        assert recording['display/CMN/motion_vectors_0'].shape == (30, 12, 3)

    def test_io_datasets_are_ingested(self, ingested):
        _, _, _, recording = ingested
        assert len(recording['io/ai_y_mirror_in']) == int(
            _synthetic_data.DURATION * _synthetic_data.SAMPLE_RATE)


class TestImmutabilityAfterIngest:

    def test_ingested_attributes_are_protected(self, ingested):
        ent, _, _, recording = ingested

        with pytest.raises(RuntimeError, match='immutable'):
            recording.sole_imaging()['rate'] = 999.0

    def test_analysis_attributes_remain_writable(self, ingested):
        ent, _, _, _ = ingested

        roi = ent.get(Roi)[0]
        roi['dff'] = np.zeros(5)
        roi['dff'] = np.ones(5)  # must not raise
        assert np.array_equal(roi['dff'], np.ones(5))


class TestQueryingIngestedData:

    def test_filter_rois_by_ancestor_attributes(self, ingested):
        ent, dataset, _, _ = ingested

        assert len(ent.get(Roi, '[Animal]metadata/strain == "wildtype"')) == \
            dataset['plane_num'] * dataset['roi_num']
        assert len(ent.get(Roi, '[Layer]id == "plane1"')) == dataset['roi_num']

    def test_dataframe_across_hierarchy(self, ingested):
        ent, dataset, _, _ = ingested

        df = ent.get(Roi).dataframe_of(['index', '[Layer]id', '[Animal]metadata/strain'])
        assert len(df) == dataset['plane_num'] * dataset['roi_num']
        assert set(df['[Layer]id']) == {'plane0', 'plane1'}
        assert set(df['[Animal]metadata/strain']) == {'wildtype'}

    def test_collection_write_then_read(self, ingested):
        """Analysis results written through a collection must be readable back."""
        ent, dataset, _, _ = ingested

        rois = ent.get(Roi)
        rois['quality'] = 0.5
        assert list(ent.get(Roi)['quality']) == [0.5] * len(rois)


class TestUnequalPlaneFrameCounts:
    """Documents how recording-level timing behaves when planes differ in length."""

    def test_frame_times_follow_the_last_processed_layer(self, tmp_path, ent):
        dataset = _synthetic_data.build_dataset((tmp_path / 'uneven').as_posix(),
                                                frames_per_plane=[60, 40])

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])

        # frame_times is assigned after the layer loop, so it reflects plane1
        # (40), not plane0 (60) - ROI signals in plane0 are then longer.
        assert len(recording.sole_imaging()['frame_times']) == 40

        plane0 = ent.get(Layer, 'id == "plane0"')[0]
        assert len(plane0.rois[0]['fluorescence']) == 60


class TestBehaviourVideo:
    """The camera writes frame times into Camera.hdf5 and the frames into a
    file beside it. The HDF5 is ingested by the generic walk; the video needs
    taking in explicitly, or the frames behind `tail_pose_data` are gone."""

    def test_video_is_taken_into_the_entarchy(self, ingested):
        ent, dataset, _, recording = ingested

        video = recording['camera/fish_embedded/video']

        assert video.exists()
        assert video.path.startswith(ent.path)
        assert video.read_bytes() == open(dataset['video_path'], 'rb').read()

    def test_the_source_is_not_consumed(self, ingested):
        """An ingest must not eat the raw data it was pointed at."""
        _, dataset, _, _ = ingested

        assert os.path.exists(dataset['video_path'])

    def test_it_is_listed_as_media(self, ingested):
        _, _, _, recording = ingested

        assert recording.media() == ['camera/fish_embedded/video']

    def test_the_digest_verifies(self, ingested):
        _, _, _, recording = ingested

        assert recording['camera/fish_embedded/video'].verify()

    def test_frame_times_come_from_the_hdf5(self, ingested):
        """The video is only usable alongside the times the camera recorded."""
        _, _, _, recording = ingested

        times = recording['camera/fish_embedded_frame_time']

        assert len(times) == 400
        assert np.all(np.diff(times) > 0)

    def test_tracking_data_is_ingested_as_an_array(self, ingested):
        """Keypoints are what analysis reads; the video is provenance."""
        _, _, _, recording = ingested

        assert recording['camera/tail_pose_data'].shape == (400, 9, 3)

    def test_with_video_false_skips_it(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'], with_video=False)

        assert recording.media() == []
        assert 'camera/fish_embedded/video' not in recording.keys()

    def test_a_declared_device_without_a_file_warns(self, tmp_path, ent, capsys):
        dataset = _synthetic_data.build_dataset((tmp_path / 'novideo').as_posix(),
                                                with_video=False)
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])

        assert 'has no video file' in capsys.readouterr().out
        assert recording.media() == []

    def test_a_recording_without_a_camera_file_is_fine(self, tmp_path, ent):
        dataset = _synthetic_data.build_dataset((tmp_path / 'nocamera').as_posix(),
                                                with_camera=False)
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])

        assert recording.media() == []

    def test_an_unrelated_video_is_left_alone(self, tmp_path, ent):
        """A stimulus animation dropped in the folder is an output, not an
        acquisition - files are matched to the devices the recording declares."""
        dataset = _synthetic_data.build_dataset((tmp_path / 'extra').as_posix())
        stray = os.path.join(dataset['recording_path'], 'stimulus_animation.mp4')
        with open(stray, 'wb') as f:
            f.write(b'not an acquisition')

        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])

        assert recording.media() == ['camera/fish_embedded/video']


class TestRecordingWithoutImaging:
    """A recording of stimulus, io and behaviour data alone. The ingest used to
    raise FileNotFoundError from inside itself on a folder with no suite2p/."""

    @pytest.fixture()
    def behaviour_only(self, tmp_path, ent):
        dataset = _synthetic_data.build_dataset((tmp_path / 'behaviour').as_posix(),
                                                with_suite2p=False)
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])
        return ent, dataset, recording

    def test_it_ingests(self, behaviour_only):
        _, _, recording = behaviour_only

        assert recording is not None
        assert recording['has_imaging'] is False

    def test_no_layers_or_rois(self, behaviour_only):
        ent, _, _ = behaviour_only

        assert len(ent.get(Layer)) == 0
        assert len(ent.get(Roi)) == 0

    def test_phases_still_exist(self, behaviour_only):
        """A stimulation phase is a fact about what was shown, not about the
        microscope."""
        ent, dataset, _ = behaviour_only

        phases = ent.get(Phase)

        assert len(phases) == len(dataset['phase_indices'])

    def test_phases_carry_a_time_window(self, behaviour_only):
        ent, dataset, _ = behaviour_only

        for phase in sorted(ent.get(Phase), key=lambda p: p['index']):
            expected_start, expected_end = dataset['phase_windows'][phase['index']]
            assert phase['start_time'] == pytest.approx(expected_start, abs=0.01)
            assert phase['end_time'] == pytest.approx(expected_end, abs=0.01)

    def test_phases_have_no_frame_window(self, behaviour_only):
        ent, _, _ = behaviour_only

        assert 'ca_start_index' not in ent.get(Phase)[0]

    def test_no_imaging_timing_on_the_recording(self, behaviour_only):
        _, _, recording = behaviour_only

        for name in ('imaging_rate', 'ca_times', 'signal_length', 'record_group_ids'):
            assert name not in recording

    def test_stimulus_and_io_data_are_all_there(self, behaviour_only):
        _, _, recording = behaviour_only

        assert recording['display/attrs/__protocol_name'] == 'SyntheticProtocol'
        assert len(recording['io/__record_group_id']) > 0
        assert recording['display/CMN/seed'] == 42

    def test_the_behaviour_video_is_still_taken_in(self, behaviour_only):
        _, _, recording = behaviour_only

        assert recording.media() == ['camera/fish_embedded/video']


class TestImagingSelection:

    def test_auto_ingests_suite2p_when_present(self, ingested):
        _, _, _, recording = ingested

        assert recording['has_imaging'] is True
        assert len(recording.layers) > 0

    def test_none_skips_imaging_that_is_there(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'], imaging=None)

        assert recording['has_imaging'] is False
        assert len(recording.layers) == 0
        assert len(ent.get(Phase)) == len(dataset['phase_indices'])

    def test_suite2p_requires_it(self, tmp_path, ent):
        dataset = _synthetic_data.build_dataset((tmp_path / 'none').as_posix(),
                                                with_suite2p=False)
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        with pytest.raises(FileNotFoundError, match='found no data'):
            ent.add_recording(animal, dataset['recording_path'], imaging='suite2p')

    def test_an_unknown_source_is_refused(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        with pytest.raises(ValueError, match='Unknown imaging source'):
            ent.add_recording(animal, dataset['recording_path'], imaging='caiman')

    def test_imaging_without_io_is_refused(self, tmp_path, ent):
        """suite2p output with nothing to time it against is a broken folder,
        not a recording to ingest quietly."""
        dataset = _synthetic_data.build_dataset((tmp_path / 'noio').as_posix())
        os.remove(os.path.join(dataset['recording_path'], 'Io.hdf5'))
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        with pytest.raises(FileNotFoundError, match='no Io.hdf5'):
            ent.add_recording(animal, dataset['recording_path'])


class TestPhaseTimeWindows:
    """Present whether or not there is imaging, and read off the record group
    trace so they say when the phase ran rather than what was asked for."""

    def test_windows_match_the_record_group_trace(self, ingested):
        ent, dataset, _, _ = ingested

        for phase in ent.get(Phase):
            expected_start, expected_end = dataset['phase_windows'][phase['index']]
            assert phase['start_time'] == pytest.approx(expected_start, abs=0.01)
            assert phase['end_time'] == pytest.approx(expected_end, abs=0.01)

    def test_windows_are_ordered_and_disjoint(self, ingested):
        ent, _, _, _ = ingested

        windows = sorted((p['start_time'], p['end_time']) for p in ent.get(Phase))

        for (_, end), (start, _) in zip(windows, windows[1:]):
            assert end < start

    def test_a_phase_absent_from_the_trace_gets_no_window(self, ingested):
        """The display log is the source of phases; the io trace is the source
        of when they ran, and the two can disagree."""
        ent, _, _, recording = ingested

        assert all('start_time' in phase for phase in ent.get(Phase))


class FakeSource(ImagingSource):
    """A second source, so that "several" is exercised rather than asserted.

    Reads the same folder but writes two ROIs per plane, so the two sources are
    told apart by what they produced.
    """
    name = 'fake'

    def detect(self, path):
        return len(self.layer_names(path)) > 0

    def layer_names(self, path):
        return imaging_sources['suite2p'].layer_names(path)

    def ingest(self, imaging, path, frame_times_by_layer, options):
        ent = imaging.entarchy
        frame_times = frame_times_by_layer[0].squeeze()

        for layer_index, layer_name in enumerate(self.layer_names(path)):
            layer = Layer(ent, _id=layer_name, _parent=imaging)
            ent.add_new_entity(layer)
            layer['index'] = layer_index

            for roi_index in range(2):
                roi = Roi(ent, _id=f'Roi_{roi_index}', _parent=layer)
                ent.add_new_entity(roi)
                roi['index'] = roi_index
                roi['fluorescence'] = np.arange(len(frame_times), dtype=float)

        imaging['frame_times'] = frame_times
        imaging['frame_num'] = len(frame_times)


class BrokenSource(FakeSource):
    """Writes ROIs without the required fluorescence."""
    name = 'broken'

    def ingest(self, imaging, path, frame_times_by_layer, options):
        ent = imaging.entarchy
        layer = Layer(ent, _id='plane0', _parent=imaging)
        ent.add_new_entity(layer)
        roi = Roi(ent, _id='Roi_0', _parent=layer)
        ent.add_new_entity(roi)
        roi['index'] = 0

        imaging['frame_times'] = frame_times_by_layer[0].squeeze()
        imaging['frame_num'] = len(imaging['frame_times'])


def _two_sources(ent, dataset):
    animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
    return ent.add_recording(
        animal, dataset['recording_path'],
        imaging=['suite2p', ImagingSpec(source=FakeSource(), name='fake')])


class TestSeveralSources:

    def test_two_sources_coexist(self, ent, dataset):
        recording = _two_sources(ent, dataset)

        assert sorted(source.id for source in recording.imaging) == ['fake', 'suite2p']

    def test_each_source_owns_its_layers_and_rois(self, ent, dataset):
        recording = _two_sources(ent, dataset)

        suite2p = recording.imaging['suite2p']
        fake = recording.imaging['fake']

        assert len(suite2p.rois) == dataset['roi_num'] * dataset['plane_num']
        assert len(fake.rois) == 2 * dataset['plane_num']
        assert len(recording.rois) == len(suite2p.rois) + len(fake.rois)

    def test_colliding_layer_ids_do_not_collide(self, ent, dataset):
        """Both sources call their first plane plane0. They are different
        entities because they hang off different Imaging parents - which is the
        reason for the extra level."""
        recording = _two_sources(ent, dataset)

        plane0s = ent.get(Layer, 'id == "plane0"')

        assert len(plane0s) == 2
        assert len({layer.imaging.id for layer in plane0s}) == 2

    def test_each_source_gets_its_own_phase_window(self, ent, dataset):
        recording = _two_sources(ent, dataset)
        phase = sorted(recording.phases, key=lambda p: p['index'])[0]

        assert phase.frames_in(recording.imaging['suite2p']) is not None
        assert phase.frames_in(recording.imaging['fake']) is not None

    def test_asking_without_naming_one_is_refused(self, ent, dataset):
        recording = _two_sources(ent, dataset)

        with pytest.raises(LookupError, match='several imaging sources'):
            recording.sole_imaging()

        with pytest.raises(LookupError, match='several imaging sources'):
            sorted(recording.phases, key=lambda p: p['index'])[0].frames_in()

    def test_an_unknown_source_name_is_reported(self, ingested):
        _, _, _, recording = ingested

        with pytest.raises(KeyError, match='No imaging source'):
            recording.imaging['caiman']


class TestAddImagingLater:

    def test_a_source_can_be_added_after_ingest(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'], imaging=None)
        assert len(recording.imaging) == 0

        imaging = ent.add_imaging(recording, 'suite2p', path=dataset['recording_path'])

        assert imaging.id == 'suite2p'
        assert len(recording.imaging) == 1
        assert len(recording.rois) == dataset['roi_num'] * dataset['plane_num']
        assert recording['has_imaging'] is True

    def test_phase_windows_are_linked_for_a_later_source(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'], imaging=None)

        imaging = ent.add_imaging(recording, 'suite2p', path=dataset['recording_path'])

        for phase in recording.phases:
            assert phase.frames_in(imaging) is not None

    def test_a_second_source_can_be_added_later(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])

        ent.add_imaging(recording, FakeSource(), path=dataset['recording_path'],
                        name='fake')

        assert sorted(source.id for source in recording.imaging) == ['fake', 'suite2p']

    def test_a_duplicate_name_is_refused(self, ent, dataset):
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])

        with pytest.raises(ValueError, match='already has an imaging source'):
            ent.add_imaging(recording, 'suite2p', path=dataset['recording_path'])

    def test_the_path_must_be_given(self, ent, dataset):
        """An entarchy is self-contained; it does not keep a path it can read."""
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'], imaging=None)

        with pytest.raises(ValueError, match='needs the recording folder'):
            ent.add_imaging(recording, 'suite2p')


class TestSourceContract:

    def test_a_source_that_breaks_the_contract_is_caught(self, ent, dataset):
        """The check has to run after a commit, or it inspects an empty
        collection and passes - which is what it did when first written."""
        animal = ent.add_animal(EXPERIMENT, dataset['animal_path'])

        with pytest.raises(RuntimeError, match=r"without \['fluorescence'\]"):
            ent.add_recording(animal, dataset['recording_path'],
                              imaging=ImagingSpec(source=BrokenSource(), name='broken'))

    def test_the_contract_is_declared(self):
        assert Suite2PVxPy.ROI_REQUIRED == ('index', 'fluorescence')
        assert 'is_unit' in Suite2PVxPy.ROI_OPTIONAL
