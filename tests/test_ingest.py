"""End-to-end ingest of a synthetic vxpy + suite2p dataset."""
import numpy as np
import pytest

from entarchy.backend import SQLiteBackend
from entarchy_vxpy_suite2p.schema import Animal, Layer, Phase, Recording, Roi, Suite2PVxPy

import _synthetic_data


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
    animal = ent.add_animal(dataset['animal_path'])
    recording = ent.add_recording(animal, dataset['recording_path'])
    return ent, dataset, animal, recording


class TestAddAnimal:

    def test_creates_animal_entity(self, ent, dataset):
        animal = ent.add_animal(dataset['animal_path'])

        assert isinstance(animal, Animal)
        assert animal.id == dataset['animal_id']
        assert len(ent.get(Animal)) == 1

    def test_metadata_is_flattened_onto_entity(self, ent, dataset):
        animal = ent.add_animal(dataset['animal_path'])

        assert animal['metadata/strain'] == 'wildtype'
        assert animal['metadata/age_dpf'] == 6
        assert animal['metadata/nested/dish'] == 3

    def test_zstack_is_stored(self, ent, dataset):
        animal = ent.add_animal(dataset['animal_path'])

        assert animal['zstack_fn'] == 'zstack_1.tif'
        assert np.array_equal(animal['zstack'], dataset['zstack'])

    def test_missing_zstack_is_tolerated(self, tmp_path, ent):
        dataset = _synthetic_data.build_dataset((tmp_path / 'nostack').as_posix(),
                                                with_zstack=False)
        animal = ent.add_animal(dataset['animal_path'])

        assert 'zstack' not in animal

    def test_adding_twice_returns_existing_animal(self, ent, dataset, capsys):
        first = ent.add_animal(dataset['animal_path'])
        second = ent.add_animal(dataset['animal_path'])

        assert second.uuid == first.uuid
        assert len(ent.get(Animal)) == 1
        assert 'already exists' in capsys.readouterr().out

    def test_animal_is_child_of_root(self, ent, dataset):
        animal = ent.add_animal(dataset['animal_path'])
        assert animal.parent.uuid == ent.root.uuid


class TestAddRecording:

    def test_creates_full_entity_tree(self, ingested):
        ent, dataset, animal, recording = ingested

        assert isinstance(recording, Recording)
        assert len(ent.get(Recording)) == 1
        assert len(ent.get(Layer)) == dataset['plane_num']
        assert len(ent.get(Phase)) == len(dataset['phase_indices'])
        assert len(ent.get(Roi)) == dataset['plane_num'] * dataset['roi_num']

    def test_parent_links(self, ingested):
        ent, _, animal, recording = ingested

        assert recording.parent.uuid == animal.uuid
        layer = ent.get(Layer)[0]
        assert layer.parent.uuid == recording.uuid

        roi = ent.get(Roi)[0]
        assert roi.layer.uuid == layer.parent.uuid or roi.layer.parent.uuid == recording.uuid
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
        animal = ent.add_animal(dataset['animal_path'])
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

    def test_imaging_rate_is_plausible(self, ingested):
        _, _, _, recording = ingested
        # 10 Hz mirror -> 20 half-cycles per second, split across 2 planes
        assert recording['imaging_rate'] == pytest.approx(10.0, rel=0.05)

    def test_ca_times_are_monotonic(self, ingested):
        _, _, _, recording = ingested
        assert np.all(np.diff(recording['ca_times']) > 0)

    def test_signal_length_matches_ca_times(self, ingested):
        _, _, _, recording = ingested
        assert recording['signal_length'] == len(recording['ca_times'])

    def test_ca_times_match_fluorescence_length(self, ingested):
        """Every ROI's signal must be indexable by the recording timeline."""
        ent, dataset, _, recording = ingested
        ca_times = recording['ca_times']

        for roi in ent.get(Roi):
            assert len(roi['fluorescence']) == len(ca_times)

    def test_layer_time_offsets(self, ingested):
        ent, dataset, _, recording = ingested

        offsets = {layer.id: layer['t_offset'] for layer in ent.get(Layer)}
        assert offsets['plane0'] == 0.0
        assert offsets['plane1'] > 0.0

    def test_record_group_ids_are_interpolated_to_frames(self, ingested):
        _, _, _, recording = ingested

        ids = recording['record_group_ids']
        assert set(np.unique(ids)).issubset({-1.0, 0.0, 1.0})
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
        ca_times = recording['ca_times']

        for phase in ent.get(Phase):
            start = phase['ca_start_index']
            end = phase['ca_end_index']

            assert 0 <= start < end < len(ca_times)
            # The window must overlap the stimulus interval it was derived from
            assert ca_times[start] >= phase['display/__start_time'] - 0.5
            assert ca_times[end] <= (phase['display/__start_time']
                                     + phase['display/__target_duration'] + 0.5)

    def test_phases_are_children_of_recording(self, ingested):
        ent, _, _, recording = ingested
        for phase in ent.get(Phase):
            assert phase.recording.uuid == recording.uuid


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

    def test_iscell_is_stored(self, ingested):
        ent, _, _, _ = ingested

        roi = ent.get(Roi)[0]
        assert len(roi['iscell']) == 2

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

        animal = ent.add_animal(dataset['animal_path'])
        ent.add_recording(animal, dataset['recording_path'])

        roi = ent.get(Roi)[0]
        assert 'iscell' not in roi
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
            recording['imaging_rate'] = 999.0

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

    def test_ca_times_follow_the_last_processed_layer(self, tmp_path, ent):
        dataset = _synthetic_data.build_dataset((tmp_path / 'uneven').as_posix(),
                                                frames_per_plane=[60, 40])

        animal = ent.add_animal(dataset['animal_path'])
        recording = ent.add_recording(animal, dataset['recording_path'])

        # ca_times is assigned after the layer loop, so it reflects plane1 (40),
        # not plane0 (60) - ROI signals in plane0 are then longer than ca_times.
        assert len(recording['ca_times']) == 40

        plane0 = ent.get(Layer, 'id == "plane0"')[0]
        assert len(plane0.rois[0]['fluorescence']) == 60
