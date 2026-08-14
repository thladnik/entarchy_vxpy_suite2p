"""Schema-level helpers: hierarchy declaration, metadata flattening, frame timing."""
import numpy as np
import pytest
import yaml

import entarchy
from entarchy_vxpy_suite2p import schema
from entarchy_vxpy_suite2p.schema import (Animal, Imaging, Layer, Phase, Recording, Roi, Suite2PVxPy,
                                          ca_frame_times_from_sync_toggle,
                                          ca_frame_times_from_y_mirror,
                                          frame_time_methods, unravel_dict)


class TestHierarchyDeclaration:

    def test_resolved_hierarchy(self):
        hierarchy, entity_map = Suite2PVxPy._resolve_hierarchy()

        assert hierarchy['Animal'] == {
            'Recording': {'Imaging': {'Layer': {'Roi': {}}}, 'Phase': {}}}
        for name in ('Animal', 'Recording', 'Imaging', 'Layer', 'Roi', 'Phase'):
            assert name in entity_map

    def test_child_entity_types(self):
        assert Animal.get_child_entity_types() == [Recording]
        assert Recording.get_child_entity_types() == [Imaging, Phase]
        assert Imaging.get_child_entity_types() == [Layer]
        assert Layer.get_child_entity_types() == [Roi]
        assert Roi.get_child_entity_types() is None

    def test_collection_types_are_registered(self):
        from entarchy_vxpy_suite2p.schema import (AnimalCollection, ImagingCollection,
                                                  LayerCollection, PhaseCollection,
                                                  RecordingCollection, RoiCollection)
        assert Animal.get_collection_type() is AnimalCollection
        assert Recording.get_collection_type() is RecordingCollection
        assert Layer.get_collection_type() is LayerCollection
        assert Roi.get_collection_type() is RoiCollection
        assert Phase.get_collection_type() is PhaseCollection
        assert Imaging.get_collection_type() is ImagingCollection

    def test_entities_subclass_entarchy_entity(self):
        for entity_type in (Animal, Recording, Imaging, Layer, Roi, Phase):
            assert issubclass(entity_type, entarchy.Entity)

    def test_version_is_declared(self):
        assert Suite2PVxPy._implementation_version in Suite2PVxPy._implementation_compat_version_list


class TestUnravelDict:

    class _Recorder(dict):
        """Stands in for an Entity - only __setitem__ is used."""

    def test_flat_dict(self):
        target = self._Recorder()
        unravel_dict({'a': 1, 'b': 'x'}, target, 'meta')
        assert target == {'meta/a': 1, 'meta/b': 'x'}

    def test_nested_dict_is_flattened_with_slashes(self):
        target = self._Recorder()
        unravel_dict({'outer': {'inner': {'leaf': 5}}, 'top': 1}, target, 'meta')
        assert target == {'meta/outer/inner/leaf': 5, 'meta/top': 1}

    def test_empty_dict_writes_nothing(self):
        target = self._Recorder()
        unravel_dict({}, target, 'meta')
        assert target == {}

    def test_empty_nested_dict_writes_nothing(self):
        target = self._Recorder()
        unravel_dict({'empty': {}}, target, 'meta')
        assert target == {}

    def test_values_are_passed_through_unchanged(self):
        target = self._Recorder()
        array = np.arange(3)
        unravel_dict({'arr': array, 'none': None}, target, 's2p')
        assert target['s2p/none'] is None
        assert np.array_equal(target['s2p/arr'], array)


class TestAddMetadata:

    def test_reads_and_flattens_metadata_files(self, tmp_path):
        (tmp_path / 'rec_metadata.yaml').write_text(
            yaml.safe_dump({'strain': 'wt', 'nested': {'age': 5}}))

        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))

        assert target['metadata/strain'] == 'wt'
        assert target['metadata/nested/age'] == 5

    def test_merges_multiple_files(self, tmp_path):
        (tmp_path / 'a_metadata.yaml').write_text(yaml.safe_dump({'a': 1}))
        (tmp_path / 'b_metadata.yaml').write_text(yaml.safe_dump({'b': 2}))

        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))

        assert target['metadata/a'] == 1
        assert target['metadata/b'] == 2

    def test_no_metadata_files_is_not_an_error(self, tmp_path):
        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))
        assert target == {}

    def test_ignores_unrelated_yaml(self, tmp_path):
        (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'ignored': True}))

        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))
        assert target == {}


def triangle_mirror_signal(frequency=10.0, duration=2.0, sample_rate=1000.0):
    """Galvo mirror trace that starts falling, so a trough precedes the first peak."""
    scipy_signal = pytest.importorskip('scipy.signal')
    times = np.arange(0, duration, 1 / sample_rate)
    position = scipy_signal.sawtooth(2 * np.pi * frequency * times - np.pi / 2, width=0.5)
    return position, times


class TestFrameTimesFromYMirror:

    def test_detects_every_half_cycle(self):
        position, times = triangle_mirror_signal(frequency=10.0, duration=2.0)
        indices, frame_times = ca_frame_times_from_y_mirror(position, times)

        # 20 troughs and 20 peaks over 2 s at 10 Hz; the first trough is dropped
        assert len(frame_times) == 39
        assert len(indices) == len(frame_times)

    def test_frame_times_are_monotonic(self):
        position, times = triangle_mirror_signal()
        _, frame_times = ca_frame_times_from_y_mirror(position, times)
        assert np.all(np.diff(frame_times) > 0)

    def test_intervals_are_half_a_cycle(self):
        position, times = triangle_mirror_signal(frequency=10.0, duration=2.0)
        _, frame_times = ca_frame_times_from_y_mirror(position, times)

        intervals = np.diff(frame_times)
        assert np.allclose(intervals, 0.05, atol=2e-3)

    def test_indices_reference_the_source_signal(self):
        position, times = triangle_mirror_signal()
        indices, frame_times = ca_frame_times_from_y_mirror(position, times)
        assert np.array_equal(times[indices], frame_times)

    def test_amplitude_independent(self):
        position, times = triangle_mirror_signal()
        _, small = ca_frame_times_from_y_mirror(position * 0.01, times)
        _, large = ca_frame_times_from_y_mirror(position * 100, times)
        assert np.array_equal(small, large)

    def test_offset_independent(self):
        position, times = triangle_mirror_signal()
        _, centred = ca_frame_times_from_y_mirror(position, times)
        _, offset = ca_frame_times_from_y_mirror(position + 5.0, times)
        assert np.array_equal(centred, offset)


class TestFrameTimesFromSyncToggle:

    def test_detects_rising_edges(self):
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0])
        times = np.arange(len(signal), dtype=float)

        indices, frame_times = ca_frame_times_from_sync_toggle(signal, times)
        assert np.array_equal(frame_times, [1.0, 5.0])

    def test_ignores_falling_edges(self):
        signal = np.array([1, 1, 0, 0, 0])
        times = np.arange(len(signal), dtype=float)

        _, frame_times = ca_frame_times_from_sync_toggle(signal, times)
        assert len(frame_times) == 0

    def test_constant_signal_yields_no_frames(self):
        times = np.arange(10, dtype=float)
        _, frame_times = ca_frame_times_from_sync_toggle(np.ones(10), times)
        assert len(frame_times) == 0


class TestFrameTimeMethodRegistry:

    def test_registered_methods(self):
        assert set(frame_time_methods) == {'y_mirror', 'frame_sync_toggle'}
        assert frame_time_methods['y_mirror'] is ca_frame_times_from_y_mirror
        assert frame_time_methods['frame_sync_toggle'] is ca_frame_times_from_sync_toggle

    def test_unknown_method_raises(self):
        with pytest.raises(KeyError):
            frame_time_methods['does_not_exist']
