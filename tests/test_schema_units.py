"""Schema-level helpers: hierarchy declaration, metadata flattening, frame timing."""
import numpy as np
import pytest
import yaml

import h5py

import entarchy
from entarchy_vxpy_suite2p import schema
from entarchy_vxpy_suite2p.schema import (Animal, ClockDivisionTiming, Imaging, Layer,
                                          Phase, Recording, Roi, Suite2PVxPy,
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
        """The frame time is the first sample recorded high, not the last one
        recorded low: the line goes high when the frame starts."""
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0])
        times = np.arange(len(signal), dtype=float)

        indices, frame_times = ca_frame_times_from_sync_toggle(signal, times)
        assert np.array_equal(frame_times, [2.0, 6.0])

    def test_ignores_falling_edges(self):
        signal = np.array([1, 1, 0, 0, 0])
        times = np.arange(len(signal), dtype=float)

        _, frame_times = ca_frame_times_from_sync_toggle(signal, times)
        assert len(frame_times) == 0

    def test_constant_signal_yields_no_frames(self):
        times = np.arange(10, dtype=float)
        _, frame_times = ca_frame_times_from_sync_toggle(np.ones(10), times)
        assert len(frame_times) == 0

    def test_a_boolean_line_counts_each_frame_once(self):
        """vxpy records a digital line as bool, and np.diff of a bool array is
        XOR - so without widening, every fall reads as a rise and the frame
        count comes out doubled."""
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        times = np.arange(len(signal), dtype=float)

        _, frame_times = ca_frame_times_from_sync_toggle(signal, times)
        assert np.array_equal(frame_times, [2.0, 6.0])

    def test_a_boolean_line_matches_an_integer_one(self):
        rng = np.random.default_rng(0)
        signal = rng.integers(0, 2, size=500).astype(bool)
        times = np.arange(len(signal), dtype=float)

        _, from_bool = ca_frame_times_from_sync_toggle(signal, times)
        _, from_int = ca_frame_times_from_sync_toggle(signal.astype(np.int8), times)
        assert np.array_equal(from_bool, from_int)


def _sync_line(tmp_path, edge_num: int, samples_per_period: int = 40,
               sample_rate: float = 600.0, signal: str = 'di_frame_sync'):
    """An Io.hdf5 holding a uniform square wave, as a free-running clock is.

    Built from sample counts rather than a modulo of the times, so that the
    number of rising edges is exactly `edge_num` and the period is exact.
    At the defaults that is a 15 Hz clock sampled at 600 Hz.
    """
    samples = (edge_num + 1) * samples_per_period
    times = np.arange(samples) / sample_rate
    line = np.zeros(samples, dtype=bool)

    # Start one sample in, so the first rise is an edge rather than the
    #  recording already having begun high
    for index in range(edge_num):
        start = 1 + index * samples_per_period
        line[start:start + samples_per_period // 2] = True

    with h5py.File(tmp_path / 'Io.hdf5', 'w') as h5file:
        h5file.create_dataset(signal, data=line[:, None])
        h5file.create_dataset(f'{signal}_time', data=times[:, None])

    return str(tmp_path)


class TestClockDivisionTiming:
    """A sync line that ticks faster than frames arrive, and not by a whole
    number - the tailtracking setup free-runs at 7.5 ticks per volume."""

    def test_divides_the_clock(self, tmp_path):
        path = _sync_line(tmp_path, edge_num=150)
        timing = ClockDivisionTiming(7.5)

        times = timing.frame_times(path, layer_num=5)

        assert len(times) == 5
        # 149 gaps between 150 edges, at 7.5 per volume, plus the first volume
        assert len(times[0]) == 20

    def test_a_fractional_ratio_does_not_drift(self, tmp_path):
        """Interpolating along the edges keeps the frames on the clock; a
        nominal rate would accumulate error over a long recording."""
        path = _sync_line(tmp_path, edge_num=1500)
        timing = ClockDivisionTiming(7.5)

        times = timing.frame_times(path, layer_num=5)[0]

        gaps = np.diff(times)
        assert np.allclose(gaps, gaps[0], atol=1e-3)
        # 7.5 ticks of a 15 Hz clock is half a second
        assert gaps[0] == pytest.approx(0.5, abs=1e-3)

    def test_layers_are_offset_within_the_volume(self, tmp_path):
        path = _sync_line(tmp_path, edge_num=150)

        times = ClockDivisionTiming(7.5).frame_times(path, layer_num=5)

        offsets = [t[0] - times[0][0] for t in times]
        assert offsets[0] == 0
        assert np.allclose(np.diff(offsets), 0.5 / 5, atol=1e-3)

    def test_a_whole_number_ratio_works_too(self, tmp_path):
        path = _sync_line(tmp_path, edge_num=100)

        times = ClockDivisionTiming(4).frame_times(path, layer_num=1)[0]

        assert len(times) == 25
        assert np.allclose(np.diff(times), 4 / 15, atol=1e-3)

    def test_a_silent_line_is_refused(self, tmp_path):
        with h5py.File(tmp_path / 'Io.hdf5', 'w') as h5file:
            h5file.create_dataset('di_frame_sync', data=np.zeros((100, 1), dtype=bool))
            h5file.create_dataset('di_frame_sync_time',
                                  data=np.arange(100, dtype=float)[:, None])

        with pytest.raises(ValueError, match='cannot time anything'):
            ClockDivisionTiming(7.5).frame_times(str(tmp_path), layer_num=1)

    def test_a_nonsense_ratio_is_refused(self):
        with pytest.raises(ValueError, match='must be positive'):
            ClockDivisionTiming(0)


class TestFrameTimeMethodRegistry:

    def test_registered_methods(self):
        assert set(frame_time_methods) == {'y_mirror', 'frame_sync_toggle'}
        assert frame_time_methods['y_mirror'] is ca_frame_times_from_y_mirror
        assert frame_time_methods['frame_sync_toggle'] is ca_frame_times_from_sync_toggle

    def test_unknown_method_raises(self):
        with pytest.raises(KeyError):
            frame_time_methods['does_not_exist']
