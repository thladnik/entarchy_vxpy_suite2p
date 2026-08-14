"""Analysis steps that operate on ingested entities."""
import numpy as np
import pytest
import tifffile

torch = pytest.importorskip('torch', reason='the cmn analysis module imports torch at module level')

from entarchy.backend import SQLiteBackend
from entarchy_vxpy_suite2p.analysis.cmn import analysis, functions
from entarchy_vxpy_suite2p.schema import Roi, Suite2PVxPy

import _synthetic_data


@pytest.fixture()
def ingested(tmp_path):
    base = (tmp_path / 'archy').as_posix()
    ent = Suite2PVxPy.create(base, SQLiteBackend(base, dbname='analysis.db'))

    dataset = _synthetic_data.build_dataset((tmp_path / 'data').as_posix())
    animal = ent.add_animal(dataset['animal_path'])
    ent.add_recording(animal, dataset['recording_path'])

    yield ent
    ent.backend.close()


class TestCalculateDff:

    def test_shape_and_finiteness(self, ingested):
        roi = ingested.get(Roi)[0]
        functions.calculate_dff(roi, window_size=2, percentile=10)

        dff = roi['dff']
        assert len(dff) == len(roi['fluorescence'])
        assert np.all(np.isfinite(dff))

    def test_baseline_is_removed(self, ingested):
        """dF/F is relative, so a constant offset added to F must not change it much."""
        roi = ingested.get(Roi)[0]
        functions.calculate_dff(roi, window_size=2, percentile=10)
        original = roi['dff'].copy()

        # Rewriting ingested (immutable) data requires digest mode; the write must
        # commit while digest mode is still active, so no entarchy context here
        ingested.start_digest()
        try:
            roi['fluorescence'] = roi['fluorescence'] + 1000.0
        finally:
            ingested.end_digest()

        functions.calculate_dff(roi, window_size=2, percentile=10)
        assert np.abs(roi['dff']).max() < np.abs(original).max() * 1.5

    def test_window_size_scales_with_imaging_rate(self, ingested):
        roi = ingested.get(Roi)[0]

        functions.calculate_dff(roi, window_size=1, percentile=10)
        short = roi['dff'].copy()
        functions.calculate_dff(roi, window_size=5, percentile=10)
        long = roi['dff']

        assert not np.allclose(short, long)

    def test_constant_signal_yields_nan(self, ingested):
        """Known edge case: for a perfectly flat trace no sample falls below the
        10th percentile, so the baseline is the mean of an empty slice (NaN) and
        the whole dF/F trace becomes NaN rather than zero."""
        roi = ingested.get(Roi)[0]

        ingested.start_digest()
        try:
            roi['fluorescence'] = np.full(len(roi['fluorescence']), 500.0)
        finally:
            ingested.end_digest()

        with np.errstate(invalid='ignore'):
            with pytest.warns(RuntimeWarning):
                functions.calculate_dff(roi, window_size=2, percentile=10)

        assert np.isnan(roi['dff']).all()


class TestAreRoisInRegion:

    @staticmethod
    def write_region_map(path, shape=(6, 5, 4), inside=((1, 2, 3),)):
        """Region map as stored on disk; are_rois_in_region reorders the axes."""
        volume = np.zeros(shape, dtype=np.uint8)
        reordered = np.swapaxes(np.moveaxis(volume, 0, 2), 0, 1)
        for coordinate in inside:
            reordered[coordinate] = 255
        volume = np.moveaxis(np.swapaxes(reordered, 0, 1), 2, 0)
        tifffile.imwrite(path, volume, photometric='minisblack')

    def assign_coordinates(self, ent, coordinates):
        # Registration coordinates are analysis output, so they are written normally
        with ent:
            for roi, coordinate in zip(ent.get(Roi), coordinates):
                roi['ants/x'] = float(coordinate[0])
                roi['ants/y'] = float(coordinate[1])
                roi['ants/z'] = float(coordinate[2])

    def test_flags_rois_inside_the_region(self, ingested, tmp_path):
        region_path = (tmp_path / 'region.tif').as_posix()
        self.write_region_map(region_path, inside=((1, 2, 3),))

        rois = ingested.get(Roi)
        coordinates = [(1, 2, 3)] + [(0, 0, 0)] * (len(rois) - 1)
        self.assign_coordinates(ingested, coordinates)

        result = analysis.are_rois_in_region(ingested.get(Roi), region_path)

        assert result.sum() == 1
        assert len(result) == len(rois)

    def test_no_rois_inside(self, ingested, tmp_path):
        region_path = (tmp_path / 'region.tif').as_posix()
        self.write_region_map(region_path, inside=((3, 4, 5),))

        rois = ingested.get(Roi)
        self.assign_coordinates(ingested, [(0, 0, 0)] * len(rois))

        result = analysis.are_rois_in_region(ingested.get(Roi), region_path)
        assert not result.any()

    def test_float_coordinates_are_truncated(self, ingested, tmp_path):
        region_path = (tmp_path / 'region.tif').as_posix()
        self.write_region_map(region_path, inside=((1, 2, 3),))

        rois = ingested.get(Roi)
        coordinates = [(1.9, 2.9, 3.9)] + [(0, 0, 0)] * (len(rois) - 1)
        self.assign_coordinates(ingested, coordinates)

        result = analysis.are_rois_in_region(ingested.get(Roi), region_path)
        assert result.sum() == 1


class TestProcessRecordingGuards:

    def test_recording_without_imaging_is_skipped(self, tmp_path, capsys):
        """A recording may legitimately have no imaging, and this analysis is
        about calcium signals."""
        from entarchy_vxpy_suite2p.schema import Animal, Recording

        base = (tmp_path / 'partial').as_posix()
        ent = Suite2PVxPy.create(base, SQLiteBackend(base, dbname='partial.db'))
        try:
            with ent:
                animal = Animal(ent, _id='a', _parent=ent.root)
                ent.add_new_entity(animal)
                recording = Recording(ent, _id='r', _parent=animal)
                ent.add_new_entity(recording)

            functions.process_recording(recording)

            assert 'no imaging data' in capsys.readouterr().out
        finally:
            ent.backend.close()
