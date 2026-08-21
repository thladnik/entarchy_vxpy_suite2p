"""The command line ingest.

Every test calls `main(argv)` directly rather than spawning a process, so a
failure points at the line that caused it.
"""
import os

import pytest

from entarchy_vxpy_suite2p.cli import main
from entarchy_vxpy_suite2p.schema import Animal, Experiment, Recording, Roi, Suite2PVxPy

import _synthetic_data


@pytest.fixture()
def dataset(tmp_path):
    return _synthetic_data.build_dataset((tmp_path / 'cmn').as_posix())


@pytest.fixture()
def destination(tmp_path):
    return (tmp_path / 'archy').as_posix()


def opened(path):
    """The entarchy at `path`, for a test to read and close."""
    return Suite2PVxPy(path)


class TestCreate:

    def test_creates_an_entarchy(self, destination):
        assert main(['create', destination]) == 0
        assert os.path.exists(os.path.join(destination, 'entarchy.yaml'))

        ent = opened(destination)
        assert ent.root is not None
        assert len(ent.get(Experiment)) == 0
        ent.backend.close()

    def test_the_database_can_be_named(self, destination):
        main(['create', destination, '--dbname', 'named.db'])

        assert os.path.exists(os.path.join(destination, 'named.db'))

    def test_refuses_to_create_over_an_existing_one(self, destination):
        main(['create', destination])

        with pytest.raises(SystemExit, match='already an entarchy'):
            main(['create', destination])


class TestScan:

    def test_reports_the_folder(self, dataset, capsys):
        assert main(['scan', dataset['root']]) == 0

        printed = capsys.readouterr().out
        assert dataset['animal_id'] in printed
        assert '1 animals, 1 recordings' in printed

    def test_a_folder_with_nothing_in_it_is_an_error(self, tmp_path):
        empty = (tmp_path / 'empty').as_posix()
        os.makedirs(empty)

        with pytest.raises(SystemExit, match='no animal folder'):
            main(['scan', empty])

    def test_a_missing_folder_is_an_error(self, tmp_path):
        with pytest.raises(SystemExit, match='No experiment folder'):
            main(['scan', (tmp_path / 'nope').as_posix()])


class TestAdd:

    def test_ingests_into_an_existing_entarchy(self, destination, dataset):
        main(['create', destination])

        assert main(['add', destination, dataset['root']]) == 0

        ent = opened(destination)
        assert len(ent.get(Experiment)) == 1
        assert len(ent.get(Animal)) == 1
        assert len(ent.get(Recording)) == 1
        assert len(ent.get(Roi)) > 0
        ent.backend.close()

    def test_will_not_make_the_entarchy_by_accident(self, destination, dataset):
        """A typo in the path must not quietly produce a second empty store."""
        with pytest.raises(SystemExit, match='No entarchy at'):
            main(['add', destination, dataset['root']])

        assert not os.path.exists(destination)

    def test_create_makes_it_on_the_way_in(self, destination, dataset):
        assert main(['add', destination, dataset['root'], '--create']) == 0

        ent = opened(destination)
        assert len(ent.get(Recording)) == 1
        ent.backend.close()

    def test_dry_run_touches_nothing(self, destination, dataset, capsys):
        main(['create', destination])

        assert main(['add', destination, dataset['root'], '--dry-run']) == 0
        assert dataset['animal_id'] in capsys.readouterr().out

        ent = opened(destination)
        assert len(ent.get(Recording)) == 0
        ent.backend.close()

    def test_name_overrides_the_folder_name(self, destination, dataset):
        main(['add', destination, dataset['root'], '--create', '--name', 'cmn_v2'])

        ent = opened(destination)
        assert [e.id for e in ent.get(Experiment)] == ['cmn_v2']
        ent.backend.close()

    def test_running_it_again_adds_what_is_new(self, destination, dataset, tmp_path):
        """The workflow this is for: the folder grows between runs."""
        main(['add', destination, dataset['root'], '--create'])

        _synthetic_data.build_dataset(dataset['root'], animal_id='fish2',
                                      recording_id='rec_01')

        assert main(['add', destination, dataset['root']]) == 0

        ent = opened(destination)
        assert len(ent.get(Experiment)) == 1
        assert len(ent.get(Animal)) == 2
        assert len(ent.get(Recording)) == 2
        ent.backend.close()

    def test_imaging_none_leaves_the_rois_out(self, destination, dataset):
        main(['add', destination, dataset['root'], '--create', '--imaging', 'none'])

        ent = opened(destination)
        assert len(ent.get(Recording)) == 1
        assert len(ent.get(Roi)) == 0
        assert ent.get(Experiment)[0]['imaging'] == 'none'
        ent.backend.close()

    def test_the_timing_reaches_the_experiment(self, destination, dataset):
        main(['add', destination, dataset['root'], '--create',
              '--imaging', 'suite2p', '--timing', 'sync-signal',
              '--method', 'y_mirror', '--signal', 'ai_y_mirror_in'])

        ent = opened(destination)
        experiment = ent.get(Experiment)[0]
        assert experiment['imaging/suite2p/timing/type'] == 'SyncSignalTiming'
        assert experiment['imaging/suite2p/timing/method'] == 'y_mirror'
        assert experiment['imaging/suite2p/timing/signal'] == 'ai_y_mirror_in'
        ent.backend.close()

    def test_a_second_run_may_not_retime_it(self, destination, dataset):
        main(['add', destination, dataset['root'], '--create',
              '--imaging', 'suite2p', '--frame-avg-num', '1'])

        with pytest.raises(ValueError, match='a different imaging choice'):
            main(['add', destination, dataset['root'],
                  '--imaging', 'suite2p', '--frame-avg-num', '2'])

    def test_a_recording_that_did_not_make_it_exits_non_zero(self, destination,
                                                             dataset, monkeypatch,
                                                             capsys):
        main(['create', destination])

        def explode(*args, **kwargs):
            raise RuntimeError('this recording is broken')

        monkeypatch.setattr(Suite2PVxPy, 'add_recording', explode)

        assert main(['add', destination, dataset['root']]) == 1

        printed = capsys.readouterr().out
        assert 'did not make it in' in printed
        assert f'{dataset["animal_id"]}/{dataset["recording_id"]}' in printed


class TestTimingOptions:
    """Options that would be silently ignored are refused instead."""

    def test_a_timing_option_without_a_timing(self, destination, dataset):
        with pytest.raises(SystemExit, match='none was chosen'):
            main(['add', destination, dataset['root'], '--signal', 'di_frame_sync'])

    def test_clock_division_without_its_ratio(self, destination, dataset):
        with pytest.raises(SystemExit, match='needs --edges-per-volume'):
            main(['add', destination, dataset['root'],
                  '--imaging', 'suite2p', '--timing', 'clock-division'])

    def test_camera_without_a_device(self, destination, dataset):
        with pytest.raises(SystemExit, match='needs --device'):
            main(['add', destination, dataset['root'],
                  '--imaging', 'suite2p', '--timing', 'camera'])

    def test_timing_options_under_auto(self, destination, dataset):
        with pytest.raises(SystemExit, match='would be ignored'):
            main(['add', destination, dataset['root'],
                  '--imaging', 'auto', '--timing', 'sync-signal'])

    def test_frame_avg_num_under_auto(self, destination, dataset):
        with pytest.raises(SystemExit, match='would be ignored'):
            main(['add', destination, dataset['root'],
                  '--imaging', 'auto', '--frame-avg-num', '2'])

    def test_the_options_are_checked_before_the_entarchy_is_opened(self, destination,
                                                                   dataset):
        """So a mistake costs nothing rather than half an ingest."""
        with pytest.raises(SystemExit, match='needs --edges-per-volume'):
            main(['add', destination, dataset['root'], '--create',
                  '--imaging', 'suite2p', '--timing', 'clock-division'])

        assert not os.path.exists(destination)
