"""Command line ingest: make an entarchy, and put experiments into it.

    python -m entarchy_vxpy_suite2p create /data/entarchy
    python -m entarchy_vxpy_suite2p add /data/entarchy /raw/cmn
    python -m entarchy_vxpy_suite2p scan /raw/cmn

`add` is the whole of an experiment folder - `<experiment>/<animal>/<recording>`
- and may be run again whenever the folder has grown, since animals and
recordings already in the entarchy are skipped.

Frame timing cannot be detected and is named here rather than guessed:

    python -m entarchy_vxpy_suite2p add /data/entarchy /raw/rot_trans \\
        --imaging suite2p --timing clock-division --edges-per-volume 7.5 \\
        --signal di_frame_sync

Whatever is named is written onto the Experiment, so the entarchy records how it
read its own data. Handing the same experiment a different timing later is
refused rather than silently applied to the recordings that come after.
"""
from __future__ import annotations

import argparse
import os
import sys

from .schema import (CameraTiming, ClockDivisionTiming, ImagingSpec,
                     Suite2PVxPy, SyncSignalTiming, scan_experiment)

CONFIG_NAME = 'entarchy.yaml'


def _add_backend_options(parser):
    group = parser.add_argument_group('backend', 'only used when creating')
    group.add_argument('--backend', choices=['sqlite', 'mysql'], default='sqlite')
    group.add_argument('--dbname', default=None,
                       help='database file for sqlite (default entarchy.db), '
                            'schema name for mysql')
    group.add_argument('--dbhost', default=None, help='mysql only')
    group.add_argument('--dbuser', default=None, help='mysql only')


def _backend_config(args) -> tuple[str, dict]:
    """The backend name and its configuration, as create() takes them.

    A mysql password is never taken as an argument - it would end up in the
    shell history - and never written to entarchy.yaml. The backend reads
    ENTARCHY_DB_PASSWORD or asks.
    """
    if args.backend == 'sqlite':
        return 'SQLiteBackend', {'dbname': args.dbname or 'entarchy.db'}

    return 'MySQLBackend', {'dbname': args.dbname, 'dbhost': args.dbhost,
                            'dbuser': args.dbuser}


def _timing_from_args(args):
    """The FrameTiming the timing options name, or None if none was named."""
    named = {'--signal': args.signal, '--method': args.method,
             '--edges-per-volume': args.edges_per_volume, '--device': args.device}
    given = sorted(name for name, value in named.items() if value is not None)

    if args.timing is None:
        if given:
            raise SystemExit(f'{", ".join(given)} configures a frame timing, but '
                             f'none was chosen. Add --timing.')
        return None

    if args.timing == 'sync-signal':
        options = {name: value for name, value in
                   (('method', args.method), ('signal', args.signal))
                   if value is not None}
        return SyncSignalTiming(frame_avg_num=args.frame_avg_num, **options)

    if args.timing == 'clock-division':
        if args.edges_per_volume is None:
            raise SystemExit(
                '--timing clock-division needs --edges-per-volume. It is the '
                'volume period measured in clock ticks and cannot be guessed: '
                'measure it as edges/volumes on a recording where both are '
                'known, then check that it reproduces the frame count of the '
                'others.')

        options = {} if args.signal is None else {'signal': args.signal}
        return ClockDivisionTiming(args.edges_per_volume, **options)

    if args.device is None:
        raise SystemExit('--timing camera needs --device, the camera whose frame '
                         'times to use.')

    return CameraTiming(args.device)


def _imaging_from_args(args):
    """What `add_experiment(imaging=...)` should be given."""
    timing = _timing_from_args(args)

    if args.imaging in ('auto', 'none'):
        if timing is not None or args.frame_avg_num != 1:
            raise SystemExit(
                f'--imaging {args.imaging} leaves the source and its timing to '
                f'the defaults, so the timing options would be ignored. Name a '
                f'source instead, e.g. --imaging suite2p.')

        return None if args.imaging == 'none' else 'auto'

    return ImagingSpec(source=args.imaging, timing=timing,
                       frame_avg_num=args.frame_avg_num)


def _contents(folder: str, limit: int = None) -> list:
    if not os.path.isdir(folder):
        raise SystemExit(f'No experiment folder {folder}')

    contents = scan_experiment(folder)
    if limit is not None:
        contents = [(animal_path, recordings[:limit])
                    for animal_path, recordings in contents]

    if len(contents) == 0:
        raise SystemExit(f'{folder} holds no animal folder with a vxpy recording '
                         f'in it. An experiment folder is '
                         f'<experiment>/<animal>/<recording>.')

    return contents


def _print_contents(folder: str, contents: list) -> None:
    print(f'\n{folder}')
    for animal_path, recordings in contents:
        print(f'  {os.path.basename(animal_path):<46} {len(recordings)} recordings')

    print(f'  {"":<46} {len(contents)} animals, '
          f'{sum(len(r) for _, r in contents)} recordings')


def _open(args) -> Suite2PVxPy:
    if os.path.exists(os.path.join(args.entarchy, CONFIG_NAME)):
        return Suite2PVxPy(args.entarchy)

    if not args.create:
        raise SystemExit(
            f'No entarchy at {args.entarchy}. Make one with "create", or pass '
            f'--create to make it here.')

    return Suite2PVxPy.open_or_create(args.entarchy, *_backend_config(args))


def _create(args) -> int:
    if os.path.exists(os.path.join(args.entarchy, CONFIG_NAME)):
        raise SystemExit(f'There is already an entarchy at {args.entarchy}.')

    backend, config = _backend_config(args)
    ent = Suite2PVxPy.create(args.entarchy, backend, config)

    print(f'\nCreated {args.entarchy} ({backend})')
    ent.backend.close()

    return 0


def _add(args) -> int:
    # Read the folder and the options before opening anything, so that a
    #  mistake in either is an error now rather than halfway through a run
    imaging = _imaging_from_args(args)
    contents = _contents(args.folder, args.limit)
    _print_contents(args.folder, contents)

    if args.dry_run:
        return 0

    ent = _open(args)

    try:
        experiment = ent.add_experiment(
            args.folder, imaging=imaging, name=args.name,
            with_video=not args.no_video, skip_broken=not args.stop_on_error,
            limit=args.limit)

        # Check against the folder rather than against a counter, so that a
        #  recording which failed for any reason is named here
        wanted = {(os.path.basename(animal_path), os.path.basename(recording))
                  for animal_path, recordings in contents for recording in recordings}
        present = {(recording.animal.id, recording.id)
                   for recording in experiment.recordings}
        missing = sorted(wanted - present)

        print(f'\n{experiment.id}: {len(experiment.animals)} animals, '
              f'{len(experiment.recordings)} recordings, '
              f'{len(experiment.rois)} ROIs')

        if missing:
            print(f'\n{len(missing)} of the folder did not make it in:')
            for animal_id, recording_id in missing:
                print(f'  {animal_id}/{recording_id}')

            return 1
    finally:
        ent.backend.close()

    return 0


def _scan(args) -> int:
    _print_contents(args.folder, _contents(args.folder, args.limit))

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m entarchy_vxpy_suite2p',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    create_parser = subparsers.add_parser(
        'create', help='make an empty entarchy')
    create_parser.add_argument('entarchy', help='directory to create')
    _add_backend_options(create_parser)

    add_parser = subparsers.add_parser(
        'add', help='ingest an experiment folder',
        description='Ingest <experiment>/<animal>/<recording>. Run it again '
                    'whenever the folder has grown: what is already in the '
                    'entarchy is skipped.')
    add_parser.add_argument('entarchy', help='entarchy directory')
    add_parser.add_argument('folder', help='experiment folder')
    add_parser.add_argument('--name', default=None,
                            help='experiment id (default: the folder name)')
    add_parser.add_argument('--create', action='store_true',
                            help='make the entarchy if it is not there yet')
    add_parser.add_argument('--dry-run', action='store_true',
                            help='say what would be ingested and stop')
    add_parser.add_argument('--limit', type=int, default=None,
                            help='at most this many recordings per animal, for a '
                                 'trial run before committing hours to it')
    add_parser.add_argument('--no-video', action='store_true',
                            help='leave the behaviour videos out')
    add_parser.add_argument('--stop-on-error', action='store_true',
                            help='end the run on the first folder that fails, '
                                 'instead of reporting it and carrying on')

    imaging_group = add_parser.add_argument_group(
        'imaging', 'which signals to take, and how their frames were timed')
    imaging_group.add_argument('--imaging', default='auto',
                               help='"auto" for every source found in the folder, '
                                    '"none" for stimulus and behaviour only, or a '
                                    'source name such as suite2p')
    imaging_group.add_argument('--timing',
                               choices=['sync-signal', 'clock-division', 'camera'],
                               help='how frames are timed; the source default is '
                                    'used when this is left out')
    imaging_group.add_argument('--signal', default=None,
                               help='channel of Io.hdf5 carrying the sync trace')
    imaging_group.add_argument('--method', default=None,
                               help='sync-signal only, e.g. y_mirror')
    imaging_group.add_argument('--edges-per-volume', type=float, default=None,
                               help='clock-division only: the volume period in '
                                    'clock ticks, measured rather than guessed')
    imaging_group.add_argument('--device', default=None,
                               help='camera only: which camera recorded the times')
    imaging_group.add_argument('--frame-avg-num', type=int, default=1,
                               help='how many frames the scanner averaged into one')

    _add_backend_options(add_parser)

    scan_parser = subparsers.add_parser(
        'scan', help='read an experiment folder without ingesting it')
    scan_parser.add_argument('folder', help='experiment folder')
    scan_parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == 'create':
        return _create(args)

    if args.command == 'add':
        return _add(args)

    return _scan(args)


if __name__ == '__main__':
    sys.exit(main())
