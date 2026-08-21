## Entarchy schema for data analysis using vxpy- and suite2p-based data

Defines the entity hierarchy `Experiment > Animal > Recording > {Layer > Roi, Phase}`
on top of
[entarchy](https://github.com/thladnik/entarchy), ingests vxpy recordings and
suite2p output, and implements the contiguous-motion-noise (CMN) receptive field
analysis in `analysis/cmn`.

### Ingest

A whole experiment folder goes in with one call:

```python
ent = Suite2PVxPy.open_or_create(analysis_path, 'SQLiteBackend',
                                 {'dbname': 'entarchy.db'})
ent.add_experiment('/data/cmn')
```

`add_experiment` expects `<experiment>/<animal>/<recording>` and walks it: a
folder is a recording if it holds one of the files vxpy writes, and an animal
if it holds at least one recording, which leaves out `ants_registration` and
anything else in the tree. `scan_experiment(path)` returns the same reading
without ingesting anything, for a look before a run that takes hours.

Re-running continues rather than duplicating — animals and recordings that are
already there are skipped — so an ingest that stopped is resumed by running it
again. A folder that fails is reported and the run carries on; `skip_broken=False`
re-raises instead.

The levels are separately available, each taking its parent first:

```python
animal = ent.add_animal('cmn', '/data/cmn/2024-08-02_fish1')
ent.add_recording(animal, '/data/cmn/2024-08-02_fish1/rec_01')
```

**Frame timing is chosen per experiment, and recorded there.** It cannot be
detected — a rig that recorded no galvo mirror trace has to be timed from a
divided clock whose ratio was measured by hand — so whatever is passed to
`add_experiment` is written onto the Experiment under `imaging/`, and the
entarchy says how it read its own data:

```python
ent.add_experiment('/data/rot_trans', imaging=ImagingSpec(
    timing=ClockDivisionTiming(7.5, signal='di_frame_sync')))

experiment['imaging/suite2p/timing/type']             # 'ClockDivisionTiming'
experiment['imaging/suite2p/timing/edges_per_volume']  # 7.5
```

An experiment reaches everything below it, and an animal names the experiment
it belongs to:

```python
experiment.animals, experiment.recordings, experiment.rois, experiment.phases
animal.experiment
ent.get(Roi, '[Experiment]id == "cmn"')
```

`add_recording` expects a vxpy recording folder containing `Io.hdf5` (with the
galvo mirror sync signal) and the stimulus log HDF5 files. Behaviour videos
beside them are taken into the entarchy as media attributes; pass
`with_video=False` to skip that.

**Imaging is optional, and its source is a choice.** A recording of stimulus,
io and behaviour data alone ingests completely — phases included, since a
stimulation phase is a fact about what was shown rather than about the
microscope.

```python
ent.add_recording(animal, path)                     # every source found in the folder
ent.add_recording(animal, path, imaging='suite2p')  # this one, and require it
ent.add_recording(animal, path, imaging=None)       # none

# several, and later
ent.add_recording(animal, path, imaging=['suite2p', ImagingSpec(CaImAnSource(),
                                                               name='caiman')])
ent.add_imaging(recording, 'suite2p', path='/data/fish1/rec_01')
```

Each source is an `Imaging` entity under the Recording, named after itself, and
its layers and ROIs hang off it:

```
Experiment > Animal > Recording > Imaging > Layer > Roi
                                > Phase
```

so `plane0` from suite2p and `plane0` from CaImAn are different entities rather
than a collision. Everything about when frames happened belongs to the source —
`imaging['rate']`, `imaging['frame_times']` — because two sources of one
recording need not agree.

```python
recording.imaging               # every source
recording.imaging['suite2p']    # one by name
recording.sole_imaging()        # the only one, or an error saying there are several
recording.rois                  # every ROI, whichever source found it
```

Phases always carry `start_time` and `end_time` in seconds, read off the record
group trace so they say when the phase actually ran. Which *frames* a phase
covers depends on the source, so it is a link rather than an attribute:

```python
phase.frames_in(imaging)    # (start_index, end_index)
phase.frames_in()           # when there is only one source
```

### What an ROI carries

Whatever segmented it, an ROI has:

| attribute | |
|---|---|
| `index` | position within its layer — required |
| `fluorescence` | the trace, one sample per frame — required |
| `spikes` | deconvolved activity, if the method produces it |
| `is_unit` | whether the method calls it a real cell |
| `unit_probability` | how sure it was |

A source's own vocabulary stays namespaced beside these — `s2p/npix`,
`s2p/skew` — so nothing is lost. What the shared names buy is analysis that
reads an ROI without knowing what produced it, which is why the CMN analysis
mentions suite2p nowhere. The required names are checked at ingest.

Adding a source means implementing `ImagingSource` — `detect`, `layer_names`,
`ingest` — and registering it in `imaging_sources`. Frame timing is a separate
interface (`FrameTiming`), because how a scanner was timed and what read its
output are independent choices. The reasoning is in
[the proposal](docs/proposals/imaging-sources.md).

### Example notebook

[`examples/01_synthetic_dataset.ipynb`](examples/01_synthetic_dataset.ipynb)
builds a small entarchy in this schema with synthetic calcium imaging data, then
works through querying it, DataFrames, parallel analysis with `map_async`, links
between entities, and archiving. It needs no experimental data and runs in about
half a minute.

```sh
pip install -e ".[examples]"
jupyter lab examples/01_synthetic_dataset.ipynb
```

It also runs from a checkout without installing, since the first cell puts the
repository root on the path.

### Running tests

```sh
pip install -e . pytest
pytest
```

The CMN analysis tests require `torch` (part of the `cmn` extra) and are skipped
if it is missing. The ingest tests build a synthetic vxpy + suite2p dataset on
disk, so no experimental data is needed.

A few tests are marked `xfail` to document known defects without failing the
suite; they are listed with a reason and will report as unexpectedly passing once
the underlying issue is fixed.
