## Entarchy schema for data analysis using vxpy- and suite2p-based data

Defines the entity hierarchy `Animal > Recording > {Layer > Roi, Phase}` on top of
[entarchy](https://github.com/thladnik/entarchy), ingests vxpy recordings and
suite2p output, and implements the contiguous-motion-noise (CMN) receptive field
analysis in `analysis/cmn`.

### Ingest

```python
ent = Suite2PVxPy(analysis_path)
animal = ent.add_animal('/data/animal_01')
ent.add_recording(animal, '/data/animal_01/rec_01')
```

`add_recording` expects a vxpy recording folder containing `Io.hdf5` (with the
galvo mirror sync signal) and the stimulus log HDF5 files. Behaviour videos
beside them are taken into the entarchy as media attributes; pass
`with_video=False` to skip that.

**Imaging is optional.** A recording of stimulus, io and behaviour data alone
ingests completely — phases included, since a stimulation phase is a fact about
what was shown rather than about the microscope.

```python
ent.add_recording(animal, path)                     # suite2p if the folder has it
ent.add_recording(animal, path, imaging='suite2p')  # require it
ent.add_recording(animal, path, imaging=None)       # skip it
```

Phases always carry `start_time` and `end_time` in seconds, read off the record
group trace so they say when the phase actually ran. `ca_start_index` /
`ca_end_index` are added only when there are imaging frames to index.

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

Making the source itself a choice, rather than suite2p with the serial numbers
filed off, is [a proposal](docs/proposals/imaging-sources.md); this is the first
two steps of it.

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
