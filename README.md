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
galvo mirror sync signal), the stimulus log HDF5 files, and a `suite2p/planeN/`
directory per imaging layer. Behaviour videos beside them are taken into the
entarchy as media attributes; pass `with_video=False` to skip that.

That imaging is present, and that suite2p produced it, are currently
assumptions rather than options —
[a proposal](docs/proposals/imaging-sources.md) sets out what it would take to
make a recording work without imaging and to ingest signals from more than one
source.

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
