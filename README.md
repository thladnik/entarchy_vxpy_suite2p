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
directory per imaging layer.

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
