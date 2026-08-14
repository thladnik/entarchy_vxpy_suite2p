# Making imaging optional, and its source a choice

**Implemented.** All five steps are done; what follows is the reasoning, kept
because the alternatives it rejects are the questions anyone changing this will
ask again. Where the implementation deviated, it says so.

A Recording currently *is* a two-photon recording processed by suite2p. That is
not stated anywhere; it is implied by what `add_recording` does and what it
crashes on. This proposal separates three things the code treats as one — a
recording, an imaging acquisition, and the software that extracted signals from
it — so that a recording may have no imaging, one source, or several, chosen at
ingest or added later.

Nothing here is backward compatible. The entity hierarchy changes, so existing
entarchies will not open; `_implementation_version` goes to 0.3 and says so.

## What is coupled today

Seven places, in the order `add_recording` hits them.

**1. The sync signal is a galvo mirror.** [schema.py:308](../../entarchy_vxpy_suite2p/schema.py#L308)
opens `Io.hdf5` and runs `frame_time_methods[sync_type]` over `ai_y_mirror_in`
before anything else happens. There is no path through the function that skips
it. `frame_time_methods` is already a registry of two, which is the one piece of
pluggability that exists.

**2. suite2p is assumed to be on disk.**
```python
for _name in os.listdir(os.path.join(path, 'suite2p')):
```
No guard. A recording folder without a `suite2p/` directory raises
`FileNotFoundError` from inside the ingest.

**3. Multi-plane demultiplexing assumes sequential galvo acquisition.**
`frame_times_all[layer_idx + frame_avg_num // 2 :: layer_num * frame_avg_num]`
encodes how a resonant/galvo scanner interleaves planes. It is correct for that
instrument and meaningless for a light-sheet stack or a single-plane camera.

**4. Recording-level timing is imaging timing.** `imaging_rate`, `ca_times`,
`signal_length` and `record_group_ids` all live on the Recording, and all are
derived from imaging frame times. A recording with two imaging sources has
nowhere to put two of each; a recording with none has nothing to put there.

**5. Phases are indexed in imaging frames.** [schema.py:409](../../entarchy_vxpy_suite2p/schema.py#L409)
gives every Phase a `ca_start_index` and `ca_end_index`, computed from
`record_group_ids` interpolated onto the first layer's frame times. So **Phase
entities cannot be created without imaging**, even though a stimulation phase is
a fact about the stimulus, not about the microscope.

**6. Layer is a suite2p plane directory.** Its id is `plane0`, its attributes are
`s2p/...` from `ops.npy`, and its existence comes from the folder listing.

**7. Roi attribute names are suite2p's.** `fluorescence` (from `F.npy`), `spikes`
(`spks.npy`), `iscell`, `s2p/npix` and so on. `calculate_dff` reads
`roi['fluorescence']` and `roi.recording['imaging_rate']` directly, so the
analysis is coupled to suite2p through the attribute names alone.

## What the requirements actually demand

*Imaging is optional* means a behaviour-only recording — tail tracking with a
stimulus and no microscope — must ingest completely: io channels, stimulus log,
phases, camera videos. Everything except signals.

*The source is a choice* means suite2p is one implementation of an interface,
not the thing the ingest is written around.

*Several sources* covers two different cases, and the design has to serve both:

- **Different software on the same acquisition** — suite2p and CaImAn on one
  stack, to compare. They share frame times and differ in ROIs.
- **Different acquisitions in one recording** — two-colour imaging, or a 2p
  stack alongside a widefield channel. They differ in frame times too.

*At ingest or later* means the ingest cannot be a single function that must know
everything up front.

## The proposed structure

### A level between Recording and Layer

```
Animal
└── Recording                     stimulus, io, camera, videos, phases
    ├── Phase                     what was shown, in seconds
    └── Imaging                   one acquisition-and-extraction
        └── Layer                 a plane within it
            └── Roi               a unit within the plane
```

An `Imaging` entity is one source: how its frames were timed, what extracted the
signals, and the parameters of both. Its id is the source name given at ingest —
`suite2p`, `suite2p_rerun`, `caiman`, `widefield` — so two sources on one
recording are two entities rather than two colliding `plane0` ids.

Everything currently on the Recording that is imaging-derived moves here:

```python
imaging['method'] = 'suite2p'
imaging['frame_times']  = ...   # was recording['ca_times']
imaging['rate']         = ...   # was recording['imaging_rate']
imaging['frame_num']    = ...   # was recording['signal_length']
imaging['layer_num']    = 2
imaging['s2p/version']  = '0.14.2'
```

The Recording keeps only what the acquisition software wrote: `io/*`,
`display/*`, `camera/*`, the videos, and `record_group_ids` **on the io
timebase** rather than resampled onto imaging frames.

### The normalised Roi contract

This is the part that decouples analysis, and it is worth more than the
hierarchy change. Every source must write:

| attribute | meaning | required |
|---|---|---|
| `index` | position within its layer | yes |
| `fluorescence` | the trace, one sample per frame | yes |
| `spikes` | deconvolved activity, if the method produces it | no |
| `is_unit` | whether the method classifies it as a real cell | no |
| `unit_probability` | how sure it was | no |

*(Implemented in step 2 with `fluorescence` and `spikes` kept rather than
renamed to `signal` and `deconvolved`, as first drafted. Neither is suite2p's
word — suite2p's files are `F.npy` and `spks.npy` — and `roi['signal']` would
have sat beside `roi['signal_length']` and `roi['signal_proportion']`, which
the CMN analysis already writes meaning something unrelated. `iscell` was
suite2p's word, and its packed `[verdict, probability]` row was suite2p's
format, so that one did get split and renamed.)*

Everything a source knows that is its own stays namespaced as it is now —
`s2p/npix`, `s2p/skew`, `caiman/SNR_comp`. Nothing is lost; what changes is that
these names mean the same thing whoever wrote them.

What remains coupling `calculate_dff` to imaging is structural rather than
lexical - it reads `roi.recording['imaging_rate']`, which step 3 moves onto the
`Imaging` entity:

```python
def calculate_dff(roi, window_size=120, percentile=10):
    rate = roi.imaging['rate']          # was roi.recording['imaging_rate']
```

### Sources as a registry

`frame_time_methods` already establishes the pattern; this generalises it.

```python
class ImagingSource:
    name: str

    def detect(self, path: str) -> bool:
        """Whether this source has data in a recording folder."""

    def frame_times(self, path: str, context: IngestContext) -> np.ndarray:
        """When each frame was acquired, on the recording's timebase."""

    def ingest(self, imaging: Imaging, path: str, frame_times: np.ndarray) -> None:
        """Create the layers and ROIs, writing the contract above."""


IMAGING_SOURCES = {'suite2p': Suite2pSource(), ...}
```

Splitting `frame_times` from `ingest` matters: they are independent choices. A
suite2p extraction may be timed from a galvo mirror, a frame-sync toggle, or a
timestamps file the acquisition wrote, and the same timing may serve suite2p and
CaImAn. Keeping them separate means the galvo demultiplexing in coupling point 3
belongs to a `GalvoMirrorTiming`, not to suite2p.

The suite2p source is the existing code lifted out of `add_recording`
essentially unchanged — the plane scan, `ops.npy`, `F/spks/stat/iscell`, the
frame-count reconciliation, and the ANTs coordinates.

### The API

```python
# no imaging at all - the default
recording = ent.add_recording(animal, path)

# one source, named explicitly
recording = ent.add_recording(animal, path, imaging='suite2p')

# whatever is found in the folder
recording = ent.add_recording(animal, path, imaging='auto')

# several, with their own timing
recording = ent.add_recording(animal, path, imaging=[
    ImagingSpec('suite2p', timing='galvo_mirror', frame_avg_num=2),
    ImagingSpec('caiman', timing='galvo_mirror'),
])

# added later, against the original folder
ent.add_imaging(recording, 'caiman', path=recording_path)
```

`add_imaging` takes the path again rather than reading it from the Recording.
The Recording records `source_path` as provenance, but an entarchy is
self-contained and must not *rely* on a path outside itself — which is exactly
why videos are copied in rather than referenced.

Access:

```python
recording.imaging               # ImagingCollection
recording.imaging['suite2p']    # one source by name
recording.rois                  # every ROI, whatever the source
recording.imaging['suite2p'].rois
```

`recording.rois` staying meaningful matters: most analysis does not care which
source an ROI came from, and should not have to say so.

## The hard part: phases without imaging

A Phase currently carries `ca_start_index` / `ca_end_index`, which only exist if
there is imaging, and are ambiguous if there are two sources.

Every Phase should carry what the stimulus log knows, which needs no microscope:

```python
phase['index'], phase['start_time'], phase['end_time']   # seconds, io timebase
```

The frame window is then a property of the *pair* (phase, imaging source) — and
that is precisely what a link is for, now that entarchy has them:

```python
ent.define_link_type('phase_frames', Phase, Imaging)
ent.link(phase, imaging, 'phase_frames', start_index=..., end_index=...)

phase.frames_in(imaging)     # -> (start, end), a helper over the link
```

**The alternative is a namespaced attribute** — `phase['suite2p/ca_start_index']`
— which is simpler to read and write and needs no link machinery. I recommend
the link anyway: the window genuinely belongs to the pair, links make it
queryable (`ent.links('phase_frames', '@Phase.index == 3')`), and the
alternative bakes a source name into an attribute string that nothing validates.
But it is the one place in this proposal where the simpler option is defensible,
and a `frames_in` helper hides the difference either way.

## What breaks

99 references to `Layer` or `Roi` across nine modules, plus 31 in the example
notebook. Concretely:

- `Layer.recording` and `Roi.recording` become two- and three-step traversals.
  Both are properties, so the change is in one place each.
- `[Recording]uuid == ...` filters keep working — ancestor traversal does not
  care how many levels are between.
- `roi['fluorescence']` → `roi['signal']` in `analysis/cmn/functions.py`,
  `plotting.py`, and the tests.
- `recording['imaging_rate']` → `roi.imaging['rate']`, `recording['ca_times']` →
  `imaging['frame_times']`, in `functions.py` and `analysis.py`.
- `phase['ca_start_index']` → `phase.frames_in(imaging)`.
- The synthetic dataset needs a no-imaging variant and a second-source variant,
  which the ingest tests currently cannot express.

The `update_roi_coordinates_from_registration` method is shaped around a suite2p
path and should become an annotator attached to an `Imaging` rather than a
recording folder — ANTs coordinates are another thing computed *about* an
extraction, not part of it.

## Two names that will be wrong afterwards

The package is called `entarchy_vxpy_suite2p`. If suite2p is one source among
several, the name says something untrue about what the package is for.
`entarchy_vxpy` is the honest name, and renaming is cheapest now rather than
after the imports are in more places.

`Roi` is suite2p's word for a segmented unit. `Unit` is what it is, and the
generic vocabulary would help make the point that the schema is not suite2p's.
This one I would *not* do — `Roi` is what the lab says out loud, and a schema
that argues with its users about vocabulary loses.

## Staging

Each step leaves the suite passing, and the first two are worth doing whether or
not the rest happens. **Steps 1 and 2 are done.**

1. ~~**Split the ingest.**~~ Pull io/display/camera/phase ingest out of
   `add_recording` into functions that never touch imaging. Add `imaging=None`
   and make the suite2p path conditional. Phases get `start_time`/`end_time`.
   *After this, a behaviour-only recording ingests.*
2. ~~**Normalise the Roi contract.**~~ Done, and checked at ingest: the CMN
   analysis now names suite2p nowhere.
3. ~~**Add the `Imaging` level**~~ and move the timing attributes onto it.
4. ~~**Extract the source interface**~~ and re-implement suite2p behind it.
5. ~~**Allow several**~~, and `add_imaging` after the fact.

Step 3 was the breaking one. `_implementation_version` is 0.3.

One thing the implementation found that the proposal did not anticipate:
entities queued inside a write context are invisible to a query until they are
written, so both the ROI contract check and the phase-frame linking had to
follow a `commit()`. The contract check written in step 2 had been inspecting an
empty collection and passing; there is now a test with a deliberately broken
source that fails if that regresses.
