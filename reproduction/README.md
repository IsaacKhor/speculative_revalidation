# Reproducing the EuroSys 2027 paper figures

This directory contains the reproduction workflow for `paper/v2-eurosys27`.
It reads the public traces and repository sources. Generated data, models,
build products, caches, logs, and figures belong under `reproduction/`.

## Downloading and formatting traces

The downloader follows the public Cloudflare, Meta, and Wikimedia URLs listed
in the repository trace documentation and creates the exact flat input layout
used below:

- Cloudflare 2025: <https://objects.research.cloudflare.com/@ikhor/cdn-traces/readme.md>
- Meta 2023: <https://s3.amazonaws.com/cache-datasets/index.html#cache_dataset_txt/2023_metaCDN/>
- Wikimedia 2019: <https://analytics.wikimedia.org/published/datasets/caching/2019/>

```console
python3 reproduction/download_traces.py --dry-run
python3 reproduction/download_traces.py
```

Completed objects are reused, and partial objects resume when the publisher
supports byte ranges. Cloudflare's eight and Meta's three published CSV objects
are validated and placed under `traces/cdn_cf25_csv/` and
`traces/cdn_fb23_csv/`; Meta's published `rnha` object is renamed to the
paper's `rhna.csv.zst` input. Wikimedia's daily gzip TSVs are projected to
`timestamp,key,size` and assembled into `traces/cdn_wm19_csv/t-all.csv.zst`
and `u-all.csv.zst`. Formatting streams through compressed files and removes
staging archives after each successful conversion. Use `--keep-downloads` to
retain them, or repeat `--dataset cf`, `--dataset fb`, and `--dataset wm` to
prepare selected workloads separately.

The committed experiment's Wikimedia text trace contains published days
01--20 (timestamps 86400 through 1814399); the downloader intentionally omits
text day 00 to reproduce that input. The upload trace contains days 00--20.
Running the downloader acknowledges each publisher's dataset terms, including
Cloudflare's CC BY-NC-SA 4.0 non-commercial research restriction.

## Running

From the repository root, the single entrypoint is:

```console
python3 reproduction/reproduce.py
```

The default is full recomputation from the current traces, including binary
preprocessing, expiry collection, model training, simulation, and plotting.
Use `--resume` to continue completed stages. To inspect the proposed work
without running it, or to run the small verification experiment:

```console
python3 reproduction/reproduce.py --dry-run
python3 reproduction/reproduce.py --smoke
python3 reproduction/reproduce.py --validate
python3 reproduction/reproduce.py --resume
```

Once generated inputs and results exist, regenerate plots with:

```console
python3 reproduction/reproduce.py --plots-only
```

`--help` describes the available settings. The script resolves paths from its
own location, so it can also be called by absolute path from another directory.
The paper directory and its figures are not overwritten.

`--parallel N` controls simultaneous simulator configurations;
`--trace-root PATH` selects the existing read-only trace tree. A usable local
simulator build can be supplied with the pair `--sim-bin PATH --oracle-bin PATH`.
The default key-sampling ratio is 1 for the full experiment, 8 for smoke, and
256 for validation; `--key-sample-ratio N` overrides it and changes the
resulting experiment.

## Generated outputs

All paths below are relative to `reproduction/`:

| Path | Contents |
| --- | --- |
| `traces/sim/` | Compressed simulator-format traces with next-access metadata |
| `traces/expiry/` | Expiry-event data used for fresh model training and analysis |
| `models/` | Newly trained classifier models |
| `results/` | Baseline, oracle, ML sweep, and time-series measurements |
| `output/figs/` | Ten empirical paper plots |
| `output/diagrams/` | Two copied authored diagram assets |
| `output/manifest.json` | Sizes and SHA-256 hashes for all twelve assets |
| `logs/` | Command output and failure details |
| `.work/` | Staged build sources, build products, tool caches, and temporary data |

Smoke artifacts use the same structure under `reproduction/smoke/`, keeping
them separate from the full experiment. Use `--smoke --plots-only` to plot smoke
results and `--smoke --resume` to continue smoke work. `run.json` records the run
configuration. `--force` removes the selected mode's generated artifact
directories before recomputation; it does not remove the source scripts or
input traces. Generated artifacts removed by this option must be recomputed.

`--validate` is a more comprehensive, still bounded run over one complete trace
per workload (`cf_b`, `fb_a`, and `wm_t`). It scans each full CSV duration and
retains every request for keys matching the simulator's exact `key % KSR == 0`
predicate (KSR 256 by default) before materializing the simulator traces. It
exercises all five cache policies and the full ML and oracle threshold grids.
Its artifacts are isolated under `reproduction/validation/`.
After simulation it writes `output/validation.md` and
`output/validation.json`, checking grid
completeness, request-counter conservation, disabled revalidation in `never`,
and exact equivalence between ML threshold 1 and the matching baseline. The
report also aligns candidate rows with the committed `results/r10i1_*.csv`
files and shows rate deltas. Those deltas are informational, not pass/fail
tolerances: validation uses one spatially sampled trace per workload and
freshly trained models, while the committed results use full workload suites
and their committed models.

Cheap validation of the runner and the manuscript's figure inventory:

```console
python3 -B -m unittest discover -s reproduction -p 'test_*.py' -v
```

## Inputs

The current flat trace layout is required. The Cloudflare 2025 traces are the
paper's CDN1 workload; the 2021 and 2026 Cloudflare collections are not inputs
to these figures.

| Input CSV directory | Input stems | Simulator names, in corresponding order |
| --- | --- | --- |
| `traces/cdn_cf25_csv/` | `106m105`, `106m106`, `243m12`, `243m13`, `411m264`, `411m325`, `472m378`, `472m379` | `cf_a`, `cf_b`, `cf_c`, `cf_d`, `cf_e`, `cf_f`, `cf_g`, `cf_h` |
| `traces/cdn_fb23_csv/` | `reag`, `rhna`, `rprn` | `fb_a`, `fb_b`, `fb_c` |
| `traces/cdn_wm19_csv/` | `t-all`, `u-all` | `wm_t`, `wm_u` |

CSV inputs have the `.csv.zst` suffix. If legacy Parquet mirrors are present,
the TTL-distribution plot can read `cdn_cf25_parquet/server=<server>/*.parquet`
and `cdn_fb23_parquet/<stem>.parquet` for speed; otherwise it reads the
downloaded CSV/Zstandard files directly. Parquet files are not required.
The simulator names matter: its Wikimedia upload (`wm_u`) handling applies an
additional factor of eight to key sampling.

## Prerequisites and resources

Use Linux with Python 3.11 or newer, `curl`, `gzip`, GNU `cut`/`tail`, `uv`, a
C++20 compiler, xmake, and the zstd command-line tools (`zstd` and `zstdcat`).
On its first run, uv installs the
locked Python environment and xmake obtains Boost 1.83, fmt 9.1, Abseil, zstd,
and ONNX Runtime in a reproduction-local cache; this requires network access.
Existing binaries copied from another machine can have incompatible or missing
shared libraries.

The public downloads total approximately 73 GiB; the formatted CSV inputs
occupy approximately 66 GiB compressed. Conversion
uses 48 bytes per request before compression and a reverse pass holding a key
index in memory. Allow substantial additional disk space for temporary binary
traces, compressed binaries, expiry events, and simulation results. Memory use
and run time depend strongly on the largest trace and simulation parallelism;
the full experiment comprises repeated passes over billions of requests.
Start with low parallelism on an unfamiliar machine. A short validation run is
not evidence that the full experiment fits available memory or disk.

The two architecture/timeline diagrams are authored illustrations rather than
trace-derived plots. Reproduction retains their committed image assets; the
empirical figures must be computed from trace data.

## Methodology and interpretation

A smoke run exercises representative Cloudflare, Meta, and Wikimedia traces
with aggressive key sampling and a reduced simulation grid. It uses two-million
request prefixes for Cloudflare and Meta and a twenty-million-request Wikimedia
prefix; the latter must span the workload's fixed 24-hour TTL to produce expiry
events. It validates that the pipeline works on this machine, but its output is
not a reproduction of the full experiment or a substitute for the default full
run.
