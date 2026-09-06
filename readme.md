# Speculative revalidation

The EuroSys 2027 paper, simulator, analysis, and artifact-reproduction workflow
live in this repository. Detailed implementation and resource notes are in
[`reproduction/README.md`](reproduction/README.md).

## Reproducing the artifact

Use Linux with Python 3.11+, `curl`, `gzip`, GNU `cut`/`tail`, `uv`, a C++20
compiler, xmake, and `zstd`/`zstdcat`. The public downloads are approximately
73 GiB and the formatted inputs approximately 66 GiB. The full simulation also
needs substantial temporary disk and memory.

1. Inspect and then run the restart-safe trace downloader. It reuses completed
   objects and resumes partial objects where the publisher supports ranges. It obtains the
   Cloudflare 2025, Meta 2023, and Wikimedia 2019 traces from the public URLs
   documented by the [Cloudflare](https://objects.research.cloudflare.com/@ikhor/cdn-traces/readme.md),
   [Meta](https://s3.amazonaws.com/cache-datasets/index.html#cache_dataset_txt/2023_metaCDN/),
   and [Wikimedia](https://analytics.wikimedia.org/published/datasets/caching/2019/)
   publishers and writes the exact layout expected by the simulator:

   ```console
   python3 reproduction/download_traces.py --dry-run
   python3 reproduction/download_traces.py
   ```

2. Run the full-duration, KSR-256 validation on one trace from each workload.
   It exercises every cache policy and paper threshold while remaining much
   smaller than the complete experiment:

   ```console
   python3 reproduction/reproduce.py --validate --resume
   ```

   A successful run reports `PASS`, zero hard errors, and the complete expected
   grid in `reproduction/validation/output/validation.md`. Numerical deltas in
   that report are informational because it uses spatial sampling and freshly
   trained per-trace models.

3. Run the complete experiment and generate all active paper figures:

   ```console
   python3 reproduction/reproduce.py --resume
   ```

   All generated build products, models, measurements, and plots remain under
   `reproduction/`; the committed paper and reference results are not replaced.

## What to compare

The following pairs are the verification targets for a full run:

| Generated artifact | Committed reference | Verification |
| --- | --- | --- |
| `reproduction/output/figs/*.png` | the same active filenames under `paper/v2-eurosys27/figs/` | Compare all ten empirical plots visually and by their plotted values. |
