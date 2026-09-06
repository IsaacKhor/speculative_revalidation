#!/usr/bin/env python3
"""Run the cache-revalidation experiments from raw traces to paper figures.

Repository files and the input trace tree are treated as read-only.  Every
path handed to a program as an output, as well as tool caches and temporary
files controlled by this driver, lives below ``reproduction/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


@dataclass(frozen=True)
class TraceSpec:
    dataset: str
    source_dir: str
    source_name: str
    sim_name: str
    input_format: str


TRACE_SPECS: tuple[TraceSpec, ...] = (
    *(TraceSpec("cf", "cdn_cf25_csv", f"{stem}.csv.zst", f"cf_{letter}", "cf")
      for stem, letter in zip(
          ("106m105", "106m106", "243m12", "243m13", "411m264",
           "411m325", "472m378", "472m379"),
          "abcdefgh",
          strict=True,
      )),
    *(TraceSpec("fb", "cdn_fb23_csv", f"{stem}.csv.zst", f"fb_{letter}", "fb23")
      for stem, letter in zip(("reag", "rhna", "rprn"), "abc", strict=True)),
    TraceSpec("wm", "cdn_wm19_csv", "t-all.csv.zst", "wm_t", "wm"),
    TraceSpec("wm", "cdn_wm19_csv", "u-all.csv.zst", "wm_u", "wm"),
)

SMOKE_NAMES = frozenset(("cf_b", "fb_a", "wm_t"))
FULL_CACHE_TYPES = ("lru", "gdsf", "sieve", "arc", "fifo")
FULL_ML_THRESHOLDS = ("1", "0.95", "0.9", "0.85", "0.8", "0.75", "0.7", "0.68", "0.66")
FULL_ORACLE_THRESHOLDS = ("1", "2", "3", "4")
VALIDATION_CACHE_TYPES = FULL_CACHE_TYPES
VALIDATION_ML_THRESHOLDS = FULL_ML_THRESHOLDS
VALIDATION_ORACLE_THRESHOLDS = FULL_ORACLE_THRESHOLDS
VALIDATION_KEY_SAMPLE_RATIO = 256
# Wikimedia's fixed 24-hour TTL produces no expiry events in its first two
# million requests, so retain a longer prefix for that workload only.
SMOKE_ROWS = {"cf_b": 2_000_000, "fb_a": 2_000_000, "wm_t": 20_000_000}


@dataclass(frozen=True)
class Layout:
    root: Path = HERE

    @property
    def traces(self) -> Path:
        return self.root / "traces"

    @property
    def sim_traces(self) -> Path:
        return self.traces / "sim"

    @property
    def expiry(self) -> Path:
        return self.traces / "expiry"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "output" / "figs"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def work(self) -> Path:
        return self.root / ".work"


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Smoke artifacts must never replace full-reproduction artifacts.
        self.layout = Layout(
            HERE / "validation" if args.validate
            else HERE / "smoke" if args.smoke
            else HERE
        )
        self.trace_root = args.trace_root.expanduser().resolve()
        self._step = 0
        self.env = os.environ.copy()
        shared_work = HERE / ".work"
        # Keep caches and temporary data controlled by common tools local.
        self.env.update({
            "MPLCONFIGDIR": str(self.layout.work / "mplconfig"),
            "TMPDIR": str(self.layout.work / "tmp"),
            "JOBLIB_TEMP_FOLDER": str(self.layout.work / "tmp"),
            "UV_CACHE_DIR": str(shared_work / "uv-cache"),
            "UV_PROJECT_ENVIRONMENT": str(shared_work / "python-env"),
            "UV_PYTHON_INSTALL_DIR": str(shared_work / "uv-python"),
            "XMAKE_GLOBALDIR": str(self.layout.work / "xmake-global"),
            "XMAKE_PKG_CACHEDIR": str(self.layout.work / "xmake-packages"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "42",
        })

    @property
    def python(self) -> Path:
        return HERE / ".work" / "python-env" / "bin" / "python"

    def announce(self, description: str, command: Sequence[object] | None = None) -> None:
        self._step += 1
        print(f"[{self._step:02d}] {description}")
        if command is not None:
            print("     " + shlex.join(str(part) for part in command))

    def mkdirs(self, paths: Iterable[Path]) -> None:
        if self.args.dry_run:
            return
        for path in paths:
            self._assert_output(path)
            path.mkdir(parents=True, exist_ok=True)

    def _assert_output(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        if resolved != HERE and HERE not in resolved.parents:
            raise RuntimeError(f"refusing to write outside {HERE}: {resolved}")

    def run(self, description: str, command: Sequence[object], *, cwd: Path | None = None) -> None:
        cmd = [str(part) for part in command]
        self.announce(description, cmd)
        if self.args.dry_run:
            return
        log_path = self.layout.logs / f"{self._step:02d}.log"
        self._assert_output(log_path)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + shlex.join(cmd) + "\n")
            log.flush()
            completed = subprocess.run(
                cmd,
                cwd=cwd or self.layout.root,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"step failed with exit status {completed.returncode}: {description}\n"
                f"see {log_path}"
            )

    def require_inputs(self, specs: Sequence[TraceSpec]) -> None:
        if not self.trace_root.is_dir():
            raise FileNotFoundError(f"trace root is not a directory: {self.trace_root}")
        missing = [self.source_path(spec) for spec in specs if not self.source_path(spec).is_file()]
        if missing:
            shown = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"missing required raw traces:\n{shown}")

    def require_tools(self, names: Sequence[str]) -> None:
        missing = [name for name in names if shutil.which(name) is None]
        if missing:
            raise RuntimeError("missing required command(s): " + ", ".join(missing))

    def _fingerprint(self, paths: Sequence[Path], extra: object = None) -> str:
        records = []
        for path in paths:
            try:
                stat = path.stat()
                records.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                records.append((str(path.resolve(strict=False)), None, None))
        blob = json.dumps((records, extra), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _marker_path(self, name: str) -> Path:
        safe_name = name.replace("/", "_")
        return self.layout.work / "done" / f"{safe_name}.json"

    def completed(
        self, name: str, outputs: Sequence[Path], inputs: Sequence[Path] = (), extra: object = None,
    ) -> bool:
        if not self.args.resume:
            return False
        marker = self._marker_path(name)
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        if record.get("fingerprint") != self._fingerprint(inputs, extra):
            return False
        return all(path.is_file() and path.stat().st_size > 0 for path in outputs)

    def mark_completed(
        self, name: str, outputs: Sequence[Path], inputs: Sequence[Path] = (), extra: object = None,
    ) -> None:
        if self.args.dry_run:
            return
        if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
            raise RuntimeError(f"stage {name} did not produce every expected non-empty output")
        marker = self._marker_path(name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "fingerprint": self._fingerprint(inputs, extra),
            "outputs": [str(path) for path in outputs],
        }, indent=2) + "\n", encoding="utf-8")

    def source_path(self, spec: TraceSpec) -> Path:
        return self.trace_root / spec.source_dir / spec.source_name

    def binary_path(self, spec: TraceSpec) -> Path:
        return self.layout.sim_traces / f"{spec.sim_name}.bin.zst"

    def count_path(self, spec: TraceSpec) -> Path:
        return Path(str(self.binary_path(spec)) + ".count")

    def expiry_path(self, spec: TraceSpec) -> Path:
        return self.layout.expiry / f"{spec.sim_name}.csv"

    def prepare(self) -> None:
        generated = (
            self.layout.traces,
            self.layout.models,
            self.layout.results,
            self.layout.output,
            self.layout.logs,
            self.layout.work,
        )
        if self.args.force:
            self.announce("remove prior generated artifacts")
            if not self.args.dry_run:
                for path in generated:
                    self._assert_output(path)
                    if path.exists():
                        shutil.rmtree(path)
        self.mkdirs((
            self.layout.sim_traces,
            self.layout.expiry,
            self.layout.models,
            self.layout.results,
            self.layout.figures,
            self.layout.logs,
            self.layout.work / "tmp",
            self.layout.work / "mplconfig",
        ))
        if not self.args.dry_run:
            manifest = self.layout.root / "run.json"
            self._assert_output(manifest)
            manifest.write_text(json.dumps({
                "mode": "validation" if self.args.validate else "smoke" if self.args.smoke else "full",
                "trace_root": str(self.trace_root),
                "key_sample_ratio": self.args.key_sample_ratio or (
                    VALIDATION_KEY_SAMPLE_RATIO if self.args.validate
                    else 8 if self.args.smoke else 1
                ),
                "parallel": self.args.parallel,
                "input_scope": (
                    "full trace" if self.args.validate
                    else SMOKE_ROWS if self.args.smoke
                    else "full trace"
                ),
                "preprocessing": (
                    "spatial key sample" if self.args.validate else "all selected rows"
                ),
                "cache_types": list(VALIDATION_CACHE_TYPES) if self.args.validate else None,
                "ml_thresholds": list(VALIDATION_ML_THRESHOLDS) if self.args.validate else None,
                "oracle_thresholds": list(VALIDATION_ORACLE_THRESHOLDS) if self.args.validate else None,
            }, indent=2) + "\n", encoding="utf-8")

    def bootstrap_python(self) -> None:
        if self.completed("python", (self.python,), (REPO_ROOT / "uv.lock", REPO_ROOT / "pyproject.toml")):
            self.announce("reuse local Python environment")
            return
        self.run(
            "create local locked Python environment",
            ("uv", "sync", "--frozen", "--no-install-project", "--project", REPO_ROOT),
            cwd=self.layout.root,
        )
        self.mark_completed("python", (self.python,), (REPO_ROOT / "uv.lock", REPO_ROOT / "pyproject.toml"))

    def build(self) -> tuple[Path, Path]:
        if self.args.sim_bin or self.args.oracle_bin:
            if not (self.args.sim_bin and self.args.oracle_bin):
                raise ValueError("--sim-bin and --oracle-bin must be supplied together")
            sim_bin = self.args.sim_bin.expanduser().resolve()
            oracle_bin = self.args.oracle_bin.expanduser().resolve()
            if not self.args.dry_run:
                for binary in (sim_bin, oracle_bin):
                    if not binary.is_file():
                        raise FileNotFoundError(binary)
            self.announce("use supplied simulator binaries")
            return sim_bin, oracle_bin

        stage = self.layout.work / "source"
        build_dir = self.layout.work / "build"
        built_sim = build_dir / "linux" / "x86_64" / "releasedbg" / "sim"
        built_oracle = build_dir / "linux" / "x86_64" / "releasedbg" / "oracle_backpass"
        build_inputs = tuple(sorted(
            path for path in (REPO_ROOT / "sim").iterdir()
            if path.suffix in (".cpp", ".h", ".hpp")
        )) + (REPO_ROOT / "xmake.lua",)
        build_config = {
            "smoke": self.args.smoke,
            "validate": self.args.validate,
            "smoke_rows": (
                SMOKE_ROWS if self.args.smoke and not self.args.validate else None
            ),
        }
        if self.completed("build", (built_sim, built_oracle), build_inputs, build_config):
            self._activate_packaged_runtime()
            self.announce("reuse completed isolated simulator build")
            return built_sim, built_oracle
        self.announce("stage simulator sources", (REPO_ROOT / "sim", stage / "sim"))
        if not self.args.dry_run:
            self._assert_output(stage)
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            shutil.copy2(REPO_ROOT / "xmake.lua", stage / "xmake.lua")
            shutil.copytree(REPO_ROOT / "sim", stage / "sim")
            # The repository build file pins fmt and ONNX Runtime to a
            # machine-wide installation.  In the staged copy, let xmake use
            # its reproduction-local package cache like the other packages.
            staged_build = stage / "xmake.lua"
            build_text = staged_build.read_text(encoding="utf-8")
            # Boost.Process v2 removed the compatibility header used by this
            # simulator; match the 1.83 ABI/header generation it was written
            # and previously built against.
            build_text = build_text.replace(
                'add_requires("boost", "zstd", "fmt", "abseil")',
                'add_requires("boost 1.83.0", '
                '{configs = {program_options = true}})\n'
                'add_requires("zstd", "fmt 9.1.0", "abseil")',
            )
            build_text = build_text.replace(
                'add_requires("fmt", {system = true})\nadd_packages("fmt")\n', ""
            )
            build_text = build_text.replace(
                'add_requires("libonnxruntime", {system = true})',
                'add_requires("onnxruntime")',
            )
            build_text = build_text.replace(
                'add_packages("libonnxruntime")', 'add_packages("onnxruntime")'
            )
            staged_build.write_text(build_text, encoding="utf-8")
            for source in (stage / "sim").glob("*.cpp"):
                source_text = source.read_text(encoding="utf-8").replace(
                    '"libonnxruntime/onnxruntime_cxx_api.h"',
                    '"onnxruntime_cxx_api.h"',
                )
                source.write_text(source_text, encoding="utf-8")
            if self.args.validate:
                # Sampled preprocessing writes the exact retained request
                # count next to each compressed trace.  Use it to preserve
                # proportional warm-up/cooldown windows over the full trace.
                sim_source = stage / "sim" / "sim.cpp"
                source_text = sim_source.read_text(encoding="utf-8")
                source_text = source_text.replace(
                    "#include <filesystem>\n",
                    "#include <filesystem>\n#include <fstream>\n",
                    1,
                )
                function_start = (
                    "static auto trace_uncompressed_reqs(strv base) -> u64\n"
                    "{\n"
                )
                replacement = (
                    "static auto trace_uncompressed_reqs(strv path) -> u64\n"
                    "{\n"
                    "    std::ifstream count_file(str(path) + \".count\");\n"
                    "    u64 sampled_count = 0;\n"
                    "    if (count_file >> sampled_count)\n"
                    "        return sampled_count;\n\n"
                    "    auto base = std::filesystem::path(path).filename().string();\n"
                    "    base = base.substr(0, base.find_first_of('.'));\n"
                )
                if function_start not in source_text:
                    raise RuntimeError("could not patch sampled trace count lookup")
                source_text = source_text.replace(function_start, replacement, 1)
                old_call = "trace_uncompressed_reqs(cfg.infile_base())"
                if old_call not in source_text:
                    raise RuntimeError("could not patch sampled stats window lookup")
                source_text = source_text.replace(
                    old_call, "trace_uncompressed_reqs(cfg.infile)", 1
                )
                old_minimum = (
                    "if (effective_requests < 1'000'000)\n"
                    "            FAIL(\"effective request window < 1,000,000 after warmup/cooldown\");"
                )
                new_minimum = (
                    "if (effective_requests < 10'000)\n"
                    "            FAIL(\"sampled effective request window < 10,000 after warmup/cooldown\");"
                )
                if old_minimum not in source_text:
                    raise RuntimeError("could not patch sampled request-window minimum")
                source_text = source_text.replace(old_minimum, new_minimum, 1)
                sim_source.write_text(source_text, encoding="utf-8")
            elif self.args.smoke:
                # Prefix traces contain exactly SMOKE_ROWS requests.  The
                # simulator otherwise uses a legacy basename-to-count table
                # when placing its warm-up and cooldown boundaries.
                sim_source = stage / "sim" / "sim.cpp"
                source_text = sim_source.read_text(encoding="utf-8")
                for name in SMOKE_NAMES:
                    pattern = rf'(\{{"{re.escape(name)}",\s*)\d+'
                    source_text, replacements = re.subn(
                        pattern,
                        rf'\g<1>{SMOKE_ROWS[name] // 1_000_000} /* smoke */',
                        source_text,
                        count=1,
                    )
                    if replacements != 1:
                        raise RuntimeError(
                            f"could not patch request count for smoke trace {name}"
                        )
                sim_source.write_text(source_text, encoding="utf-8")

        self.run(
            "configure isolated releasedbg build",
            # The staged project is rewritten for portable dependencies, so
            # discard xmake's prior requirement-resolution cache each time.
            ("xmake", "f", "-P", ".", "-c", "-y", "-m", "releasedbg", "-o", build_dir),
            cwd=stage,
        )
        self.run(
            "build simulator and reverse-pass tool",
            ("xmake", "-P", ".", "-y", "-v"),
            cwd=stage,
        )

        candidates = (
            build_dir / "linux" / "x86_64" / "releasedbg",
            build_dir / "linux" / "x86_64" / "release",
        )
        if self.args.dry_run:
            return candidates[0] / "sim", candidates[0] / "oracle_backpass"
        for directory in candidates:
            sim_bin, oracle_bin = directory / "sim", directory / "oracle_backpass"
            if sim_bin.is_file() and oracle_bin.is_file():
                self._activate_packaged_runtime()
                self.mark_completed("build", (sim_bin, oracle_bin), build_inputs, build_config)
                return sim_bin, oracle_bin
        raise FileNotFoundError(f"xmake completed but binaries were not found below {build_dir}")

    def _activate_packaged_runtime(self) -> None:
        """Make xmake-provided shared libraries visible to built binaries."""
        package_root = self.layout.work / "xmake-global" / ".xmake" / "packages"
        runtime_dirs = sorted({
            str(path.parent)
            for path in package_root.glob("o/onnxruntime/*/*/lib/libonnxruntime.so*")
        })
        if not runtime_dirs:
            raise FileNotFoundError(
                f"xmake ONNX Runtime shared library was not found below {package_root}"
            )
        inherited = self.env.get("LD_LIBRARY_PATH")
        if inherited:
            runtime_dirs.append(inherited)
        self.env["LD_LIBRARY_PATH"] = os.pathsep.join(runtime_dirs)

    def preprocess(self, specs: Sequence[TraceSpec], oracle_bin: Path, ksr: int) -> None:
        for spec in specs:
            source = self.source_path(spec)
            smoke_rows = (
                SMOKE_ROWS[spec.sim_name]
                if self.args.smoke and not self.args.validate else None
            )
            raw_binary = self.layout.work / "tmp" / f"{spec.sim_name}.bin"
            output = self.binary_path(spec)
            count_output = self.count_path(spec)
            marker_name = f"preprocess_{spec.sim_name}"
            preprocess_script = (
                HERE / "sampled_preprocess.py"
                if self.args.validate else REPO_ROOT / "sim" / "preprocess.py"
            )
            preprocess_inputs = (source, oracle_bin, preprocess_script)
            expected_outputs = (output, count_output) if self.args.validate else (output,)
            preprocess_config = {
                "rows": smoke_rows,
                "spatial_key_sample_ratio": (
                    ksr if self.args.validate else None
                ),
            }
            if self.completed(marker_name, expected_outputs, preprocess_inputs, preprocess_config):
                self.announce(f"reuse preprocessed trace {spec.sim_name}")
                continue
            zstd_cmd = ("zstdcat", source)
            if self.args.validate:
                preprocess_cmd = (
                    self.python, preprocess_script, "--format", spec.input_format,
                    "--key-sample-ratio", str(ksr),
                    "--count-file", count_output,
                )
            else:
                preprocess_cmd = (
                    self.python, preprocess_script, "--format", spec.input_format,
                )
            shown_command: tuple[object, ...] = (*zstd_cmd, "|")
            if smoke_rows is not None:
                shown_command += ("head", "-n", str(smoke_rows + 1), "|")
            shown_command += (*preprocess_cmd, ">", raw_binary)
            self.announce(
                f"preprocess {spec.sim_name}",
                shown_command,
            )
            if not self.args.dry_run:
                log_path = self.layout.logs / f"{self._step:02d}.log"
                with raw_binary.open("wb") as output_stream, log_path.open("wb") as log:
                    first = subprocess.Popen(zstd_cmd, stdout=subprocess.PIPE, stderr=log, env=self.env)
                    assert first.stdout is not None
                    input_stream = first.stdout
                    limiter = None
                    if smoke_rows is not None:
                        limiter = subprocess.Popen(
                            ("head", "-n", str(smoke_rows + 1)),
                            stdin=first.stdout,
                            stdout=subprocess.PIPE,
                            stderr=log,
                            env=self.env,
                        )
                        first.stdout.close()
                        assert limiter.stdout is not None
                        input_stream = limiter.stdout
                    second = subprocess.Popen(
                        [str(part) for part in preprocess_cmd],
                        stdin=input_stream,
                        stdout=output_stream,
                        stderr=log,
                        cwd=self.layout.root,
                        env=self.env,
                    )
                    input_stream.close()
                    second_status = second.wait()
                    limiter_status = limiter.wait() if limiter is not None else 0
                    first_status = first.wait()
                # zstdcat reports a broken pipe after head deliberately closes
                # a smoke prefix.  The limiter and preprocessor statuses are
                # authoritative for that path.
                first_failed = bool(first_status) and smoke_rows is None
                if first_failed or limiter_status or second_status:
                    raw_binary.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"preprocessing {source} failed (zstdcat={first_status}, "
                        f"limiter={limiter_status}, preprocessor={second_status}); see {log_path}"
                    )
            self.run(f"compute next-access metadata for {spec.sim_name}",
                     (oracle_bin, raw_binary))
            self.run(f"compress binary trace {spec.sim_name}",
                     ("zstd", "-T0", "-f", raw_binary, "-o", output))
            if not self.args.dry_run:
                raw_binary.unlink(missing_ok=True)
            self.mark_completed(
                marker_name, expected_outputs, preprocess_inputs, preprocess_config
            )

    def simulator(self, description: str, sim_bin: Path, args: Sequence[object]) -> None:
        # Running from reproduction/ contains the simulator's two hard-coded
        # relative outputs: traces/expiry and results/zonestats.
        self.run(description, (sim_bin, *args), cwd=self.layout.root)

    def generate_expiry(self, specs: Sequence[TraceSpec], sim_bin: Path, ksr: int) -> None:
        outputs = tuple(self.expiry_path(spec) for spec in specs)
        inputs = tuple(self.binary_path(spec) for spec in specs)
        if self.args.validate:
            inputs += tuple(self.count_path(spec) for spec in specs)
        inputs += (sim_bin,)
        if self.completed("expiry", outputs, inputs, {"ksr": ksr}):
            self.announce("reuse completed expiry-event traces")
            return
        self.simulator(
            "generate fresh expiry-event traces",
            sim_bin,
            (
                f"--key-sample-ratio={ksr}", f"--parallel={self.args.parallel}",
                "--capacity=2048", *(f"--input-file={self.binary_path(spec)}" for spec in specs),
                "--trace-expiry=true", "--rv-mode=oracle", "--ml-conf-thres=1",
                f"--csvout={self.layout.results / 'expiry_generation.csv'}",
            ),
        )
        self.mark_completed("expiry", outputs, inputs, {"ksr": ksr})

    def train_models(self, specs: Sequence[TraceSpec]) -> dict[str, Path]:
        models: dict[str, Path] = {}
        for dataset in ("cf", "fb", "wm"):
            selected = [spec for spec in specs if spec.dataset == dataset]
            if not selected:
                continue
            prefix = self.layout.models / f"fresh_{dataset}_rfc"
            models[dataset] = prefix.with_suffix(".onnx")
            outputs = (models[dataset], prefix.with_suffix(".pkl"))
            training_inputs = tuple(self.expiry_path(spec) for spec in selected) + (
                REPO_ROOT / "sim" / "train_ml.py", HERE / "seeded_train.py",
            )
            sample_frac = (
                "1" if self.args.smoke or self.args.validate
                else {"cf": "0.1", "fb": "0.33", "wm": "0.33"}[dataset]
            )
            if self.completed(f"model_{dataset}", outputs, training_inputs, {"sample_frac": sample_frac}):
                self.announce(f"reuse trained {dataset.upper()} model")
                continue
            self.run(
                f"train {dataset.upper()} random-forest model",
                (
                    self.python, HERE / "seeded_train.py",
                    "--inputs", *(self.expiry_path(spec) for spec in selected),
                    "--output", prefix, "--sample-frac", sample_frac,
                ),
            )
            self.mark_completed(f"model_{dataset}", outputs, training_inputs, {"sample_frac": sample_frac})
        return models

    def run_experiments(
        self,
        specs: Sequence[TraceSpec],
        sim_bin: Path,
        models: dict[str, Path],
        ksr: int,
    ) -> None:
        trace_args = tuple(f"--input-file={self.binary_path(spec)}" for spec in specs)
        common = (
            f"--key-sample-ratio={ksr}", f"--parallel={self.args.parallel}",
            "--capacity=2048", *trace_args,
        )
        cache_types = (
            VALIDATION_CACHE_TYPES if self.args.validate
            else ("lru",) if self.args.smoke
            else FULL_CACHE_TYPES
        )
        cache_args = tuple(f"--cache-type={kind}" for kind in cache_types)

        baseline = self.layout.results / "baseline.csv"
        experiment_inputs = tuple(self.binary_path(spec) for spec in specs)
        if self.args.validate:
            experiment_inputs += tuple(self.count_path(spec) for spec in specs)
        experiment_inputs += (sim_bin,)
        if self.completed("result_baseline", (baseline,), experiment_inputs, {"ksr": ksr, "caches": cache_types}):
            self.announce("reuse no-revalidation baselines")
        else:
            self.simulator(
                "run no-revalidation baselines", sim_bin,
                (*common, *cache_args, "--rv-mode=never", f"--csvout={baseline}"),
            )
            self.mark_completed("result_baseline", (baseline,), experiment_inputs, {"ksr": ksr, "caches": cache_types})
        baseline_keep = self.layout.results / "baseline_keep.csv"
        if self.completed("result_baseline_keep", (baseline_keep,), experiment_inputs, {"ksr": ksr, "caches": cache_types}):
            self.announce("reuse baselines that retain expired objects")
        else:
            self.simulator(
                "run baselines that retain expired objects", sim_bin,
                (*common, *cache_args, "--rv-mode=never", "--evict-expired=false",
                 f"--csvout={baseline_keep}"),
            )
            self.mark_completed("result_baseline_keep", (baseline_keep,), experiment_inputs, {"ksr": ksr, "caches": cache_types})
        oracle_thresholds = (
            VALIDATION_ORACLE_THRESHOLDS if self.args.validate
            else ("1",) if self.args.smoke
            else FULL_ORACLE_THRESHOLDS
        )
        oracle = self.layout.results / "oracle.csv"
        if self.completed("result_oracle", (oracle,), experiment_inputs,
                          {"ksr": ksr, "caches": cache_types, "thresholds": oracle_thresholds}):
            self.announce("reuse oracle bounds")
        else:
            self.simulator(
                "run oracle bounds", sim_bin,
                (*common, *cache_args, "--rv-mode=oracle",
                 *(f"--ml-conf-thres={value}" for value in oracle_thresholds),
                 f"--csvout={oracle}"),
            )
            self.mark_completed("result_oracle", (oracle,), experiment_inputs,
                                {"ksr": ksr, "caches": cache_types, "thresholds": oracle_thresholds})

        ml_thresholds = (
            VALIDATION_ML_THRESHOLDS if self.args.validate
            else ("1", "0.8") if self.args.smoke
            else FULL_ML_THRESHOLDS
        )
        for dataset, model in models.items():
            dataset_specs = [spec for spec in specs if spec.dataset == dataset]
            if not dataset_specs:
                continue
            ml_output = self.layout.results / f"ml_{dataset}_rfc.csv"
            ml_inputs = tuple(self.binary_path(spec) for spec in dataset_specs)
            if self.args.validate:
                ml_inputs += tuple(self.count_path(spec) for spec in dataset_specs)
            ml_inputs += (sim_bin, model)
            if self.completed(f"result_ml_{dataset}", (ml_output,), ml_inputs,
                              {"ksr": ksr, "caches": cache_types, "thresholds": ml_thresholds}):
                self.announce(f"reuse {dataset.upper()} ML threshold sweep")
                continue
            self.simulator(
                f"run {dataset.upper()} ML threshold sweep", sim_bin,
                (
                    f"--key-sample-ratio={ksr}", f"--parallel={self.args.parallel}",
                    "--capacity=2048",
                    *(f"--input-file={self.binary_path(spec)}" for spec in dataset_specs),
                    *cache_args, "--rv-mode=ml", f"--ml-model-path={model}",
                    *(f"--ml-conf-thres={value}" for value in ml_thresholds),
                    f"--csvout={ml_output}",
                ),
            )
            self.mark_completed(f"result_ml_{dataset}", (ml_output,), ml_inputs,
                                {"ksr": ksr, "caches": cache_types, "thresholds": ml_thresholds})

        if self.args.validate or not self.args.smoke:
            self._run_extended_ml(sim_bin, specs, models, ksr)

        time_specs = [spec for spec in specs if spec.sim_name in (
            ("cf_b",) if self.args.smoke or self.args.validate
            else ("cf_b", "cf_c", "cf_f", "cf_h")
        )]
        if time_specs:
            timeseries_csv = self.layout.results / "ml_timeseries.csv"
            timeseries_dir = self.layout.results / "r10i3_stats_ts"
            thresholds = (
                VALIDATION_ML_THRESHOLDS if self.args.validate
                else ("1", "0.8") if self.args.smoke
                else FULL_ML_THRESHOLDS
            )
            ts_files = tuple(timeseries_dir.glob("*_stats_ts.csv")) if timeseries_dir.is_dir() else ()
            ts_inputs = tuple(self.binary_path(spec) for spec in time_specs)
            if self.args.validate:
                ts_inputs += tuple(self.count_path(spec) for spec in time_specs)
            ts_inputs += (sim_bin, models["cf"])
            expected_ts_files = len(time_specs) * len(thresholds)
            if len(ts_files) == expected_ts_files and self.completed(
                "result_timeseries", (timeseries_csv, *ts_files), ts_inputs,
                {"ksr": ksr, "thresholds": thresholds},
            ):
                self.announce("reuse miss-over-time simulation")
            else:
                self.simulator(
                    "run CF miss-over-time simulation", sim_bin,
                    (
                        f"--key-sample-ratio={ksr}", f"--parallel={self.args.parallel}",
                        "--capacity=2048",
                        *(f"--input-file={self.binary_path(spec)}" for spec in time_specs),
                        "--cache-type=lru", "--rv-mode=ml",
                        f"--ml-model-path={models['cf']}",
                        *(f"--ml-conf-thres={value}" for value in thresholds),
                        f"--dump-stats-ts={timeseries_dir}",
                        f"--csvout={timeseries_csv}",
                    ),
                )
                produced_ts = tuple(timeseries_dir.glob("*_stats_ts.csv"))
                if not self.args.dry_run and len(produced_ts) != expected_ts_files:
                    raise RuntimeError(
                        f"miss-over-time run produced {len(produced_ts)} time-series files; "
                        f"expected {expected_ts_files}"
                    )
                self.mark_completed("result_timeseries", (timeseries_csv, *produced_ts), ts_inputs,
                                    {"ksr": ksr, "thresholds": thresholds})

    def _run_extended_ml(
        self, sim_bin: Path, specs: Sequence[TraceSpec], models: dict[str, Path], ksr: int,
    ) -> None:
        sweeps = (
            ("fb", "ml_fb_rfc_extended2.csv", ("0.89", "0.88", "0.87", "0.86", "0.84", "0.83", "0.82", "0.81")),
            ("wm", "ml_wm_rfc_extended.csv", ("0.64", "0.62", "0.6", "0.58", "0.56")),
            ("wm", "ml_wm_rfc_extended2.csv", ("0.54", "0.52", "0.5")),
        )
        for dataset, filename, thresholds in sweeps:
            output = self.layout.results / filename
            selected = [spec for spec in specs if spec.dataset == dataset]
            extended_inputs = tuple(self.binary_path(spec) for spec in selected)
            if self.args.validate:
                extended_inputs += tuple(self.count_path(spec) for spec in selected)
            extended_inputs += (sim_bin, models[dataset])
            marker_name = "result_" + filename.removesuffix(".csv")
            if self.completed(marker_name, (output,), extended_inputs, {"ksr": ksr, "thresholds": thresholds}):
                self.announce(f"reuse extended {dataset.upper()} sweep {filename}")
                continue
            self.simulator(
                f"run extended {dataset.upper()} threshold sweep", sim_bin,
                (
                    f"--key-sample-ratio={ksr}", f"--parallel={self.args.parallel}",
                    "--capacity=2048",
                    *(f"--input-file={self.binary_path(spec)}" for spec in selected),
                    "--cache-type=lru", "--rv-mode=ml",
                    f"--ml-model-path={models[dataset]}",
                    *(f"--ml-conf-thres={value}" for value in thresholds),
                    f"--csvout={output}",
                ),
            )
            self.mark_completed(marker_name, (output,), extended_inputs, {"ksr": ksr, "thresholds": thresholds})

    def plot(self) -> None:
        figures_script = HERE / "figures.py"
        if not self.args.dry_run and not figures_script.is_file():
            raise FileNotFoundError(f"plotting driver is missing: {figures_script}")
        self.run(
            "regenerate empirical figures",
            (
                self.python, figures_script, "all",
                "--trace-root", self.trace_root,
                "--expiry-dir", self.layout.expiry,
                "--results-dir", self.layout.results,
                "--models-dir", self.layout.models,
                "--output-dir", self.layout.figures,
                "--sources", *(spec.sim_name for spec in self.selected_specs()),
            ),
        )
        self.copy_static_artifacts()

    def validate_results(self) -> None:
        ksr = self.args.key_sample_ratio or VALIDATION_KEY_SAMPLE_RATIO
        self.run(
            "validate sampled results against committed full-run results",
            (
                self.python, HERE / "validate_results.py",
                "--candidate-dir", self.layout.results,
                "--reference-dir", REPO_ROOT / "results",
                "--output-dir", self.layout.output,
                "--traces", *(spec.sim_name for spec in self.selected_specs()),
                "--cache-types", *VALIDATION_CACHE_TYPES,
                "--ml-thresholds", *VALIDATION_ML_THRESHOLDS,
                "--oracle-thresholds", *VALIDATION_ORACLE_THRESHOLDS,
                "--capacity-gib", "2048",
                "--key-sample-ratio", str(ksr),
            ),
        )

    def selected_specs(self) -> tuple[TraceSpec, ...]:
        return tuple(
            spec for spec in TRACE_SPECS
            if not (self.args.smoke or self.args.validate)
            or spec.sim_name in SMOKE_NAMES
        )

    def copy_static_artifacts(self) -> None:
        diagrams_dir = self.layout.output / "diagrams"
        diagram_sources = REPO_ROOT / "paper" / "v2-eurosys27" / "diagrams"
        self.announce("copy authored diagrams and write output manifest")
        if self.args.dry_run:
            return
        self._assert_output(diagrams_dir)
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        for name in ("reval_arch.png", "timeline.png"):
            shutil.copy2(diagram_sources / name, diagrams_dir / name)
        manifest_path = self.layout.output / "manifest.json"
        self._assert_output(manifest_path)
        files = sorted(
            path for path in self.layout.output.rglob("*")
            if path.is_file() and path != manifest_path
        )
        manifest = {
            str(path.relative_to(self.layout.output)): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def execute(self) -> None:
        specs = self.selected_specs()
        if self.args.plots_only:
            self.prepare()
            self.bootstrap_python()
            if self.args.validate:
                self.validate_results()
            self.plot()
            return

        if not self.args.dry_run:
            self.require_inputs(specs)
            required = ["uv", "zstd", "zstdcat"]
            if self.args.smoke and not self.args.validate:
                required.append("head")
            if not (self.args.sim_bin and self.args.oracle_bin):
                required.append("xmake")
            self.require_tools(required)
        self.prepare()
        self.bootstrap_python()
        sim_bin, oracle_bin = self.build()
        # Validation prefilters complete traces spatially, preserving their
        # time span while greatly reducing the materialized binary traces.
        ksr = self.args.key_sample_ratio or (
            VALIDATION_KEY_SAMPLE_RATIO if self.args.validate
            else 8 if self.args.smoke else 1
        )
        self.preprocess(specs, oracle_bin, ksr)
        self.generate_expiry(specs, sim_bin, ksr)
        models = self.train_models(specs)
        self.run_experiments(specs, sim_bin, models, ksr)
        if self.args.validate:
            self.validate_results()
        self.plot()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute simulator traces, models, experiment results, and paper figures.",
    )
    parser.add_argument(
        "--trace-root", type=Path, default=REPO_ROOT / "traces",
        help="read-only root containing cdn_cf25_csv, cdn_fb23_csv, and cdn_wm19_csv",
    )
    parser.add_argument("--parallel", type=int, default=max(1, min(8, os.cpu_count() or 1)),
                        help="parallel simulator configurations (default: up to 8)")
    parser.add_argument("--key-sample-ratio", type=int,
                        help="override key sampling ratio (full: 1; smoke: 8; validation: 256)")
    sampled_mode = parser.add_mutually_exclusive_group()
    sampled_mode.add_argument("--smoke", action="store_true",
                              help="run one trace per dataset with high sampling and a small grid")
    sampled_mode.add_argument(
        "--validate", action="store_true",
        help="run a broader sampled grid and compare it with committed full-run results",
    )
    parser.add_argument("--plots-only", action="store_true",
                        help="regenerate plots from reproduction/results and reproduction/models")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the complete plan without writing or executing commands")
    run_policy = parser.add_mutually_exclusive_group()
    run_policy.add_argument("--resume", action="store_true",
                            help="reuse completed artifacts and continue missing stages")
    run_policy.add_argument("--force", action="store_true",
                            help="remove this mode's prior generated artifacts before running")
    parser.add_argument("--sim-bin", type=Path,
                        help="use an existing simulator binary (requires --oracle-bin)")
    parser.add_argument("--oracle-bin", type=Path,
                        help="use an existing oracle_backpass binary (requires --sim-bin)")
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.key_sample_ratio is not None and args.key_sample_ratio < 1:
        parser.error("--key-sample-ratio must be at least 1")
    if args.plots_only and args.force:
        parser.error("--plots-only cannot be combined with --force (it would remove plot inputs)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        Pipeline(args).execute()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
