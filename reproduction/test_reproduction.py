"""Cheap CLI and inventory checks; no full traces or simulations are run.

Run with: python3 -B -m unittest discover -s reproduction -p 'test_*.py' -v
"""

from pathlib import Path
from contextlib import redirect_stdout
import ast
import gzip
import io
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "reproduction" / "reproduce.py"
DOWNLOADER = ROOT / "reproduction" / "download_traces.py"
PAPER = ROOT / "paper" / "v2-eurosys27"
sys.path.insert(0, str(ROOT / "reproduction"))
import download_traces
import reproduce


def paper_images():
    images = set()
    for source in PAPER.glob("*.tex"):
        # Ignore commented examples and inactive figures in the manuscript.
        text = "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0]
                         for line in source.read_text().splitlines())
        images.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))
    return images


class PaperInventoryTests(unittest.TestCase):
    def test_active_paper_inventory(self):
        images = paper_images()
        self.assertEqual(len(images), 12, sorted(images))
        self.assertEqual(sum(p.startswith("figs/") for p in images), 10)
        self.assertEqual({p for p in images if p.startswith("diagrams/")}, {
            "diagrams/reval_arch.png", "diagrams/timeline.png",
        })
        self.assertNotIn("figs/placeholder.png", images)

    def test_plotting_inventory_covers_every_empirical_figure(self):
        # Parse the declaration without importing plotting libraries or creating
        # their runtime caches. This check has no scientific-Python dependency.
        module = ast.parse((reproduce.HERE / "figures.py").read_text())
        assignment = next(node for node in module.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name)
                                  and target.id == "ACTIVE_FIGURES"
                                  for target in node.targets))
        names = ast.literal_eval(assignment.value)
        expected = {Path(path).stem for path in paper_images() if path.startswith("figs/")}
        self.assertEqual(set(names), expected)
        self.assertEqual(len(names), len(set(names)))

    @unittest.skipUnless(
        (ROOT / "reproduction" / ".work" / "python-env" / "bin" / "python").is_file()
        and shutil.which("zstd"),
        "locked plotting environment or zstd is unavailable",
    )
    def test_ttl_figure_can_read_downloaded_csv_without_parquet(self):
        python = ROOT / "reproduction" / ".work" / "python-env" / "bin" / "python"
        with tempfile.TemporaryDirectory(prefix="test-csv-figure-", dir=reproduce.HERE) as directory:
            trace_root = Path(directory) / "traces"
            cf_dir = trace_root / "cdn_cf25_csv"
            fb_dir = trace_root / "cdn_fb23_csv"
            cf_dir.mkdir(parents=True)
            fb_dir.mkdir(parents=True)
            fixtures = (
                (cf_dir / "106m106.csv", (
                    "timestamp,key,zone,size,expiry_time,stale_time,method,mime\n"
                    "1,a,1,10,60,0,0,text/plain\n"
                    "2,b,1,20,120,0,0,text/plain\n"
                )),
                (fb_dir / "reag.csv", (
                    download_traces.META_HEADER.decode() + "\n"
                    "1,a,1,10,10,0,0,0,60,0,0,0,0,1,0\n"
                    "2,b,1,20,20,0,0,0,120,0,0,0,0,1,0\n"
                )),
            )
            for plain, content in fixtures:
                plain.write_text(content, encoding="utf-8")
                subprocess.run(
                    ("zstd", "-q", "-f", plain, "-o", Path(str(plain) + ".zst")),
                    check=True,
                )
            output_dir = Path(directory) / "output"
            env = dict(os.environ, MPLCONFIGDIR=str(Path(directory) / "mplconfig"))
            result = subprocess.run(
                (python, reproduce.HERE / "figures.py", "ttl_cdf_by_key",
                 "--trace-root", trace_root, "--results-dir", Path(directory),
                 "--models-dir", Path(directory), "--output-dir", output_dir,
                 "--sources", "cf_b", "fb_a"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=30, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater((output_dir / "ttl_cdf_by_key.png").stat().st_size, 0)


class RunnerCliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [sys.executable, "-B", str(RUNNER), *args],
            cwd=ROOT / "reproduction", env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )

    def test_help_from_outside_repository_root(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--dry-run", "--smoke", "--plots-only"):
            self.assertIn(flag, result.stdout)

    def test_dry_run(self):
        result = self.run_cli("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())

    def test_smoke_dry_run(self):
        result = self.run_cli("--smoke", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())

    def test_validation_dry_run(self):
        result = self.run_cli("--validate", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validate_results.py", result.stdout)
        self.assertIn("sampled_preprocess.py", result.stdout)
        self.assertIn("--key-sample-ratio 256", result.stdout)
        self.assertIn("--cache-type=sieve", result.stdout)
        self.assertIn("--ml-conf-thres=0.7", result.stdout)
        self.assertNotIn(" head ", " " + result.stdout.replace("\n", " ") + " ")

    def test_invalid_parallelism_is_rejected(self):
        result = self.run_cli("--parallel", "0", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--parallel", result.stderr)

    def test_binary_overrides_must_be_paired(self):
        result = self.run_cli("--sim-bin", "/nonexistent/sim", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--oracle-bin", result.stderr)


class TraceDownloaderTests(unittest.TestCase):
    def test_download_inventory_matches_reproduction_inputs(self):
        outputs = {
            Path(spec.relative_output).name
            for spec in download_traces.DIRECT_TRACES
        }
        outputs.update(spec.output_name for spec in download_traces.WM_TRACES)
        expected = {spec.source_name for spec in reproduce.TRACE_SPECS}
        self.assertEqual(outputs, expected)

    def test_download_dry_run_lists_sources_without_writing(self):
        with tempfile.TemporaryDirectory(prefix="test-download-plan-", dir=reproduce.HERE) as directory:
            target = Path(directory) / "absent"
            result = subprocess.run(
                [sys.executable, "-B", str(DOWNLOADER), "--dry-run",
                 "--trace-root", str(target)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(target.exists())
            self.assertIn("106m105.csv.zst", result.stdout)
            self.assertIn("rnha0c01_20230315_20230322_0.8000.csv.zst", result.stdout)
            self.assertIn("cache-t-01.gz", result.stdout)
            self.assertNotIn("cache-t-00.gz", result.stdout)
            self.assertIn("cache-u-00.gz", result.stdout)
            self.assertIn("cache-u-20.gz", result.stdout)

    @unittest.skipUnless(all(shutil.which(name) for name in ("gzip", "tail", "cut", "zstd")),
                         "trace-formatting commands are unavailable")
    def test_wikimedia_formatter_projects_and_assembles_csv(self):
        with tempfile.TemporaryDirectory(prefix="test-wm-format-", dir=reproduce.HERE) as directory:
            root = Path(directory)
            downloader = download_traces.Downloader(
                download_traces.parse_args(["--trace-root", str(root)])
            )
            cases = (
                (download_traces.WM_TRACES[0], (
                    b"10\t111\t1234\t0.1\n", b"11\t222\t5678\t0.2\n",
                ), "timestamp,key,size\n10,111,1234\n11,222,5678\n"),
                (download_traces.WM_TRACES[1], (
                    b"20\t333\tpng\t42\t0.3\n", b"21\t444\tjpeg\t84\t0.4\n",
                ), "timestamp,key,size\n20,333,42\n21,444,84\n"),
            )
            for index, (spec, rows, expected) in enumerate(cases):
                raw = root / f"raw-{index}.gz"
                fragment = root / f"fragment-{index}.zst"
                output = root / f"output-{index}.zst"
                with gzip.open(raw, "wb") as stream:
                    stream.write(spec.source_header + b"\n")
                    stream.writelines(rows)
                downloader.format_wikimedia_day(raw, fragment, spec)
                downloader.assemble_wikimedia(output, (fragment,), (f"test-{index}",))
                result = subprocess.run(
                    ("zstd", "-dcq", output), check=True,
                    stdout=subprocess.PIPE, text=True,
                )
                self.assertEqual(result.stdout, expected)


class PipelineTests(unittest.TestCase):
    def make_pipeline(self, *args):
        return reproduce.Pipeline(reproduce.parse_args(["--dry-run", *args]))

    def test_current_trace_mappings(self):
        mappings = {spec.sim_name: (spec.source_dir, spec.source_name)
                    for spec in reproduce.TRACE_SPECS}
        expected = {}
        for letter, server in zip("abcdefgh", (
            "106m105", "106m106", "243m12", "243m13", "411m264",
            "411m325", "472m378", "472m379",
        )):
            expected[f"cf_{letter}"] = ("cdn_cf25_csv", f"{server}.csv.zst")
        for letter, stem in zip("abc", ("reag", "rhna", "rprn")):
            expected[f"fb_{letter}"] = ("cdn_fb23_csv", f"{stem}.csv.zst")
        expected.update(wm_t=("cdn_wm19_csv", "t-all.csv.zst"),
                        wm_u=("cdn_wm19_csv", "u-all.csv.zst"))
        self.assertEqual(mappings, expected)

    def test_outputs_are_confined(self):
        pipeline = self.make_pipeline()
        for name in ("traces", "sim_traces", "expiry", "models", "results",
                     "figures", "logs", "work"):
            output = getattr(pipeline.layout, name)
            pipeline._assert_output(output)
            self.assertTrue(output.resolve().is_relative_to(reproduce.HERE))
        with self.assertRaises(RuntimeError):
            pipeline._assert_output(ROOT / "results" / "must-not-write.csv")
        with self.assertRaises(RuntimeError):
            pipeline._assert_output(reproduce.HERE / ".." / "must-not-write")

    def test_smoke_outputs_are_separate_from_full_outputs(self):
        full = self.make_pipeline()
        smoke = self.make_pipeline("--smoke")
        self.assertNotEqual(full.layout.root, smoke.layout.root)
        for name in ("sim_traces", "expiry", "models", "results", "figures", "work"):
            smoke_path = getattr(smoke.layout, name)
            full_path = getattr(full.layout, name)
            self.assertNotEqual(smoke_path, full_path)
            smoke._assert_output(smoke_path)

    def test_validation_outputs_are_separate_and_grid_is_broader(self):
        full = self.make_pipeline()
        smoke = self.make_pipeline("--smoke")
        validation = self.make_pipeline("--validate")
        self.assertNotEqual(validation.layout.root, full.layout.root)
        self.assertNotEqual(validation.layout.root, smoke.layout.root)
        with redirect_stdout(io.StringIO()) as output:
            validation.execute()
        plan = output.getvalue()
        self.assertIn("--cache-type=lru", plan)
        self.assertIn("--cache-type=sieve", plan)
        self.assertIn("--ml-conf-thres=0.9", plan)
        self.assertIn("--oracle-thresholds 1 2", plan)
        self.assertFalse(validation.args.smoke)
        for cache_type in reproduce.FULL_CACHE_TYPES:
            self.assertIn(f"--cache-type={cache_type}", plan)
        self.assertNotIn(" head ", " " + plan.replace("\n", " ") + " ")

    def test_spatial_preprocessor_retains_complete_selected_key_histories(self):
        script = reproduce.HERE / "sampled_preprocess.py"
        csv_data = (
            "timestamp,key,zone,size,ttl,ttstale,purge,mime\n"
            "100,1,0,10,20,0,0,text/plain\n"
            "110,2,0,11,20,0,0,text/plain\n"
            "120,1,0,12,20,0,0,text/plain\n"
            "130,4,0,13,20,0,0,text/plain\n"
        )
        with tempfile.TemporaryDirectory(prefix="test-spatial-", dir=reproduce.HERE) as directory:
            count_file = Path(directory) / "sample.bin.zst.count"
            result = subprocess.run(
                [sys.executable, "-B", str(script), "--format", "cf",
                 "--key-sample-ratio", "2", "--count-file", str(count_file)],
                input=csv_data.encode(), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            packer = struct.Struct("@QQQIIIII?3x")
            self.assertEqual(len(result.stdout), 2 * packer.size)
            rows = [packer.unpack_from(result.stdout, offset)
                    for offset in range(0, len(result.stdout), packer.size)]
            self.assertEqual([row[0] for row in rows], [2, 4])
            self.assertEqual([row[3] for row in rows], [11, 31])
            self.assertEqual(count_file.read_text(), "2\n")

    def test_symlink_escape_is_rejected(self):
        pipeline = self.make_pipeline()
        with tempfile.TemporaryDirectory(prefix="test-confinement-", dir=reproduce.HERE) as directory:
            link = Path(directory) / "escape"
            link.symlink_to(ROOT, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                pipeline._assert_output(link / "results" / "must-not-write.csv")

    def test_missing_trace_diagnostic_names_inputs(self):
        pipeline = self.make_pipeline("--trace-root", str(reproduce.HERE))
        with self.assertRaises(FileNotFoundError) as raised:
            pipeline.require_inputs(reproduce.TRACE_SPECS[:1])
        self.assertIn("cdn_cf25_csv", str(raised.exception))
        self.assertIn("106m105.csv.zst", str(raised.exception))

    def test_dry_run_does_not_launch_commands_or_make_directories(self):
        pipeline = self.make_pipeline()
        with patch.object(Path, "mkdir", side_effect=AssertionError("dry run wrote a directory")), \
             patch.object(subprocess, "run", side_effect=AssertionError("dry run ran a process")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("dry run ran a process")), \
             redirect_stdout(io.StringIO()) as output:
            pipeline.execute()
        plan = output.getvalue()
        self.assertIn("train", plan)
        self.assertIn("oracle", plan)
        self.assertIn("figures.py", plan)

    def test_smoke_selects_representative_workloads(self):
        self.assertEqual(reproduce.SMOKE_NAMES, {"cf_b", "fb_a", "wm_t"})
        pipeline = self.make_pipeline("--smoke")
        with redirect_stdout(io.StringIO()) as output:
            pipeline.execute()
        plan = output.getvalue()
        for name in reproduce.SMOKE_NAMES:
            self.assertIn(name + ".bin.zst", plan)
        for name in ("cf_a", "fb_b", "wm_u"):
            self.assertNotIn(name + ".bin.zst", plan)
        self.assertNotIn("--cache-type=arc", plan)

    def test_simulator_runs_with_contained_working_directory(self):
        pipeline = self.make_pipeline()
        with patch.object(pipeline, "run") as run:
            pipeline.generate_expiry(reproduce.TRACE_SPECS[:1], Path("/read-only/sim"), 1)
        command = run.call_args.args[1]
        self.assertEqual(run.call_args.kwargs["cwd"], pipeline.layout.root)
        self.assertIn("--trace-expiry=true", command)
        csv_outputs = [arg.split("=", 1)[1] for arg in command if str(arg).startswith("--csvout=")]
        self.assertEqual(len(csv_outputs), 1)
        pipeline._assert_output(Path(csv_outputs[0]))

    def test_plots_only_skips_build_training_and_simulation(self):
        pipeline = self.make_pipeline("--plots-only")
        with patch.object(pipeline, "build", side_effect=AssertionError("plots only built simulator")), \
             patch.object(pipeline, "preprocess", side_effect=AssertionError("plots only read raw traces")), \
             patch.object(pipeline, "train_models", side_effect=AssertionError("plots only trained models")), \
             patch.object(pipeline, "run_experiments", side_effect=AssertionError("plots only ran simulations")), \
             patch.object(pipeline, "plot") as plot:
            pipeline.execute()
        plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
