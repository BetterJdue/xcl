import json
import math
import tempfile
import unittest
from pathlib import Path

from compile_bench_compare import compare, environment_notes, geomean, main, render


def program(name, insts=100, hlsl=1000, dxil=300, dxc=10.0):
    return {"name": name, "instructions": insts, "hlsl_bytes": hlsl, "dxil_bytes": dxil, "dxc_ms_median": dxc}


ENV = {"commit": "a" * 40, "dirty": False, "dxc": {"dxcompiler": {"file_version": "1.9.2602.17"}},
       "profile": "release", "cpu": "x", "logical_cpus": 12, "set": "full", "reps": 3}


class CompareTests(unittest.TestCase):
    def test_ratios_and_geometric_mean(self):
        base = {"a": program("a", dxc=10.0), "b": program("b", hlsl=2000, dxc=20.0)}
        head = {"a": program("a", dxc=5.0), "b": program("b", hlsl=1000, dxc=80.0)}
        r = compare(base, head)
        self.assertEqual(["a", "b"], [row["name"] for row in r["rows"]])
        self.assertEqual((10.0, 5.0, 0.5), r["rows"][0]["dxc"])
        self.assertAlmostEqual(math.sqrt(0.5 * 4.0), r["means"]["dxc"])
        self.assertAlmostEqual(math.sqrt(1.0 * 0.5), r["means"]["hlsl"])
        self.assertEqual(1, r["size_changes"])

    def test_programs_in_one_run_only_are_listed(self):
        r = compare({"a": program("a"), "old": program("old")}, {"a": program("a"), "new": program("new")})
        self.assertEqual((["old"], ["new"]), (r["only_base"], r["only_head"]))
        self.assertEqual(1, len(r["rows"]))

    def test_geomean_skips_missing_ratios(self):
        self.assertAlmostEqual(2.0, geomean([2.0, math.nan, 0.0]))
        self.assertTrue(math.isnan(geomean([])))


class EnvironmentTests(unittest.TestCase):
    def test_same_environment_has_no_notes(self):
        self.assertEqual([], environment_notes(ENV, dict(ENV, commit="b" * 40)))

    def test_differences_that_make_times_incomparable_are_noted(self):
        other = dict(ENV, dirty=True, profile="debug", dxc={"dxcompiler": {"file_version": "1.8.2505.1"}})
        notes = environment_notes(ENV, other)
        self.assertTrue(any("head was measured with uncommitted changes" in n for n in notes))
        self.assertTrue(any("different DXC" in n for n in notes))
        self.assertTrue(any("different profile" in n for n in notes))
        self.assertEqual(3, len(notes))


class RenderTests(unittest.TestCase):
    def test_table_rows_have_equal_columns(self):
        r = compare({"a": program("a"), "b": program("b", insts=7)}, {"a": program("a", dxil=0), "b": program("b", insts=9)})
        lines = [l for l in render(ENV, ENV, r).splitlines() if l.startswith("|")]
        self.assertEqual(4, len(lines))
        self.assertEqual(1, len({l.count("|") for l in lines}))
        self.assertIn("| 7 → 9 |", lines[3])

    def test_main_reads_files_and_rejects_bad_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run.json"
            p.write_text(json.dumps({"environment": ENV, "programs": [program("a")]}), encoding="utf-8")
            self.assertEqual(0, main([str(p), str(p)]))
            (Path(d) / "bad.json").write_text("{", encoding="utf-8")
            self.assertEqual(2, main([str(p), str(Path(d) / "bad.json")]))
            self.assertEqual(2, main([str(p)]))


if __name__ == "__main__":
    unittest.main()
