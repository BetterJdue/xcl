//! The measurement: each generated program is translated to HLSL and compiled with DXC, one
//! compile at a time, and its sizes and times are written as JSON and CSV. The same program
//! also goes through the two direct DXIL paths (#1426's translator, #1745's per-lane form),
//! timed with the same repeats; a refusal is recorded, not a failure.
//!
//! - `XCL_COMPILE_BENCH=full`: the full set (sizes up to 20 000 operations); `large`: only the
//!   multi-megabyte programs (up to 40 000 operations); unset: a small set that runs in seconds,
//!   so CI and every test run go through the same code
//! - `XCL_COMPILE_BENCH_OUT=<folder>`: write `compile_bench.json` and `compile_bench.csv` there
//! - `XCL_COMPILE_BENCH_REPS=<n>`: compiles per program (default 1, full set 3)
//! - `XCL_COMPILE_BENCH_DXC_ARGS="-O1 …"`: extra DXC arguments; unset, `dxc.rs` picks the
//!   optimisation level from the source size, as it does for a title
//!
//! Times are wall-clock and only comparable on one machine; run the full set with nothing else
//! running (`--test-threads=1`).

use std::time::Instant;

use serde::Serialize;
use xcl::gpu::dxil::kernel::KernelOptions;
use xcl::gpu::dxil::{coverage, validate};
use xcl::gpu::recomp::{dxc, lower_dxil, translate, Mode, Options};

use crate::gen::{program, Knobs};

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub name: String,
    #[serde(flatten)]
    pub knobs: Knobs,
    pub split_wave64: bool,
    pub mem_windows: u32,
    pub structured: bool,
    pub instructions: usize,
    pub blocks: usize,
    pub code_bytes: usize,
    pub hlsl_bytes: usize,
    pub translate_ms: f64,
    pub dxc_args: String,
    pub dxc_ms: Vec<f64>,
    pub dxc_ms_median: f64,
    pub dxc_ms_min: f64,
    pub dxil_bytes: usize,
    /// #1426's direct translator (`XCL_DIRECT_DXIL=1`): the same program straight to DXIL
    pub direct_ms_median: f64,
    pub direct_dxil_bytes: usize,
    pub direct_refused: String,
    /// #1745's per-lane form (`XCL_PER_LANE=1`); compute programs only, as in `pso.rs`
    pub per_lane_ms_median: f64,
    pub per_lane_dxil_bytes: usize,
    pub per_lane_refused: String,
}

/// A program of the set: its name, knobs and the recompiler options it is translated with.
pub struct Case {
    pub name: String,
    pub knobs: Knobs,
    pub options: Options,
}

fn case(name: &str, knobs: Knobs, split: bool, windows: u32, structured: bool) -> Case {
    let options = Options { split_wave64: split, mem_windows: windows, structured, ..knobs.options() };
    Case { name: name.to_string(), knobs, options }
}

/// The small set: one program per kind of control flow, wave size and form, a few hundred
/// operations each.
pub fn small() -> Vec<Case> {
    let k = |ops| Knobs { ops, ..Knobs::default() };
    vec![
        case("w64", k(300), false, 1, true),
        case("w32", Knobs { wave64: false, ..k(300) }, false, 1, true),
        case("w64-split-4win", k(300), true, 4, true),
        case("loops", Knobs { loops: 2, loop_depth: 2, ..k(300) }, false, 1, true),
        case("divergent", Knobs { if_depth: 4, divergent_pct: 40, ..k(300) }, false, 1, true),
        case("cs-resources", Knobs { compute: true, buffer_pct: 10, image_pct: 5, ..k(300) }, false, 1, true),
        // what the per-lane form takes: no divergent ifs, no global or scalar memory
        case("cs-uniform", Knobs { compute: true, if_depth: 0, mem_pct: 0, buffer_pct: 10, image_pct: 5, loops: 1, ..k(300) }, false, 1, true),
    ]
}

/// The full set: five shapes at four sizes, each in the form a 64-lane GPU gets and the one a
/// 32-lane GPU gets (split wave64, 4 memory windows), plus controls whose effect on the HLSL is
/// known (the guarded block chain, 1 against 4 memory windows).
pub fn full() -> Vec<Case> {
    let mut v = Vec::new();
    for ops in [500, 2000, 8000, 20000] {
        let shapes = [
            ("base", Knobs { ops, ..Knobs::default() }),
            ("divergent", Knobs { ops, if_depth: 6, divergent_pct: 40, ..Knobs::default() }),
            ("loops", Knobs { ops, loops: 4, loop_depth: 3, ..Knobs::default() }),
            ("resources", Knobs { ops, compute: true, mem_pct: 10, buffer_pct: 10, image_pct: 5, ..Knobs::default() }),
            ("cs-uniform", Knobs { ops, compute: true, if_depth: 0, mem_pct: 0, buffer_pct: 10, image_pct: 5, loops: 1, ..Knobs::default() }),
            ("w32", Knobs { ops, wave64: false, ..Knobs::default() }),
        ];
        for (shape, knobs) in shapes {
            v.push(case(&format!("{shape}-{ops}"), knobs.clone(), false, 1, true));
            if knobs.wave64 {
                v.push(case(&format!("{shape}-{ops}-split-4win"), knobs, true, 4, true));
            }
        }
        let base = Knobs { ops, ..Knobs::default() };
        v.push(case(&format!("control-guarded-chain-{ops}"), base.clone(), false, 1, false));
        v.push(case(&format!("control-4win-{ops}"), base, false, 4, true));
    }
    v
}

/// The large set: only the multi-megabyte tail, where most of the compile time goes. Four
/// shapes at 8 000, 20 000 and 40 000 operations, in both forms; the larger ones cross the
/// source sizes at which `dxc.rs` lowers the optimisation level.
pub fn large() -> Vec<Case> {
    let mut v = Vec::new();
    for ops in [8000, 20000, 40000] {
        let shapes = [
            ("base", Knobs { ops, ..Knobs::default() }),
            ("divergent", Knobs { ops, if_depth: 6, divergent_pct: 40, ..Knobs::default() }),
            ("resources", Knobs { ops, compute: true, mem_pct: 10, buffer_pct: 10, image_pct: 5, ..Knobs::default() }),
            ("cs-uniform", Knobs { ops, compute: true, if_depth: 0, mem_pct: 0, buffer_pct: 10, image_pct: 5, loops: 1, ..Knobs::default() }),
        ];
        for (shape, knobs) in shapes {
            v.push(case(&format!("{shape}-{ops}"), knobs.clone(), false, 1, true));
            v.push(case(&format!("{shape}-{ops}-split-4win"), knobs, true, 4, true));
        }
    }
    v
}

/// The set's DXC times (each program's median), so the slow tail shows next to the typical
/// program: the nearest-rank p90, the maximum, and the share of the total compile time spent
/// on the slowest tenth of the programs (at least one).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub programs: usize,
    pub dxc_ms_total: f64,
    pub dxc_ms_median: f64,
    pub dxc_ms_p90: f64,
    pub dxc_ms_max: f64,
    pub slowest_tenth_share: f64,
    /// programs each direct path translated (and their total time), and the compute programs
    /// the per-lane form refused
    pub direct_translated: usize,
    pub direct_ms_total: f64,
    pub per_lane_translated: usize,
    pub per_lane_ms_total: f64,
    pub per_lane_refused: usize,
}

pub fn summary(rows: &[Row]) -> Summary {
    let mut t: Vec<f64> = rows.iter().map(|r| r.dxc_ms_median).collect();
    t.sort_by(f64::total_cmp);
    let n = t.len();
    let total: f64 = t.iter().sum();
    let tenth: f64 = t.iter().rev().take(n.div_ceil(10)).sum();
    Summary {
        programs: n,
        dxc_ms_total: total,
        dxc_ms_median: if n > 0 { median(&t) } else { 0.0 },
        dxc_ms_p90: if n > 0 { t[(n * 9).div_ceil(10) - 1] } else { 0.0 },
        dxc_ms_max: t.last().copied().unwrap_or(0.0),
        slowest_tenth_share: if total > 0.0 { tenth / total } else { 0.0 },
        direct_translated: rows.iter().filter(|r| r.direct_refused.is_empty()).count(),
        direct_ms_total: rows.iter().map(|r| r.direct_ms_median).sum(),
        per_lane_translated: rows.iter().filter(|r| r.per_lane_refused.is_empty()).count(),
        per_lane_ms_total: rows.iter().map(|r| r.per_lane_ms_median).sum(),
        per_lane_refused: rows.iter().filter(|r| !r.per_lane_refused.is_empty() && r.per_lane_refused != NOT_COMPUTE).count(),
    }
}

/// The rows from the slowest DXC compile to the fastest (ties by name).
pub fn by_time(rows: &[Row]) -> Vec<&Row> {
    let mut v: Vec<&Row> = rows.iter().collect();
    v.sort_by(|a, b| b.dxc_ms_median.total_cmp(&a.dxc_ms_median).then_with(|| a.name.cmp(&b.name)));
    v
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.len() % 2 == 1 { s[s.len() / 2] } else { (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0 }
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1e6).round() / 1e3
}

/// Translate and compile one program `reps` times. Panics when it does not translate completely,
/// DXC refuses it, or a repeat gives different HLSL or DXIL.
pub fn measure(c: &Case, reps: usize, dxc_args: &[&str]) -> Row {
    let code = program(&c.knobs);
    let t0 = Instant::now();
    let t = translate(&code, &c.options).unwrap_or_else(|e| panic!("{}: {e}", c.name));
    let translate_ms = ms(t0);
    assert!(t.complete(), "{}: {:?}", c.name, t.unsupported_why);
    assert_eq!(translate(&code, &c.options).unwrap().hlsl, t.hlsl, "{}: the HLSL differs between two translations", c.name);
    let (mut times, mut dxil_bytes) = (Vec::new(), None);
    for _ in 0..reps {
        let t0 = Instant::now();
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", dxc_args).unwrap_or_else(|e| panic!("{}: {}", c.name, e.lines().next().unwrap_or("")));
        times.push(ms(t0));
        assert!(!dxil.is_empty(), "{}: empty DXIL", c.name);
        assert_eq!(*dxil_bytes.get_or_insert(dxil.len()), dxil.len(), "{}: the DXIL size differs between compiles", c.name);
    }
    let o = KernelOptions {
        wave64: c.options.wave64,
        split: c.options.wave64 && c.options.split_wave64,
        mem_windows: c.options.mem_windows,
        sampler_base: c.options.static_sampler_base,
    };
    let direct = direct_path(c, reps, || coverage::translate_mode(&code, &c.options.mode, o, false));
    let per_lane = match &c.options.mode {
        Mode::Compute(cs) => direct_path(c, reps, || lower_dxil::compute_with(&code, cs, o).map(|k| k.dxil)),
        _ => (0.0, 0, NOT_COMPUTE.to_string()),
    };
    Row {
        name: c.name.clone(),
        knobs: c.knobs.clone(),
        split_wave64: c.options.split_wave64,
        mem_windows: c.options.mem_windows,
        structured: c.options.structured,
        instructions: t.instructions,
        blocks: t.blocks,
        code_bytes: code.len(),
        hlsl_bytes: t.hlsl.len(),
        translate_ms,
        dxc_args: dxc_args.join(" "),
        dxc_ms_median: median(&times),
        dxc_ms_min: times.iter().copied().fold(f64::INFINITY, f64::min),
        dxc_ms: times,
        dxil_bytes: dxil_bytes.unwrap_or(0),
        direct_ms_median: direct.0,
        direct_dxil_bytes: direct.1,
        direct_refused: direct.2,
        per_lane_ms_median: per_lane.0,
        per_lane_dxil_bytes: per_lane.1,
        per_lane_refused: per_lane.2,
    }
}

/// The per-lane form takes compute programs only (`pso.rs`): the others are not refusals.
pub const NOT_COMPUTE: &str = "not a compute program";

/// One direct path `reps` times: (median ms, DXIL bytes, "") when it translates, or
/// (0, 0, the first line of its refusal). A container the DXC
/// validator refuses counts as a refusal (`validator: …`); the validation is not timed.
fn direct_path(c: &Case, reps: usize, run: impl Fn() -> Result<Vec<u8>, String>) -> (f64, usize, String) {
    let refused = |e: &str| (0.0, 0, e.lines().next().unwrap_or("").to_string());
    let mut times = Vec::new();
    let mut dxil = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        match run() {
            Ok(d) => dxil = d,
            Err(e) => return refused(&e),
        }
        times.push(ms(t0));
    }
    assert_eq!(run().as_deref(), Ok(dxil.as_slice()), "{}: the direct DXIL differs between two translations", c.name);
    if let Err(e) = validate::validate(&dxil) {
        return refused(&format!("validator: {e}"));
    }
    (median(&times), dxil.len(), String::new())
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// What the numbers depend on: the code's commit, the DXC, the build and the machine.
pub fn environment(set: &str, reps: usize) -> serde_json::Value {
    let var = |n: &str| std::env::var(n).unwrap_or_default();
    serde_json::json!({
        "set": set,
        "reps": reps,
        "commit": git(&["rev-parse", "HEAD"]),
        "dirty": !git(&["status", "--porcelain"]).is_empty(),
        "dxc": dxc::toolchain(),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "cpu": var("PROCESSOR_IDENTIFIER"),
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    })
}

/// `compile_bench.json` ({environment, summary, programs}) and `compile_bench.csv` (one row per
/// program, columns in name order; the compile times joined with `;`, commas in text made `;`).
pub fn write(dir: &std::path::Path, env: &serde_json::Value, rows: &[Row]) {
    std::fs::create_dir_all(dir).unwrap();
    let programs: Vec<serde_json::Value> = rows.iter().map(|r| serde_json::to_value(r).unwrap()).collect();
    let json = serde_json::json!({ "environment": env, "summary": summary(rows), "programs": programs });
    std::fs::write(dir.join("compile_bench.json"), serde_json::to_string_pretty(&json).unwrap() + "\n").unwrap();
    let cell = |v: &serde_json::Value| match v {
        serde_json::Value::Array(a) => a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(";"),
        serde_json::Value::String(s) => s.replace(',', ";"),
        x => x.to_string(),
    };
    let mut csv = String::new();
    if let Some(serde_json::Value::Object(first)) = programs.first() {
        csv += &first.keys().cloned().collect::<Vec<_>>().join(",");
        csv += "\n";
        for p in &programs {
            csv += &p.as_object().unwrap().values().map(cell).collect::<Vec<_>>().join(",");
            csv += "\n";
        }
    }
    std::fs::write(dir.join("compile_bench.csv"), csv).unwrap();
}
