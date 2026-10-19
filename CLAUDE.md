@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Project overview, scope boundaries, code style, testing and PR rules are in
`AGENTS.md` (imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Julia 1.12.0 (Linux, CPU only),
which is older than the 1.12.6 that the tracked `Manifest.toml` pins (`julia_version = "1.12.6"`); CI runs Julia 1.12 only.

| Command | Status |
|---|---|
| `julia --project=. -e 'using Pkg; Pkg.instantiate()'` | verified (about 12 min cold: Zygote/NNlib tree, then precompile) |
| `julia --project=. -e 'using SynapticDistill; println("OK")'` | verified |
| `julia --project=. -e 'using Pkg; Pkg.test()'` | verified (348 passed, about 70 s) |
| `julia --project=. examples/pure_snn_training.jl`, `julia --project=. examples/hybrid_moe_training.jl` | verified (exit 0) |
| `julia --project=scripts -e 'using Pkg; Pkg.instantiate()'` then `julia --project=scripts scripts/spikenaut_train.jl <state_telemetry.jsonl> <epochs> <out> [split]` | not run (needs a v3 `state_telemetry` JSONL; `Pkg.test()` exercises the script on `test/fixtures/` data) |
| `julia --project=. benchmarks/eprop_tick_benchmark.jl` | not run (it calls `create_network` and `eprop_update!`, which `src/` does not define, so it cannot run as written; it only prints and does not write `benchmarks/latest_results.txt`, which is tracked despite a `.gitignore` entry) |

CI (`.github/workflows/ci.yml`) instantiates the project and runs `julia-runtest` on Julia 1.12.
No formatter or linter config is in the repo, so there is no format gate to run.

## Architecture map

- `src/SynapticDistill.jl`: module and exports (`SpikeBatch`, `TraceBatch`, `TrainingState`, surrogates, `update_eprop!`, `update_ottt!`, `ModelStep`, `train_step!`).
- `src/types.jl`, `src/utils.jl`, `src/training.jl`: data types, shared helpers, and the `train_step!` loop.
- `src/rules/`: `surrogate.jl`, `eprop.jl`, `ottt.jl`. `predictive_eprop.jl` is a placeholder that the module does not `include`.
- `test/runtests.jl`: single test file; `test/fixtures/*.jsonl` are the sidecar's inputs.
- `scripts/spikenaut_train.jl`: standalone trainer sidecar with its own `scripts/Project.toml` (JSON3). It does not `using SynapticDistill`.
- `examples/`, `benchmarks/`: usage examples and the e-prop tick benchmark.
- `.cursor/`, `.devcontainer/`, `.devin/`: agent and container environment setup (Julia 1.12).

## Invariants and non-goals (from README and AGENTS.md)

- No STDP or Hebbian rules (they live in the external `Limen-Neural/plasticity-lab`; `rmems/plasticity-lab` does not exist, so do not invent that URL), no IPC wire types, no domain-specific architectures.
- The library does not own a global `forward`: callers inject the model step and the loss; only SNN parameters are updated.
- Do not test on Julia 1.11: the pinned `PrecompileTools` needs 1.12-only `Base` internals (`AGENTS.md`).
- `README.md` uses markdown only (badges are markdown images); keep it HTML-free.
- SPDX header `# SPDX-License-Identifier: MIT OR Apache-2.0` is on every `.jl` file under `src/`, `test/`, `examples/`, `benchmarks/`. In `scripts/spikenaut_train.jl` it is on line 2, after the shebang. Keep it on new files.
- `CHANGELOG.md` has an `Unreleased` section; `Project.toml` has no `exclude`/`include` list (see Gotchas).

## Gotchas

- `Project.toml` has no publish `exclude`/`files` list, so adding this file needs no manifest change. Never register or publish the package.
- `.claude/` is not gitignored; do not commit local agent state there. `.cursor/*` is ignored except `environment.json` and `install.sh`.
- `Manifest.toml` is tracked on purpose (`.gitignore` has `!Manifest.toml`); do not regenerate it as a side effect.
- Hit a precompile error on the first `instantiate`? Re-run it; the retry succeeded here.
- Edit only what the task needs.
