# SpecLens Procedure

This release packages the two runnable SpecLens pipelines used in our experiments:

- `realbench_safe_pipeline_156`: the Evalhuman / HDLBits-style benchmark pipeline
- `realbench_safe_pipeline_RTLLM`: the RTLLM benchmark pipeline

It also includes the task folders needed to run them directly:

- `data_Evalhuman/`
- `RTLLMv1.1/`
- `RTLLM2.0/`

## What Is Included

```text
SpecLens_procedure/
├── data_Evalhuman/
├── RTLLMv1.1/
├── RTLLM2.0/
├── realbench_safe_pipeline_156/
├── realbench_safe_pipeline_RTLLM/
├── safe_pipeline_156.py
├── safe_pipeline_RTLLM.py
├── requirements.txt
├── .env.example
└── LICENSE
```

## Tested Environment

- Python 3.10+ required, Python 3.11 recommended
- Icarus Verilog 12.0 (stable)
- Verilator 5.046 locally tested

The pipelines may also work with nearby versions, but this release was validated with `iverilog -V = 12.0 (stable)`.

## Python Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## API Key Setup

Create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and set at least:

```bash
OPENAI_API_KEY=...
LLM_PROVIDER=openai
```

If you use the direct comparison branches, you may also set:

```bash
OPENAI_DIRECT_MODEL=o3-mini
OPENAI_DIRECT_REASONING_EFFORT=medium
```

## Simulator Setup

Both pipelines rely on the benchmark-provided golden testbench for functional checking of the final Stage III Verilog candidate.

- `realbench_safe_pipeline_156` uses `iverilog + vvp`
- `realbench_safe_pipeline_RTLLM` uses `iverilog + vvp` by default, and automatically falls back to `verilator` on a small set of known Icarus-incompatible RTLLM tasks

For that reason, we recommend installing both Icarus Verilog and Verilator.

### macOS

#### Verilator

The Homebrew formula currently provides a straightforward install path:

```bash
brew install verilator
```

#### Icarus Verilog

For reproducibility, we recommend Icarus Verilog **12.0 stable**.

If your package manager already provides `iverilog 12.0`, that is the easiest option. Otherwise, build the official stable `v12-branch` from source:

```bash
brew install autoconf automake bison flex gperf readline help2man
git clone https://github.com/steveicarus/iverilog.git
cd iverilog
git checkout --track -b v12-branch origin/v12-branch
sh autoconf.sh
mkdir build
cd build
../configure --prefix=/usr/local
make -j"$(sysctl -n hw.ncpu)"
sudo make install
```

If Homebrew-installed `flex` is not discovered during `configure`, prepend it to `PATH` first:

```bash
export PATH="/opt/homebrew/opt/flex/bin:$PATH"
```

After installation, confirm:

```bash
iverilog -V
verilator --version
```



## Quick Start

The examples below use only the most common flags, but the CLIs support more than:

- `--provider`
- `--variant`
- `--dataset-root`
- `--trace-root`
- `--task`

In practice, the most useful runtime flags are:

- `--list-only`: scan tasks only, do not run the pipeline
- `--max-workers`: number of tasks processed in parallel
- `--task-max-candidate-workers`: candidate-generation parallelism inside one task
- `--task-max-prefilter-workers`: scenario-prefilter parallelism inside one task
- `--verification-timeout-sec`: simulator timeout per compile/run step
- `--strict-scan`: fail immediately if any task folder layout is invalid
- `--env-file`: use a custom `.env` file

The `safe_pipeline_156.py` CLI additionally supports:

- `--model`
- `--reasoning-effort`

One command corresponds to exactly:

- one `dataset-root`
- one `variant`
- one `trace-root`
- one task set, where `--task` may be repeated multiple times

If you want to run multiple variants or multiple datasets, launch multiple commands, each with its own `--trace-root`.

### Evalhuman / 156 pipeline

List tasks:

```bash
python safe_pipeline_156.py --list-only
```

Run the main requirements-to-constraints branch on one task:

```bash
python safe_pipeline_156.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root data_Evalhuman \
  --trace-root pipeline_runs_156_demo \
  --task lemmings4
```

Run a direct baseline:

```bash
python safe_pipeline_156.py \
  --provider openai \
  --variant spec_direct_baseline \
  --dataset-root data_Evalhuman \
  --trace-root pipeline_runs_156_direct_demo \
  --task lemmings4
```

### RTLLM pipeline

List tasks in RTLLMv1.1:

```bash
python safe_pipeline_RTLLM.py --dataset-root RTLLMv1.1 --list-only
```

Run the main branch on RTLLMv1.1:

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLMv1.1 \
  --trace-root pipeline_runs_RTLLM_v11_demo \
  --task fsm
```

Run the same pipeline on RTLLM2.0:

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLM2.0 \
  --trace-root pipeline_runs_RTLLM_v20_demo \
  --task fsm
```

## More Complete Examples

### One variant with multiple tasks

Run the same RTLLM branch on several tasks from the same dataset:

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLMv1.1 \
  --trace-root pipeline_runs_RTLLM_v11_batch1 \
  --max-workers 3 \
  --task fsm \
  --task signal_generator
```

### Two variants on the same dataset

To compare two branches on the same task set, run two separate commands with different `--variant` and different `--trace-root`:

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLMv1.1 \
  --trace-root pipeline_runs_RTLLM_v11_req_constraint \
  --max-workers 3 \
  --task fsm \
  --task signal_generator
```

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant spec_direct_baseline \
  --dataset-root RTLLMv1.1 \
  --trace-root pipeline_runs_RTLLM_v11_direct \
  --max-workers 3 \
  --task fsm \
  --task signal_generator
```

### Multiple datasets

One command cannot take multiple `--dataset-root` values at the same time. If you want to run both `RTLLMv1.1` and `RTLLM2.0`, launch them separately:

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLMv1.1 \
  --trace-root pipeline_runs_RTLLM_v11_req \
  --max-workers 3 \
  --task fsm \
  --task signal_generator
```

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLM2.0 \
  --trace-root pipeline_runs_RTLLM_v20_req \
  --max-workers 3 \
  --task fsm \
  --task alu
```

### Complex comparison example

If you want to compare two variants across two RTLLM datasets, you typically run four commands:

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLMv1.1 \
  --trace-root pipeline_runs_RTLLM_v11_req \
  --max-workers 3 \
  --task fsm \
  --task signal_generator
```

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant spec_direct_baseline \
  --dataset-root RTLLMv1.1 \
  --trace-root pipeline_runs_RTLLM_v11_direct \
  --max-workers 3 \
  --task fsm \
  --task signal_generator
```

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root RTLLM2.0 \
  --trace-root pipeline_runs_RTLLM_v20_req \
  --max-workers 3 \
  --task fsm \
  --task alu
```

```bash
python safe_pipeline_RTLLM.py \
  --provider openai \
  --variant spec_direct_baseline \
  --dataset-root RTLLM2.0 \
  --trace-root pipeline_runs_RTLLM_v20_direct \
  --max-workers 3 \
  --task fsm \
  --task alu
```

### Evalhuman example with model override

The `_156` CLI also supports OpenAI model overrides:

```bash
python safe_pipeline_156.py \
  --provider openai \
  --variant requirements_constraint_selfplanning \
  --dataset-root data_Evalhuman \
  --trace-root pipeline_runs_156_o3mini_high_demo \
  --max-workers 3 \
  --model o3-mini \
  --reasoning-effort high \
  --task countbcd \
  --task lemmings4 \
  --task gshare
```

## Supported Pipeline Variants

Both CLIs expose the same branch names:

- `requirements_constraint_selfplanning`
- `spec_direct_baseline`
- `selfplanning_only`
- `direct_few_shots`
- `direct_with_constraint`


## Outputs

Each run writes a trace directory such as:

- `pipeline_runs_156_*`
- `pipeline_runs_RTLLM_*`

The generated run folder contains:

- `runtime_status.json`
- `task_index.json`
- one per-task trace tree
- summary files such as `summary.json`

## License

This release currently ships with the MIT license in `LICENSE`.
