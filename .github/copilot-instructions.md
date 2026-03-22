# MADDPG Project Guidelines

## Project Status
- This repository is archived and maintained as-is.
- Prefer minimal, scoped fixes over broad refactors unless explicitly requested.

## Architecture
- Training entrypoint: `experiments/train.py`.
- Core algorithm: `maddpg/trainer/maddpg.py` (`MADDPGAgentTrainer`, centralized critic + per-agent actors).
- Experience replay: `maddpg/trainer/replay_buffer.py`.
- TensorFlow helpers and distribution utilities: `maddpg/common/tf_util.py`, `maddpg/common/distributions.py`.
- Agent interface contract lives in `maddpg/__init__.py` (`AgentTrainer`).

## Build And Run

### Recommended: Legacy Conda Environment
This repo requires Python 3.5 and TensorFlow 1.8.0. Set up a legacy environment:

```bash
# Create legacy Python 3.5 environment with exact dependencies
conda create -n maddpg_py35 python=3.5.6 tensorflow=1.8.0 gym=0.10.5 numpy=1.14.5
conda activate maddpg_py35
pip install -e .
pip install git+https://github.com/openai/multiagent-particle-envs.git
```

### Training
From repo root:
```bash
conda activate maddpg_py35
cd experiments
SUPPRESS_MA_PROMPT=1 python train.py --scenario simple --exp-name myexp --num-episodes 1000
```

Or use the smoke-test script for quick validation:
```bash
conda activate maddpg_py35
./run_smoke.sh
```

### Key Points
- `--exp-name` is **required** (experiment name for output files).
- `SUPPRESS_MA_PROMPT=1` suppresses MPE's interactive deprecation warning (only needed for non-interactive runs).
- Common training parameters are defined in `experiments/train.py` (`parse_args`).
- Checkpoints saved to `--save-dir` (default: `/tmp/policy/`).
- Reward curves saved to `--plots-dir` (default: `./learning_curves/`).

## Environment Constraints
- Code is TensorFlow 1.x style and uses `tensorflow.contrib`.
- README references a legacy dependency set (Python 3.5.4, TensorFlow 1.8.0, Gym 0.10.5).
- Multi-Agent Particle Environments (MPE) must be installed separately and available on `PYTHONPATH`.

## Coding Conventions
- Keep compatibility with existing TensorFlow 1.x patterns (`tf.variable_scope`, `U.function`, session-based flow).
- Preserve the current trainer lifecycle pattern:
  - `action()` for policy inference
  - `experience()` for replay insertion
  - `preupdate()` before coordinated update
  - `update(agents, t)` for synchronized learning
- Favor small, local changes in existing files instead of introducing new abstractions.

## Validation Expectations
- There is no built-in test suite in this repository.
- When changing training logic, validate by:
  - running a short training invocation in `experiments/train.py`, or
  - checking import/syntax/runtime errors in affected modules.
- If full runtime validation is not possible (for example missing MPE), state this clearly in the final summary.