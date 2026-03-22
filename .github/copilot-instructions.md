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
- Install from repo root: `pip install -e .`
- Train from `experiments/`: `python train.py --scenario simple`
- Common defaults are defined in `experiments/train.py` (`parse_args`).

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