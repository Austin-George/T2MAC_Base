# T2MAC Codebase Changes

## Dependencies Installed

- PyTorch
- SMAC
- PySC2
- Sacred
- TensorBoardX
- NumPy
- PyYAML

The StarCraft II game files and SMAC maps were also installed.

## Compatibility Fix

### PyYAML

- The repository used `yaml.load()`, but modern PyYAML versions require an explicit loader.
- Replaced `yaml.load()` with `yaml.safe_load()` throughout the codebase.

**Files modified:**
- `main.py`

## Deprecation Fix

### Collections

- Older code imported classes from `collections`, but modern Python versions moved these classes to `collections.abc`.
- Updated imports to `collections.abc`.

**Files modified:**
- `main.py`

## CUDA Assumption Removal

- Several sections assumed the availability of CUDA. Running this in a CPU-only environment caused failures.
- Replaced direct CUDA calls with device-aware operations.

**Files modified:**

- `controllers/basic_controller.py`
- `controllers/tmac_comm_rate_controller.py`
- `controllers/tmac_full_comm_controller.py`
- `controllers/tmac_p2p_comm_controller.py`
- `controllers/tmac_vffac_controller.py`
- `controllers/vffac_controller.py`
- `learners/coma_learner.py`
- `learners/q_learner.py`
- `learners/qtran_learner.py`
- `learners/tmac_comm_rate_learner.py`
- `learners/tmac_full_comm_learner.py`
- `learners/tmac_p2p_comm_learner.py`
- `learners/tmac_vffac_learner.py`
- `learners/vffac_learner.py`
- `modules/agents/tmac_p2p_comm_rnn_msg_agent.py`
- `run.py`

## StarCraft II Environment

- SMAC maps were installed.
- PySC2 environment was initialized.
- Benchmark map used: `3m`.

**Files modified:**

- `config/default.yaml`
- `config/envs/sc2.yaml`

**Environment configurations modified:**

- `save_replay`
- `test_interval`
- `log_interval`
- `runner_log_interval`
- `learner_log_interval`
- `t_max`

## End-to-End Training

Command used:

```bash
python src/main.py --config=tmac_p2p_comm --env-config=sc2 with env_args.map_name=3m
```

A complete training run was executed on the `3m` map with 100k steps.

**Result:** Achieved a test win rate of approximately **63–73%**.
