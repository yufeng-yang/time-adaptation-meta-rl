# Partial Go2 training snapshot

This folder contains the code, checkpoints and logs available at the snapshot time.
Read `SNAPSHOT_STATUS.json` for the actual progress. The three new baselines are
still training; the paired 100-trial comparison is not available yet. No missing
result should be interpreted as zero performance. The unchanged archived full
model and its original results are under `../deadline_go2_gates/runs/half_bc_steps100k_seed0`.

The existing protocol and training details are preserved in `PROTOCOL.md`.
Run `python restore_large_files.py` from the parent bundle folder before using
archived demonstration trajectories. This partial upload does not affect the
ongoing local training process. Locomotion requires the documented external
Genesis robot assets and local path adjustments on another machine.
