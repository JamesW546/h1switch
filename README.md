# h1switch

A small hot-switching framework for two H1 position-PD policies.

Both policies run on every control tick. During a switch, motor targets, `Kp`,
and `Kd` are blended with a smoothstep curve, so one controller always owns the
robot command stream.

## Quick test

```bash
pip install numpy pyyaml
python policy_switcher.py config.example.yaml
```

Press `s` to switch between the two example policies.

## Use your own policies

1. Copy `HoldPositionAdapter` in `example_plugins.py`.
2. Build the observation expected by your model in `step()`.
3. Convert the model action to full motor-order position targets and PD gains.
4. Add your adapter and model settings to a YAML file.
5. Replace `MockBackend` with a backend that reads `LowState` and writes one
   `LowCmd` per tick.

The two policies must control the same robot and use a compatible position-PD
interface. Model files are not included.
