# EV Battery Electro-Thermal Digital Twin

A 12-cell lumped thermal-network simulation with SOC tracking and comparisons of passive, air and liquid cooling.

Run:
```bash
pip install -r requirements.txt
python src/run_simulation.py
```
Outputs include simulated operating conditions, cooling comparison and ML-surrogate error metrics.

**Limitations:** Illustrative parameters and simplified heat-transfer equations; no validated electrochemistry or thermal-runaway propagation. Calibrate against cell test data and perform independent engineering/safety validation before using design conclusions.
