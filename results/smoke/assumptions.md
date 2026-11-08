# Smoke run assumptions

- Unused fees include priority and compute.
- Utilization counts declared objects; prices use the exponential update.
- Arrivals are pre-generated at the configured reference price and share identical transactions and execution outcomes across mechanisms.
- The workload uses the calibrated mixture priority distribution; seed and other parameters are recorded in `config.yaml`.
