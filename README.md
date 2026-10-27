# Privacy-Preserving Federated Learning for Collaborative Healthcare AI
## Educational prototype — synthetic data only

This project is a runnable demonstration of federated learning across three fictional hospitals (Hospital A, B, and C). Each hospital creates its own **synthetic** patient-like tabular data and trains a local logistic-regression model. Only model parameters are sent to the coordinator, which performs Federated Averaging (FedAvg).

**Important limitations**
- This is an academic simulation, not a clinical tool and not validated for medical use.
- No real patient data is included or required.
- The example labels are randomly generated synthetic labels and have no medical meaning.
- This prototype demonstrates the workflow, not production-grade security.
- Differential privacy noise is optional and illustrative; it is not a formal privacy guarantee without a complete privacy accountant and calibrated parameters.
- Secure aggregation, authentication, encrypted transport, governance, and clinical validation are not implemented.

## Architecture

1. Coordinator initializes a shared logistic-regression model.
2. Each hospital trains on its own local synthetic dataset.
3. Each hospital returns model parameters (not its raw rows).
4. Coordinator averages the parameters, weighted by local dataset size.
5. The global model is evaluated on a separate synthetic test set.
6. Results are saved to `outputs/`.

## Requirements

- Python 3.10 or newer recommended

## Setup

### Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_federated.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_federated.py
```

## Optional flags
```bash
python run_federated.py --rounds 10 --local-epochs 3
python run_federated.py --rounds 5 --dp-noise 0.01
python run_federated.py --seed 7 --output-dir outputs
```

`--dp-noise` adds simple Gaussian noise to the local model updates after clipping. It is only a teaching illustration, not a certified differential-privacy implementation.

## Files
- `run_federated.py` — coordinator and training loop
- `hospital_client.py` — local hospital data generation and training
- `model_utils.py` — sigmoid, prediction, loss, FedAvg, clipping/noise helpers
- `requirements.txt` — Python dependencies
- `outputs/` — generated metrics and model parameters after running
- `tests/test_model_utils.py` — basic unit tests

## Expected output
The script prints a row per communication round with average loss and synthetic test accuracy. Exact values vary with seed and settings. Do not interpret these metrics as real-world clinical performance.

## Extending this prototype
For a stronger research prototype, consider using Flower for real client-server orchestration, Opacus or another reviewed library for differential privacy, TLS and authenticated clients, and a formally reviewed secure-aggregation protocol. Any real healthcare deployment would require institutional, legal, security, and clinical review.
