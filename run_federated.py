"""Run a simulated three-hospital federated-learning experiment."""
from __future__ import annotations
import argparse
import csv
import json
import os
import numpy as np

from hospital_client import HospitalClient, make_synthetic_hospital_data
from model_utils import fedavg, predict_proba, binary_cross_entropy, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dp-noise", type=float, default=0.0,
                        help="Illustrative Gaussian noise standard deviation (default: 0)")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rounds < 1 or args.local_epochs < 1:
        raise SystemExit("--rounds and --local-epochs must be >= 1")
    if args.dp_noise < 0:
        raise SystemExit("--dp-noise must be >= 0")

    rng = np.random.default_rng(args.seed)
    clients = []
    configurations = [
        ("Hospital A", 240, 0.0),
        ("Hospital B", 190, 0.25),
        ("Hospital C", 210, -0.2),
    ]
    for index, (name, n, shift) in enumerate(configurations):
        X, y = make_synthetic_hospital_data(n, args.seed + 100 + index, shift)
        clients.append(HospitalClient(name, X, y, args.seed + 200 + index))

    # Separate synthetic evaluation data; no real patient records are used.
    X_test, y_test = make_synthetic_hospital_data(300, args.seed + 999, 0.0)
    weights = np.zeros(X_test.shape[1], dtype=float)
    bias = 0.0
    history = []

    print("Educational federated-learning simulation — synthetic data only")
    print("Raw local rows are not passed to the coordinator.")
    print("-" * 78)
    print(f"{'Round':>5}  {'Mean local loss':>16}  {'Synthetic test loss':>19}  {'Accuracy':>10}")

    for round_number in range(1, args.rounds + 1):
        updates = []
        local_losses = []
        for client in clients:
            local_w, local_b, n, loss = client.train(
                weights, bias,
                local_epochs=args.local_epochs,
                dp_noise=args.dp_noise,
            )
            updates.append((local_w, local_b, n))
            local_losses.append(loss)

        weights, bias = fedavg(updates)
        test_p = predict_proba(X_test, weights, bias)
        test_loss = binary_cross_entropy(y_test, test_p)
        test_accuracy = accuracy(y_test, test_p)
        row = {
            "round": round_number,
            "mean_local_loss": float(np.mean(local_losses)),
            "synthetic_test_loss": test_loss,
            "synthetic_test_accuracy": test_accuracy,
        }
        history.append(row)
        print(f"{round_number:5d}  {row['mean_local_loss']:16.4f}  "
              f"{test_loss:19.4f}  {test_accuracy:9.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    np.savez(os.path.join(args.output_dir, "global_model.npz"), weights=weights, bias=bias)
    summary = {
        "case": "fictional three-hospital educational simulation",
        "data": "synthetic only; no real patient records",
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "seed": args.seed,
        "illustrative_dp_noise_std": args.dp_noise,
        "final_metrics": history[-1],
        "limitations": [
            "Synthetic labels have no medical meaning.",
            "No clinical claims or deployment suitability can be inferred.",
            "Noise option is illustrative and is not a formal differential-privacy guarantee.",
            "Secure aggregation, authentication, encryption, and clinical validation are not implemented."
        ]
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 78)
    print(f"Saved metrics and model to: {args.output_dir}/")
    print("Reminder: this is a teaching demo, not a clinical prediction system.")


if __name__ == "__main__":
    main()
