"""
benchmark.py - DeepRubin-Explorer Performance Benchmark
========================================================
Measures the latency and throughput of the full pipeline to evaluate
its readiness for the LSST alert stream (~10M alerts/night).

Pipeline stages measured:
  1. GP Preprocessing:    Irregular time series -> uniform 100-point grid
  2. Tensor Conversion:   NumPy array -> PyTorch tensor
  3. Model Inference:     model.forward() per sample

Usage:
    python src/benchmark.py          (quick: 50 samples)
    python src/benchmark.py --n 200  (slower, more stable)
"""

import time
import warnings
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning

# Suppress GP convergence warnings (expected with synthetic short-window data)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from model import LightCurveTCN


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
LSST_ALERTS_PER_NIGHT = 10_000_000
N_CLASSES = 4
N_CHANNELS = 2
N_TIMES = 100


# ─────────────────────────────────────────────────────────────────
# GP Interpolation (mirrors preprocess.py exactly)
# ─────────────────────────────────────────────────────────────────
def gp_interpolate_single(n_obs: int = 30) -> np.ndarray:
    """
    Simulates the full GP preprocessing step for one light curve.
    Generates synthetic g-band and r-band observations and returns
    the interpolated 100-point grid (shape: 2 x 100).

    Args:
        n_obs: Number of raw observations to simulate (typical: 20-50).

    Returns:
        np.ndarray of shape (2, 100) — same format as the training tensors.
    """
    # Simulate irregular ZTF-like timestamps and magnitudes
    rng = np.random.default_rng()
    result_channels = []

    for _ in range(N_CHANNELS):
        # Irregular cadence: random gaps, like real survey data
        times = np.sort(rng.uniform(0, 100, n_obs)).reshape(-1, 1)
        mags = 19.5 + rng.normal(0, 0.3, n_obs)
        errs = rng.uniform(0.01, 0.05, n_obs)

        # Same kernel as preprocess.py
        kernel = C(1.0) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=errs ** 2,
            normalize_y=True,
            n_restarts_optimizer=3,
        )
        gp.fit(times, mags)

        x_grid = np.linspace(times.min(), times.max(), N_TIMES).reshape(-1, 1)
        y_pred, _ = gp.predict(x_grid, return_std=True)
        result_channels.append(y_pred)

    return np.array(result_channels, dtype=np.float32)  # (2, 100)


# ─────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────
def run_benchmark(model_path: Path, n_samples: int = 50, n_warmup: int = 5):
    """
    Runs the full pipeline benchmark in three stages.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  DeepRubin-Explorer | Pipeline Benchmark")
    print(f"  Device: {device.type.upper()}")
    print(f"  Samples: {n_samples}  |  Warm-up: {n_warmup}")
    print(f"{'='*60}\n")

    # ── Load model ──────────────────────────────────────────────
    print("[ 1/4 ] Loading model...")
    model = LightCurveTCN(n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"        Loaded weights from: {model_path.name}\n")

    # ── Generate synthetic light curves ─────────────────────────
    print("[ 2/4 ] Pre-generating synthetic light curves...")
    raw_curves = []
    for i in range(n_samples + n_warmup):
        raw_curves.append(gp_interpolate_single())
    print(f"        Generated {len(raw_curves)} curves.\n")

    # ── WARM-UP phase (not measured) ─────────────────────────────
    print(f"[ 3/4 ] Warming up the model ({n_warmup} runs, not measured)...")
    with torch.no_grad():
        for i in range(n_warmup):
            x = torch.from_numpy(raw_curves[i]).unsqueeze(0).to(device)
            _ = model(x)
    print(f"        Warm-up complete.\n")

    # ── MEASUREMENT phase ─────────────────────────────────────────
    print(f"[ 4/4 ] Measuring latency over {n_samples} samples...")
    gp_times_ms = []
    conv_times_ms = []
    infer_times_ms = []

    use_cuda = (device.type == "cuda")

    with torch.no_grad():
        for i in range(n_warmup, n_warmup + n_samples):

            # ─ Stage 1: GP Preprocessing ─
            t0 = time.perf_counter()
            curve = gp_interpolate_single()
            t1 = time.perf_counter()
            gp_times_ms.append((t1 - t0) * 1000)

            # ─ Stage 2: Tensor Conversion ─
            t2 = time.perf_counter()
            x = torch.from_numpy(curve).unsqueeze(0).to(device)
            if use_cuda:
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            conv_times_ms.append((t3 - t2) * 1000)

            # ─ Stage 3: Model Inference ─
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(x)
                end_event.record()
                torch.cuda.synchronize()
                infer_times_ms.append(start_event.elapsed_time(end_event))
            else:
                t4 = time.perf_counter()
                _ = model(x)
                t5 = time.perf_counter()
                infer_times_ms.append((t5 - t4) * 1000)

    # ── Results ──────────────────────────────────────────────────
    gp_mean = np.mean(gp_times_ms)
    gp_p95 = np.percentile(gp_times_ms, 95)
    conv_mean = np.mean(conv_times_ms)
    infer_mean = np.mean(infer_times_ms)
    infer_p95 = np.percentile(infer_times_ms, 95)
    total_mean = gp_mean + conv_mean + infer_mean
    throughput_per_sec = 1000.0 / total_mean  # alerts/second

    # Time to process 10M alerts on a single node
    hours_for_10M = LSST_ALERTS_PER_NIGHT / throughput_per_sec / 3600

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  {'Stage':<30} {'Mean (ms)':>10} {'P95 (ms)':>10}")
    print(f"  {'-'*50}")
    print(f"  {'GP Preprocessing':<30} {gp_mean:>10.3f} {gp_p95:>10.3f}")
    print(f"  {'Tensor Conversion':<30} {conv_mean:>10.3f} {'N/A':>10}")
    print(f"  {'TCN Inference':<30} {infer_mean:>10.3f} {infer_p95:>10.3f}")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL (end-to-end)':<30} {total_mean:>10.3f}")
    print(f"\n{'='*60}")
    print(f"  THROUGHPUT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Throughput:         {throughput_per_sec:.1f} alerts/second")
    print(f"  Throughput:         {throughput_per_sec * 3600:,.0f} alerts/hour")
    print(f"\n  Est. time for 10M alerts (1 node):  {hours_for_10M:.1f} hours")
    if hours_for_10M > 8:
        n_nodes = int(np.ceil(hours_for_10M / 8))
        print(f"  Nodes needed for real-time (<8h):   {n_nodes} nodes")
    else:
        print(f"  Status: ✓ Can process 10M alerts in a single night on 1 node!")
    print(f"{'='*60}\n")

    return {
        "gp_mean_ms": gp_mean,
        "infer_mean_ms": infer_mean,
        "total_mean_ms": total_mean,
        "throughput_alerts_per_sec": throughput_per_sec,
        "hours_for_10M": hours_for_10M,
    }


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the DeepRubin-Explorer inference pipeline."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of samples to benchmark (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up runs before measurement (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model weights .pth file (default: auto-detect)",
    )
    args = parser.parse_args()

    # Auto-detect model path relative to project root
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    model_path = Path(args.model) if args.model else root_dir / "models" / "rubin_tcn_model.pth"

    if not model_path.exists():
        print(f"ERROR: Model file not found at '{model_path}'.")
        print("       Train the model first with: python src/train.py")
        raise SystemExit(1)

    run_benchmark(
        model_path=model_path,
        n_samples=args.n,
        n_warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
