"""
Train Tokenizer - CPU K-Means to create a 1024-token candle vocabulary over
5D features: [open, high, low, close, direction], where direction = close - open.

No GPU / RAPIDS required.
Shows per-iteration progress with tqdm. Uses batched distance computation.

Outputs:
  - tokenizer/kmeans_centers.npy   (float32, shape [n_clusters, 5])
  - tokenizer/kmeans_meta.json     (engine='numpy', inertia, iters, tol, etc.)
  - tokenizer/vocab.json           (per-token representative OHLC + direction)
  - tokenizer/kmeans_model.joblib  (dict with centers + meta for convenience)
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_manager import data_manager, COIN_PAIRS  # noqa: E402

# --------------------------
# Configuration
# --------------------------
CONFIG = {
    "num_clusters": 1024,       # tokens
    "max_iter": 10,            # maximum Lloyd iterations
    "tol": 1e-4,                # stop when max center shift <= tol
    "random_state": 42,         # reproducibility
    "data_sample_fraction": 1.0, # 1.0 = use all rows
    "batch_rows": 100_000,      # distance batch size for assignment
    "init": "k-means++",        # "random" | "k-means++"
    "kpp_sample": 100_000,      # subsample size used for k-means++ seeding (for speed)
    "save_model_pickle": True
}

# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("kmeans_tokenizer_cpu")


class KMeansTokenizerTrainer:
    """CPU K-Means with tqdm and batched assignment."""

    def __init__(self):
        self.norm_dir = Path("data/norm")
        self.out_dir = Path("tokenizer")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()

    # --------------------------
    # Data loading
    # --------------------------
    def load_all_normalized_data(self) -> np.ndarray:
        """
        Loads normalized OHLC, adds direction, returns float32 ndarray [N, 5].
        """
        all_5d = []
        logger.info("Loading normalized data and adding direction dimension...")
        for symbol in tqdm(COIN_PAIRS, desc="Loading files", unit="file"):
            fp = self.norm_dir / f"{symbol.lower()}.csv"
            if not fp.exists():
                continue
            try:
                df = pd.read_csv(fp, usecols=["open", "high", "low", "close"])
                if df.empty:
                    continue
                df["direction"] = df["close"] - df["open"]
                all_5d.append(df.values)
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")

        if not all_5d:
            logger.error("No normalized data found. Run get_norm.py first.")
            sys.exit(1)

        X = np.concatenate(all_5d, axis=0).astype(np.float32)
        logger.info(f"Loaded {X.shape[0]:,} candles (5D).")

        if CONFIG["data_sample_fraction"] < 1.0:
            n = int(len(X) * CONFIG["data_sample_fraction"])
            rng = np.random.default_rng(CONFIG["random_state"])
            idx = rng.choice(len(X), size=n, replace=False)
            X = X[idx]
            logger.info(f"Sampled {n:,} candles ({CONFIG['data_sample_fraction'] * 100:.0f}%).")

        return X

    # --------------------------
    # Initialization
    # --------------------------
    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        k = CONFIG["num_clusters"]
        rs = CONFIG["random_state"]
        rng = np.random.default_rng(rs)

        if CONFIG["init"] == "random":
            idx = rng.choice(len(X), size=k, replace=False)
            return X[idx].copy()

        # k-means++ with optional subsample for speed
        n = X.shape[0]
        if n > CONFIG["kpp_sample"]:
            idx_sub = rng.choice(n, size=CONFIG["kpp_sample"], replace=False)
            Xk = X[idx_sub]
        else:
            Xk = X

        centers = np.empty((k, X.shape[1]), dtype=X.dtype)
        # pick first at random
        i0 = rng.integers(0, len(Xk))
        centers[0] = Xk[i0]

        # squared distances to nearest chosen center
        d2 = np.sum((Xk - centers[0][None, :]) ** 2, axis=1)

        for c in tqdm(range(1, k), desc="k-means++ init", unit="center"):
            probs = d2 / (d2.sum() + 1e-12)
            # draw index by probabilities
            i = rng.choice(len(Xk), p=probs)
            centers[c] = Xk[i]
            # update d2
            new_d2 = np.sum((Xk - centers[c][None, :]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)

        return centers

    # --------------------------
    # Training (Lloyd with batched assignment)
    # --------------------------
    def train_model(self, X: np.ndarray):
        """
        Returns (centers, stats)
        centers: float32 [k, d]
        stats: dict (iters, inertia, fit_seconds, tol, num_clusters, engine)
        """
        t0 = time.time()
        n, d = X.shape
        k = CONFIG["num_clusters"]
        batch = CONFIG["batch_rows"]

        centers = self._init_centers(X)
        labels = np.empty((n,), dtype=np.int32)

        for it in tqdm(range(1, CONFIG["max_iter"] + 1), desc="K-Means (CPU)", unit="iter"):
            # Assignment (batched)
            inertia = 0.0
            sums = np.zeros((k, d), dtype=np.float64)
            counts = np.zeros((k,), dtype=np.int64)

            with tqdm(total=n, desc="  assign", unit="row", leave=False) as pbar_assign:
                start = 0
                while start < n:
                    end = min(start + batch, n)
                    xb = X[start:end]                                  # [B, d]
                    # squared distances to all centers -> [B, k]
                    d2 = np.sum((xb[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                    lb = np.argmin(d2, axis=1)                          # [B]
                    labels[start:end] = lb
                    inertia += float(d2[np.arange(end - start), lb].sum())

                    # accumulate per-cluster sums via bincount
                    for dim in range(d):
                        sums[:, dim] += np.bincount(lb, weights=xb[:, dim], minlength=k)
                    counts += np.bincount(lb, minlength=k)

                    pbar_assign.update(end - start)
                    start = end

            # Update
            new_centers = centers.copy()
            nonzero = counts > 0
            new_centers[nonzero] = (sums[nonzero] / counts[nonzero, None]).astype(np.float32)

            # Re-seed any empty clusters to random points
            if np.any(~nonzero):
                nz = np.where(~nonzero)[0]
                rng = np.random.default_rng(CONFIG["random_state"] + it)
                idx = rng.choice(n, size=len(nz), replace=False)
                new_centers[nz] = X[idx]

            # Convergence check
            shift = float(np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))))
            centers = new_centers

            # Update outer tqdm postfix
            tqdm.write(f"iter={it} inertia={inertia:,.0f} shift={shift:.6f}")

            if shift <= CONFIG["tol"]:
                break

        fit_time = time.time() - t0
        centers = centers.astype(np.float32)

        stats = {
            "engine": "numpy",
            "iters": it,
            "inertia": float(inertia),
            "fit_seconds": float(fit_time),
            "tol": CONFIG["tol"],
            "num_clusters": k
        }
        return centers, stats

    # --------------------------
    # Save artifacts
    # --------------------------
    def save_centers_and_meta(self, centers: np.ndarray, stats: dict):
        centers_path = self.out_dir / "kmeans_centers.npy"
        meta_path = self.out_dir / "kmeans_meta.json"
        np.save(centers_path, centers)
        with open(meta_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ Saved centers to {centers_path}")
        logger.info(f"✓ Saved meta to {meta_path}")

        if CONFIG["save_model_pickle"]:
            try:
                joblib.dump({"engine": "numpy", "centers": centers, "meta": stats}, self.out_dir / "kmeans_model.joblib")
            except Exception as e:
                logger.warning(f"Could not write kmeans_model.joblib (okay): {e}")

    def create_and_save_vocabulary(self, centers: np.ndarray):
        """
        Make vocab.json mapping token_id -> representative OHLC + direction.
        Ensures OHLC ordering consistency (high=max, low=min) for readability.
        """
        vocab = {}
        for i, c in enumerate(centers):
            o, h, l, cl = float(c[0]), float(c[1]), float(c[2]), float(c[3])
            high = max(o, h, l, cl)
            low = min(o, h, l, cl)
            vocab[str(i)] = {
                "open": o,
                "high": high,
                "low": low,
                "close": cl,
                "direction": "bullish" if cl > o else "bearish" if cl < o else "neutral"
            }

        path = self.out_dir / "vocab.json"
        with open(path, "w") as f:
            json.dump(vocab, f, indent=4)
        logger.info(f"✓ Vocabulary saved to {path} ({len(vocab)} tokens)")

    # --------------------------
    # UX / summary
    # --------------------------
    def print_summary(self, n_rows: int, centers: np.ndarray, stats: dict):
        elapsed = datetime.now() - self.start_time
        print("\n" + "=" * 80)
        print(" " * 22 + "CPU K-MEANS TOKENIZER TRAINING SUMMARY")
        print("=" * 80)
        print(f"  Status            : Done")
        print(f"  Total Time        : {str(elapsed).split('.')[0]}")
        print(f"  Engine            : {stats.get('engine')}")
        print(f"  Clusters (Tokens) : {stats.get('num_clusters')}")
        print(f"  Iterations        : {stats.get('iters')}")
        print(f"  Inertia           : {stats.get('inertia'):,.0f}")
        print(f"  Data Rows         : {n_rows:,}")
        print(f"  Outputs           : tokenizer/kmeans_centers.npy")
        print(f"                      tokenizer/kmeans_meta.json")
        print(f"                      tokenizer/vocab.json")
        print("=" * 80 + "\n")

    # --------------------------
    # Run
    # --------------------------
    def run(self):
        os.system("cls" if os.name == "nt" else "clear")
        print("\n" + "=" * 80)
        print(" " * 18 + "STARTING CPU K-MEANS TOKENIZER TRAINING")
        print("=" * 80 + "\n")

        X = self.load_all_normalized_data()
        centers, stats = self.train_model(X)
        self.save_centers_and_meta(centers, stats)
        self.create_and_save_vocabulary(centers)
        self.print_summary(len(X), centers, stats)


def main():
    trainer = KMeansTokenizerTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
