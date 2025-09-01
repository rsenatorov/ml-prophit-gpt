"""
Get Tokens - Convert normalized data to token IDs via nearest cluster center (CPU).

Inputs:
  - data/norm/{pair}.csv  with columns: date, open, high, low, close
  - tokenizer/kmeans_centers.npy (required)
  - tokenizer/vocab.json         (for reporting)

Output:
  - data/tokens/{pair}.csv  with columns: date, token
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from collections import Counter
from datetime import datetime

# Add parent for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_manager import data_manager, COIN_PAIRS  # noqa: E402

# --------------------------
# Config
# --------------------------
CONFIG = {
    "batch_rows": 512_000,  # distance batch size
    "test_mode": False,
    "test_days": 30
}

# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tokenizer_assign_cpu")


class Tokenizer:
    def __init__(self):
        self.input_dir = Path("data/norm")
        self.output_dir = Path("data/tokens")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path("tokenizer")

        # Load centers + vocab
        centers_path = self.model_dir / "kmeans_centers.npy"
        if not centers_path.exists():
            logger.error(f"Missing centers file: {centers_path}. Run train_tokenizer.py first.")
            sys.exit(1)
        self.centers = np.load(centers_path).astype(np.float32)  # [k, 5]

        vocab_path = self.model_dir / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}

        self.start_time = datetime.now()
        self.total_processed = 0
        self.global_usage = Counter()

    def process_file(self, symbol: str) -> bool:
        fp_in = self.input_dir / f"{symbol.lower()}.csv"
        fp_out = self.output_dir / f"{symbol.lower()}.csv"
        if not fp_in.exists():
            logger.warning(f"Missing normalized data for {symbol}, skipping.")
            return False

        try:
            df = pd.read_csv(fp_in)
            if df.empty:
                logger.warning(f"No rows in {fp_in}, skipping.")
                return False

            if CONFIG["test_mode"]:
                start_idx, _ = data_manager.get_test_mode_indices(len(df), CONFIG["test_days"])
                df = df.iloc[start_idx:].reset_index(drop=True)
                if df.empty:
                    logger.warning(f"No test rows for {symbol}, skipping.")
                    return False

            # Build 5D features
            df["direction"] = df["close"] - df["open"]
            X = df[["open", "high", "low", "close", "direction"]].values.astype(np.float32)

            # Assign nearest center in batches (CPU)
            tokens = self._assign_tokens_batched(X)

            out = pd.DataFrame({"date": df["date"].values, "token": tokens})
            out.to_csv(fp_out, index=False)

            self.total_processed += len(tokens)
            self.global_usage.update(tokens)
            return True

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # --------------------------
    # Assignment (CPU batched)
    # --------------------------
    def _assign_tokens_batched(self, X_np: np.ndarray) -> np.ndarray:
        n = X_np.shape[0]
        C = self.centers  # [k,5]
        k = C.shape[0]
        batch = int(CONFIG["batch_rows"])
        tokens = np.empty((n,), dtype=np.int32)

        with tqdm(total=n, desc="Assign (CPU)", unit="row", leave=False) as pbar:
            start = 0
            while start < n:
                end = min(start + batch, n)
                xb = X_np[start:end][:, None, :]                      # [B,1,5]
                d2 = np.sum((xb - C[None, :, :]) ** 2, axis=2)        # [B,k]
                lbl = np.argmin(d2, axis=1)
                tokens[start:end] = lbl
                pbar.update(end - start)
                start = end

        return tokens

    # --------------------------
    # Summary
    # --------------------------
    def print_summary(self, ok_pairs, bad_pairs):
        elapsed = datetime.now() - self.start_time
        rate = self.total_processed / max(elapsed.total_seconds(), 1e-9)
        print("\n" + "=" * 80)
        print(" " * 28 + "TOKENIZATION SUMMARY")
        print("=" * 80)
        print(f"  Status           : Done")
        print(f"  Engine           : CPU (NumPy)")
        print(f"  Elapsed          : {str(elapsed).split('.')[0]}")
        print(f"  Rows Tokenized   : {self.total_processed:,}")
        print(f"  Throughput       : {rate:,.0f} rows/sec")
        print(f"  Unique Tokens    : {len(self.global_usage):,} / {self.centers.shape[0]} used")
        if self.global_usage:
            top = self.global_usage.most_common(10)
            print("\n  Top 10 tokens:")
            print("  Token  Count")
            print("  -----  ---------")
            for t, c in top:
                print(f"  {t:<5}  {c:,}")
        print("=" * 80 + "\n")

    # --------------------------
    # Run
    # --------------------------
    def run(self):
        os.system("cls" if os.name == "nt" else "clear")
        print("\n" + "=" * 80)
        print(" " * 26 + "DATA TOKENIZER (CPU)")
        print("=" * 80 + "\n")
        print(f"  Centers: {self.centers.shape[0]} tokens")
        print(f"  Mode   : {'TEST (recent)' if CONFIG['test_mode'] else 'FULL DATASET'}")
        print("=" * 80 + "\n")

        ok, bad = [], []
        for symbol in tqdm(COIN_PAIRS, desc="Pairs", unit="pair"):
            if self.process_file(symbol):
                ok.append(symbol)
            else:
                bad.append(symbol)

        self.print_summary(ok, bad)


def main():
    t = Tokenizer()
    t.run()


if __name__ == "__main__":
    main()
