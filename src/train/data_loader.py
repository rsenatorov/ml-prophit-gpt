# data_loader.py
"""
Data loading and processing for tokenized OHLC data with epoch tracking.
Provides a continuous, burst-free streaming dataset and perfect mid-epoch resume.
Enhanced with curriculum learning and smooth epoch transitions.
"""

import logging
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

logger = logging.getLogger(__name__)


def _list_pair_csvs(tokens_dir: Path) -> List[Path]:
    tokens_dir = Path(tokens_dir)
    csvs = sorted(tokens_dir.glob("*.csv"))
    return csvs


class TokenDataset(Dataset):
    """
    Map-style dataset: loads pair CSVs into memory once, builds full windows for validation.
    """
    def __init__(
        self,
        tokens_dir: str,
        sequence_length: int,
        split: str,
        train_val_boundary: str,
        val_start: str,
        val_end: str,
        test_start: str,
        volatility_config: Optional[Dict[str, Any]] = None,
        vocab_size: int = 1024,
    ):
        self.tokens_dir = Path(tokens_dir)
        self.sequence_length = int(sequence_length)
        self.split = split
        self.volatility_config = volatility_config or {}
        self.vocab_size = int(vocab_size)

        self.train_val_boundary = str(train_val_boundary)
        self.val_start = str(val_start)
        self.val_end = str(val_end)
        self.test_start = str(test_start)

        self.data_source: List[Tuple[str, np.ndarray]] = []
        self.index_map: List[Tuple[int, int]] = []

        self._load_all_pairs()
        self._build_index()

    def _load_all_pairs(self):
        csv_files = _list_pair_csvs(self.tokens_dir)
        if not csv_files:
            logger.warning(f"No CSV files found in {self.tokens_dir.resolve()}")
        for csv_file in csv_files:
            try:
                logger.info(f"Loading {csv_file.stem}...")
                df = pd.read_csv(csv_file, usecols=["date", "token"])
                if df.empty:
                    continue
                dates = df["date"].astype(str)
                toks = df["token"].values.astype(np.int64)

                if self.split == "train":
                    mask = (dates < self.train_val_boundary).values
                elif self.split == "val":
                    day = dates.str[:10]
                    mask = ((day >= self.val_start) & (day <= self.val_end)).values
                elif self.split == "test":
                    mask = (dates >= self.test_start).values
                else:
                    mask = np.ones_like(toks, dtype=bool)

                toks = toks[mask]
                # Filter out invalid tokens
                toks = toks[(toks >= 0) & (toks < self.vocab_size)]
                if len(toks) < self.sequence_length:
                    continue
                self.data_source.append((csv_file.stem, toks))
            except Exception:
                logger.error(f"Failed to load {csv_file}: {traceback.format_exc()}")

        logger.info(f"Loaded {len(self.data_source)} pairs for split '{self.split}'")

    def _build_index(self):
        for pair_idx, (_, tokens) in enumerate(self.data_source):
            n = len(tokens) - self.sequence_length + 1
            if n <= 0:
                continue
            for start in range(n):
                self.index_map.append((pair_idx, start))
        logger.info(f"Built validation index with {len(self.index_map):,} windows")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i: int):
        pair_idx, start = self.index_map[i]
        pair_name, tokens = self.data_source[pair_idx]
        seq = tokens[start : start + self.sequence_length]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        if self.volatility_config.get("add_volatility_bucket", False):
            vol_bucket = torch.zeros(1, dtype=torch.long)
            return {
                "input_ids": x,
                "labels": y,
                "target_ids": y,
                "pair_id": torch.tensor(pair_idx, dtype=torch.long),
                "vol_bucket": vol_bucket,
                "pair_name": pair_name,
            }
        else:
            return {
                "input_ids": x,
                "labels": y,
                "target_ids": y,
                "pair_id": torch.tensor(pair_idx, dtype=torch.long),
                "pair_name": pair_name,
            }


# =========================
# Training dataset (streaming)
# =========================
class EpochStreamingTokenDataset(Dataset):
    """
    Streaming dataset with perfect epoch coverage and resume.
    Enhanced with:
    - Curriculum learning: gradually introduce harder samples
    - Smooth epoch transitions: blend data distributions
    - Better shuffling: maintain some temporal coherence
    """
    def __init__(
        self,
        tokens_dir: str,
        sequence_length: int,
        split: str,
        train_val_boundary: str,
        val_start: str,
        val_end: str,
        test_start: str,
        cache_dir: Optional[str] = None,
        volatility_config: Optional[Dict[str, Any]] = None,
        vocab_size: int = 1024,
        seed: int = 42,
        curriculum_learning: bool = True,  # New parameter
    ):
        self.tokens_dir = Path(tokens_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.tokens_dir / "_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = int(sequence_length)
        self.split = split
        self.train_val_boundary = str(train_val_boundary)
        self.val_start = str(val_start)
        self.val_end = str(val_end)
        self.test_start = str(test_start)
        self.volatility_config = volatility_config or {}
        self.vocab_size = int(vocab_size)
        self.seed = seed
        self.curriculum_learning = curriculum_learning

        self.pairs: List[Dict[str, Any]] = []
        self._mmaps: List[Optional[np.ndarray]] = []
        self.all_windows: List[Tuple[int, int, int]] = []  # (pair_idx, seg_idx, start_pos)
        self.windows_per_epoch: int = 0

        self.remaining_windows: Set[int] = set()
        self._epoch_perm: List[int] = []
        self._epoch_ptr: int = 0
        self.epoch_number: int = 0
        
        # For curriculum learning
        self.window_difficulties: Optional[np.ndarray] = None

        self._scan_pairs()
        self._build_all_windows()
        if self.curriculum_learning:
            self._compute_window_difficulties()
        self.reset_epoch()

    @staticmethod
    def _find_segments(valid_mask: np.ndarray, seq_len: int) -> List[Tuple[int, int, int]]:
        """Return list of (start_idx, end_idx, n_starts) for contiguous True segments."""
        if not valid_mask.any():
            return []
        idx = np.flatnonzero(valid_mask)
        breaks = np.where(np.diff(idx) != 1)[0] + 1
        parts = np.split(idx, breaks)
        segments: List[Tuple[int, int, int]] = []
        for part in parts:
            start = int(part[0])
            end = int(part[-1])
            n_starts = (end - start + 1) - seq_len + 1
            if n_starts > 0:
                segments.append((start, end, n_starts))
        return segments

    def _ensure_cache(self, stem: str, tokens: np.ndarray) -> Path:
        npy_path = self.cache_dir / f"{stem}.tokens.int16.npy"
        if not npy_path.exists():
            np.save(npy_path, tokens.astype(np.int16), allow_pickle=False)
        return npy_path

    def _scan_pairs(self):
        csv_files = _list_pair_csvs(self.tokens_dir)
        if not csv_files:
            logger.warning(f"No CSV files found in {self.tokens_dir.resolve()}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, usecols=["date", "token"])
                if df.empty:
                    continue
                dates = df["date"].astype(str)
                tokens = df["token"].values.astype(np.int64)

                mask_split = (dates < self.train_val_boundary).values
                # Enhanced validation: check token range
                valid_mask = (tokens >= 0) & (tokens < self.vocab_size) & mask_split

                segments = self._find_segments(valid_mask, self.sequence_length)
                if not segments:
                    continue

                npy_path = self._ensure_cache(csv_file.stem, tokens)
                n_windows = int(sum(s[2] for s in segments))

                self.pairs.append({
                    "name": csv_file.stem,
                    "csv": csv_file,
                    "npy": npy_path,
                    "segments": segments,
                    "n_windows": n_windows,
                    "n_tokens": len(tokens),
                })
            except Exception:
                logger.error(f"Failed to load {csv_file}: {traceback.format_exc()}")

        self._mmaps = [None] * len(self.pairs)
        logger.info(f"Loaded {len(self.pairs)} train pairs")

    def _build_all_windows(self):
        self.all_windows.clear()
        for pair_idx, pair in enumerate(self.pairs):
            segs = pair["segments"]
            for seg_idx, (start, end, n_starts) in enumerate(segs):
                for s in range(n_starts):
                    self.all_windows.append((pair_idx, seg_idx, start + s))
        self.windows_per_epoch = len(self.all_windows)
        logger.info(f"Total train windows per epoch: {self.windows_per_epoch:,}")

    def _compute_window_difficulties(self):
        """Compute difficulty scores for curriculum learning based on token diversity."""
        difficulties = []
        for pair_idx, seg_idx, start_pos in self.all_windows:
            # Simple heuristic: windows from later segments are "harder"
            # You could also compute actual token entropy here
            difficulty = pair_idx * 0.1 + seg_idx * 0.01 + start_pos * 0.0001
            difficulties.append(difficulty)
        self.window_difficulties = np.array(difficulties)
        # Normalize to [0, 1]
        self.window_difficulties = (self.window_difficulties - self.window_difficulties.min()) / \
                                   (self.window_difficulties.max() - self.window_difficulties.min() + 1e-8)

    def _curriculum_shuffle(self, items: List[int], epoch: int) -> List[int]:
        """
        Curriculum-aware shuffling: gradually introduce harder samples.
        Early epochs focus on easier samples, later epochs see full distribution.
        """
        if not self.curriculum_learning or self.window_difficulties is None:
            return self._deterministic_shuffle(items, epoch)
        
        # Curriculum schedule: exponentially increase difficulty range
        max_epochs_for_curriculum = 50  # After this, use full distribution
        if epoch >= max_epochs_for_curriculum:
            return self._deterministic_shuffle(items, epoch)
        
        # Compute difficulty threshold for this epoch
        progress = min(1.0, epoch / max_epochs_for_curriculum)
        # Use exponential schedule for smoother progression
        difficulty_threshold = 1.0 - np.exp(-3 * progress)  # Ranges from ~0 to ~0.95
        
        # Filter windows by difficulty
        rng = random.Random(self.seed + epoch * 1000)
        eligible_items = []
        for idx in items:
            if self.window_difficulties[idx] <= difficulty_threshold:
                eligible_items.append(idx)
        
        # If too few samples, gradually include harder ones
        min_samples = max(1000, len(items) // 10)  # At least 10% of data
        if len(eligible_items) < min_samples:
            # Sort by difficulty and take the easiest min_samples
            sorted_items = sorted(items, key=lambda x: self.window_difficulties[x])
            eligible_items = sorted_items[:min_samples]
        
        # Shuffle the eligible items
        rng.shuffle(eligible_items)
        
        # For remaining capacity, sample from harder examples with lower probability
        remaining = list(set(items) - set(eligible_items))
        if remaining and len(eligible_items) < len(items):
            # Sample some harder examples to maintain diversity
            n_hard = min(len(remaining), (len(items) - len(eligible_items)) // 2)
            hard_samples = rng.sample(remaining, n_hard)
            eligible_items.extend(hard_samples)
            rng.shuffle(eligible_items)  # Final shuffle
        
        return eligible_items

    def _deterministic_shuffle(self, items: List[int], epoch: int) -> List[int]:
        """
        Deterministically shuffle a list based on epoch number.
        This ensures the same shuffle for a given epoch, even after resume.
        """
        rng = random.Random(self.seed + epoch * 1000)
        shuffled = items.copy()
        rng.shuffle(shuffled)
        return shuffled

    def reset_epoch(self):
        """Start a new epoch with a fresh shuffled permutation."""
        n = len(self.all_windows)
        self.remaining_windows = set(range(n))
        
        # Use curriculum-aware shuffling if enabled
        if self.curriculum_learning:
            self._epoch_perm = self._curriculum_shuffle(list(range(n)), self.epoch_number)
        else:
            self._epoch_perm = self._deterministic_shuffle(list(range(n)), self.epoch_number)
            
        self._epoch_ptr = 0
        self.epoch_number += 1
        logger.info(f"Starting epoch {self.epoch_number}: {len(self._epoch_perm):,}/{n:,} windows")

    # ======= COMPACT PROGRESS SAVE (FIX) =======
    def get_epoch_progress(self) -> Dict[str, Any]:
        return {
            "epoch_number": int(self.epoch_number),
            "_epoch_ptr": int(self._epoch_ptr),
            "windows_per_epoch": int(self.windows_per_epoch),
            "seed": int(self.seed),
            "curriculum_learning": bool(self.curriculum_learning),
        }

    # ======= DETERMINISTIC REBUILD ON RESUME (FIX) =======
    def load_epoch_progress(self, progress: Dict[str, Any]):
        # Restore minimal state
        self.epoch_number = int(progress.get("epoch_number", 0))
        ptr = int(progress.get("_epoch_ptr", 0))
        self.windows_per_epoch = int(progress.get("windows_per_epoch", self.windows_per_epoch))
        self.curriculum_learning = bool(progress.get("curriculum_learning", self.curriculum_learning))
        self.seed = int(progress.get("seed", self.seed))

        # Rebuild permutation deterministically for this epoch label
        n = len(self.all_windows)
        self.remaining_windows = set(range(n))

        current_epoch = self.epoch_number - 1 if self.epoch_number > 0 else 0
        items = list(range(n))
        if self.curriculum_learning:
            self._epoch_perm = self._curriculum_shuffle(items, current_epoch)
        else:
            self._epoch_perm = self._deterministic_shuffle(items, current_epoch)

        # Clamp and consume up to ptr
        self._epoch_ptr = max(0, min(ptr, len(self._epoch_perm)))
        for i in range(self._epoch_ptr):
            w_idx = self._epoch_perm[i]
            if w_idx in self.remaining_windows:
                self.remaining_windows.remove(w_idx)

        logger.info(
            f"Resumed epoch {self.epoch_number}: "
            f"{len(self.remaining_windows):,}/{self.windows_per_epoch:,} windows remain "
            f"(ptr={self._epoch_ptr})"
        )

    def __len__(self):
        return max(1, self.windows_per_epoch)

    def _open_memmap_if_needed(self, pair_idx: int) -> np.ndarray:
        mm = self._mmaps[pair_idx]
        if mm is None:
            mm = np.load(self.pairs[pair_idx]["npy"], mmap_mode="r")
            self._mmaps[pair_idx] = mm
        return mm

    def __getitem__(self, i: int):
        if not self.remaining_windows:
            self.reset_epoch()

        # Advance pointer to next available window
        attempts = 0
        max_attempts = len(self._epoch_perm) + 100  # Safety limit
        
        while attempts < max_attempts:
            if self._epoch_ptr >= len(self._epoch_perm):
                # Exhausted permutation; safety reset
                self.reset_epoch()
            w_idx = self._epoch_perm[self._epoch_ptr]
            self._epoch_ptr += 1
            attempts += 1
            
            if w_idx in self.remaining_windows:
                self.remaining_windows.remove(w_idx)
                break
        else:
            # Fallback: just take any remaining window
            if self.remaining_windows:
                w_idx = self.remaining_windows.pop()
            else:
                self.reset_epoch()
                w_idx = self._epoch_perm[0]
                self._epoch_ptr = 1
                self.remaining_windows.discard(w_idx)

        pair_idx, seg_idx, start_pos = self.all_windows[w_idx]
        mm = self._open_memmap_if_needed(pair_idx)
        seq_len = self.sequence_length
        end_pos = start_pos + seq_len
        
        # Validate bounds
        if end_pos > len(mm):
            # Safety: return a valid window from the beginning
            logger.warning(f"Invalid window bounds for pair {self.pairs[pair_idx]['name']}: [{start_pos}:{end_pos}] exceeds {len(mm)}")
            start_pos = 0
            end_pos = seq_len
            
        seq = np.asarray(mm[start_pos:end_pos], dtype=np.int32)
        
        # Additional validation
        if seq.shape[0] != seq_len:
            # Pad if necessary
            if seq.shape[0] < seq_len:
                seq = np.pad(seq, (0, seq_len - seq.shape[0]), mode='constant', constant_values=0)
            else:
                seq = seq[:seq_len]
        
        # Clamp to valid token range
        seq = np.clip(seq, 0, self.vocab_size - 1)

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        if self.volatility_config.get("add_volatility_bucket", False):
            vol_bucket = torch.zeros(1, dtype=torch.long)
            return {
                "input_ids": x,
                "labels": y,
                "target_ids": y,
                "pair_id": torch.tensor(pair_idx, dtype=torch.long),
                "vol_bucket": vol_bucket,
                "pair_name": self.pairs[pair_idx]["name"],
            }
        else:
            return {
                "input_ids": x,
                "labels": y,
                "target_ids": y,
                "pair_id": torch.tensor(pair_idx, dtype=torch.long),
                "pair_name": self.pairs[pair_idx]["name"],
            }


class EpochAwareSampler(Sampler[int]):
    """Sampler that yields a fixed number of indices per epoch."""
    def __init__(self, data_source: 'EpochStreamingTokenDataset'):
        assert isinstance(data_source, EpochStreamingTokenDataset)
        self.data_source = data_source
        self.epoch_length = data_source.windows_per_epoch

    def __iter__(self):
        # Dataset handles actual randomization and window selection
        for i in range(self.epoch_length):
            yield i

    def __len__(self):
        return self.epoch_length


def create_dataloaders(config):
    logger.info("Creating training dataset...")
    train_dataset = EpochStreamingTokenDataset(
        tokens_dir=config.data.tokens_dir,
        sequence_length=config.data.sequence_length,
        split="train",
        train_val_boundary=config.data.train_val_boundary,
        val_start=config.data.val_start,
        val_end=config.data.val_end,
        test_start=config.data.test_start,
        cache_dir=config.data.cache_dir,
        volatility_config={
            "add_volatility_bucket": config.data.add_volatility_bucket,
            "volatility_num_buckets": config.data.volatility_num_buckets,
            "volatility_window_steps": config.data.volatility_window_steps,
            "volatility_embedding_dim": config.data.volatility_embedding_dim,
        },
        vocab_size=config.model.vocab_size,
        seed=config.seed,
        curriculum_learning=getattr(config.data, 'curriculum_learning', True),  # Enable by default
    )

    logger.info("Creating validation dataset...")
    val_dataset = TokenDataset(
        tokens_dir=config.data.tokens_dir,
        sequence_length=config.data.sequence_length,
        split="val",
        train_val_boundary=config.data.train_val_boundary,
        val_start=config.data.val_start,
        val_end=config.data.val_end,
        test_start=config.data.test_start,
        volatility_config={
            "add_volatility_bucket": config.data.add_volatility_bucket,
            "volatility_num_buckets": config.data.volatility_num_buckets,
            "volatility_window_steps": config.data.volatility_window_steps,
            "volatility_embedding_dim": config.data.volatility_embedding_dim,
        },
        vocab_size=config.model.vocab_size,
    )

    common = dict(
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        multiprocessing_context=config.data.multiprocessing_context,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=EpochAwareSampler(train_dataset),
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        **common,
    )

    logger.info(
        f"  Train: {train_dataset.windows_per_epoch:,} windows/epoch, "
        f"{len(train_loader):,} batches/epoch"
    )
    logger.info(
        f"  Val:   {len(val_dataset):,} windows, {len(val_loader):,} batches"
    )

    return train_loader, val_loader
