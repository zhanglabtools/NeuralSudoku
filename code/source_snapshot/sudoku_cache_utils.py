"""Fast loader for either a Sudoku NPZ cache or extracted memory-mapped arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ArrayDirectory:
    def __init__(self, path):
        root = Path(path)
        self.files = [item.stem for item in sorted(root.glob("*.npy"))]
        self._arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in self.files
        }

    def __getitem__(self, key):
        return self._arrays[key]


class IndexedArray:
    """Array-like split view that gathers only the requested rows."""

    def __init__(self, array, indices, trailing_shape=()):
        self.array = array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.trailing_shape = tuple(trailing_shape)

    def __len__(self):
        return len(self.indices)

    @property
    def shape(self):
        return (len(self.indices), *self.trailing_shape)

    def __getitem__(self, item):
        values = self.array[self.indices[item]]
        if self.trailing_shape:
            values = values.reshape(-1, *self.trailing_shape) if np.asarray(values).ndim > 1 else values.reshape(self.trailing_shape)
        return values


class LazyKaggleSudokuDataset:
    """Dataset-compatible split over memory-mapped full-cache arrays."""

    def __init__(self, arrays, split, limit=0):
        indices = np.flatnonzero(np.asarray(arrays["splits"]) == split)
        if limit and len(indices) > limit:
            indices = indices[:limit]
        self.puzzles = IndexedArray(arrays["puzzles"], indices, (9, 9))
        self.solutions = IndexedArray(arrays["solutions"], indices, (9, 9))
        self.clues = np.asarray(arrays["clues"][indices])
        self.ratings = np.asarray(arrays["ratings"][indices])

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        # Return NumPy values and let PyTorch's default collate convert them
        # to tensors.  This keeps the large full cache memory-mapped instead
        # of materializing an entire split in RAM.
        return {
            "puzzle": np.asarray(self.puzzles[idx], dtype=np.int64),
            "solution": np.asarray(self.solutions[idx], dtype=np.int64),
            "clues": int(self.clues[idx]),
            "rating": float(self.ratings[idx]),
        }


def load_sudoku_cache(path):
    target = Path(path)
    if target.is_dir():
        return ArrayDirectory(target)
    return np.load(target, allow_pickle=False)


def load_sudoku_dataset(path, split, limit=0):
    target = Path(path)
    arrays = load_sudoku_cache(path)
    if target.is_dir():
        return LazyKaggleSudokuDataset(arrays, split=split, limit=limit)
    # Avoid a global dependency/cycle for users that only need array access.
    from kaggle_sudoku_experiment import KaggleSudokuDataset

    return KaggleSudokuDataset(arrays, split=split, limit=limit)
