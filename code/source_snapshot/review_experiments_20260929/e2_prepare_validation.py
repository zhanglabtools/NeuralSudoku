"""Create the shared, outcome-independent A07 validation selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from e2_eval_checkpoints import sha256_file


def prepare(cache_path, output_dir, count=16384, seed=20260929):
    output = Path(output_dir)
    with np.load(cache_path, allow_pickle=False) as data:
        candidates = np.flatnonzero(data["splits"] == 1)
    if count > len(candidates):
        raise ValueError("Not enough validation examples")
    ids = np.sort(np.random.default_rng(seed).choice(candidates, count, replace=False)).astype("<i8")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "global_ids.npy"
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), ids):
            raise ValueError("Existing validation selection differs")
    else:
        np.save(path, ids, allow_pickle=False)
    result = dict(status="complete", split=1, n=count, sample_seed=seed,
                  sampling="uniform without replacement; sorted global IDs; no outcomes",
                  cache_sha256=sha256_file(cache_path), global_ids_path=str(path.resolve()),
                  global_ids_file_sha256=sha256_file(path),
                  global_ids_array_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                  selection_rule="T64 Exact every 5000 completed updates; strict improvement, earliest tie; no test",
                  source_sha256=sha256_file(__file__))
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--count", type=int, default=16384)
    p.add_argument("--seed", type=int, default=20260929)
    print(json.dumps(prepare(**vars(p.parse_args()))))
