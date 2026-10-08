"""Portable file and model loading; no changes to the model or selection rule."""
from pathlib import Path
import hashlib
import sys

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent
SOURCE = CODE / 'source_snapshot'
REVIEW = SOURCE / 'review_experiments_20260929'


def add_source_paths():
    for path in (SOURCE, REVIEW):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'File does not exist: {path}')
    return path


def package_label(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PACKAGE).as_posix()
    except ValueError:
        return path.name


def load_symbolic_explicit(reflection_path, backbone_path, device):
    """Ignore the serialized base path and verify the explicitly supplied weights."""
    add_source_paths()
    import torch
    from eval_hybrid_hyper_rrn_restarts import load_hybrid
    from train_symbolic_primal_dual_reflection import SymbolicPrimalDualCfg, SymbolicPrimalDualReflector
    backbone, _, _ = load_hybrid(str(file_path(backbone_path)), 'cpu')
    checkpoint = torch.load(file_path(reflection_path), map_location='cpu', weights_only=False)
    state = checkpoint['model_state']
    embedded = {key[len('backbone.'):]: value for key, value in state.items() if key.startswith('backbone.')}
    if embedded:
        supplied = backbone.state_dict()
        if set(embedded) != set(supplied) or any(not torch.equal(value, supplied[key]) for key, value in embedded.items()):
            raise ValueError('Explicit backbone weights do not match the backbone stored in the reflection checkpoint.')
    cfg = SymbolicPrimalDualCfg(**checkpoint['reflection_cfg'])
    model = SymbolicPrimalDualReflector(backbone, cfg)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg, checkpoint


def load_iterative_explicit(reflection_path, backbone_path, device):
    add_source_paths()
    import torch
    from eval_hybrid_hyper_rrn_restarts import load_hybrid
    from train_iterative_hyper_reflection import IterativeReflectionCfg, IterativeHyperReflector
    backbone, _, _ = load_hybrid(str(file_path(backbone_path)), 'cpu')
    checkpoint = torch.load(file_path(reflection_path), map_location='cpu', weights_only=False)
    cfg = IterativeReflectionCfg(**checkpoint['reflection_cfg'])
    model = IterativeHyperReflector(backbone, cfg)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    return model, cfg, checkpoint
