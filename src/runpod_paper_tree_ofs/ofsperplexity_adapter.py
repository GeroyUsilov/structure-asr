"""Optional adapter for bryan-nm/OFSperplexity.

This is a third-party implementation, not the DCA authors' repository and not a
bundled pretrained head. A head trained for the selected encoder is mandatory.
"""

from __future__ import annotations

from pathlib import Path


_CACHE = {}


def make_profile_predictor(parameters):
    from ofsperplexity import OFSHead, OFSScorer, collate
    from ofsperplexity.models import load_encoder

    model_kind = str(parameters.get("model_kind", "esmc"))
    model_path = str(parameters["model_path"])
    head_path = str(parameters["head_path"])
    device = str(parameters.get("device", "cuda"))
    if not Path(head_path).is_file():
        raise FileNotFoundError(f"trained OFS head not found: {head_path}")
    key = (model_kind, model_path, head_path, device)
    if key not in _CACHE:
        encoder = load_encoder(model_kind, model_path, device=device)
        head = OFSHead.load(head_path)
        scorer = OFSScorer(encoder.adapter, head)
        _CACHE[key] = (encoder, scorer)
    encoder, scorer = _CACHE[key]

    def predict_profile(sequence: str):
        batch = collate([("query", sequence)], encoder.encode_fn, encoder.alphabet)
        profile = scorer.profiles(batch)[0]
        valid = batch.valid[0].to(profile.device)
        profile = profile[valid]
        if profile.shape != (len(sequence), 20):
            raise RuntimeError(
                f"OFS provider returned {tuple(profile.shape)} for sequence length {len(sequence)}"
            )
        return profile.detach().float().cpu().numpy()

    return predict_profile

