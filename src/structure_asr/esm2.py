"""Exact single-masked-site conditional profiles from pretrained ESM2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
_MODEL_CACHE = {}


@dataclass
class ESM2MaskedSitePredictor:
    model: object
    tokenizer: object
    device: object
    aa_token_ids: object
    torch: object
    timings_path: Path | None = None
    progress: object | None = None
    calls: int = 0

    def predict_site_profile(self, sequence: str, site: int) -> np.ndarray:
        call_started = perf_counter()
        if not 0 <= site < len(sequence):
            raise IndexError(f"site {site} is outside sequence length {len(sequence)}")
        if set(sequence) - set(CANONICAL_AA):
            raise ValueError("ESM2 exact generator accepts only canonical amino acids")
        encoded = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if input_ids.shape != (1, len(sequence) + 2):
            raise RuntimeError(
                "ESM2 tokenizer did not produce one token per residue plus BOS/EOS: "
                f"got {tuple(input_ids.shape)} for length {len(sequence)}"
            )
        token_position = site + 1
        input_ids[0, token_position] = int(self.tokenizer.mask_token_id)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        forward_started = perf_counter()
        with self.torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        forward_seconds = perf_counter() - forward_started
        aa_logits = logits[0, token_position, self.aa_token_ids].float()
        probabilities = self.torch.softmax(aa_logits, dim=-1)
        profile = probabilities.detach().cpu().numpy()
        if profile.shape != (20,) or not np.isfinite(profile).all():
            raise RuntimeError("ESM2 returned an invalid site profile")
        end_to_end_seconds = perf_counter() - call_started
        self.calls += 1
        if self.timings_path is not None:
            with self.timings_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(
                    f"{self.calls}\t{len(sequence)}\t{site}\t"
                    f"{forward_seconds:.9f}\t{end_to_end_seconds:.9f}\n"
                )
        if self.progress is not None:
            self.progress.update(1)
            if self.progress.total is not None and self.calls >= self.progress.total:
                self.progress.close()
        return profile

    __call__ = predict_site_profile


def make_site_predictor(parameters: Mapping):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_name = str(parameters.get("model_name", "facebook/esm2_t33_650M_UR50D"))
    device_name = str(parameters.get("device", "cuda"))
    dtype_name = str(parameters.get("dtype", "float16"))
    timings_value = parameters.get("timings_path")
    resume_timings = bool(parameters.get("resume_timings", False))
    progress_total = int(parameters.get("progress_total", 0))
    progress_description = str(parameters.get("progress_description", "ESM2 passes"))
    timings_path = (
        Path(str(timings_value)).expanduser().resolve()
        if timings_value is not None
        else None
    )
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unknown torch dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    device = torch.device(device_name)
    key = (model_name, str(device), dtype_name)
    if key not in _MODEL_CACHE:
        print(f"Loading exact masked-site encoder {model_name} on {device} ({dtype_name})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=dtype)
        model = model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        aa_ids = tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
        if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in aa_ids):
            raise RuntimeError("ESM2 tokenizer does not expose all 20 canonical amino acids")
        _MODEL_CACHE[key] = (
            model,
            tokenizer,
            torch.tensor(aa_ids, dtype=torch.long, device=device),
        )
    initial_calls = 0
    if timings_path is not None:
        timings_path.parent.mkdir(parents=True, exist_ok=True)
        if resume_timings and timings_path.exists():
            lines = timings_path.read_text(encoding="utf-8").splitlines()
            expected_header = (
                "pass\tsequence_length\tsite_zero_based\tforward_seconds\t"
                "end_to_end_seconds"
            )
            if not lines or lines[0] != expected_header:
                raise RuntimeError(f"invalid timing file header: {timings_path}")
            initial_calls = len(lines) - 1
        else:
            timings_path.write_text(
                "pass\tsequence_length\tsite_zero_based\tforward_seconds\t"
                "end_to_end_seconds\n",
                encoding="utf-8",
            )
    if progress_total and initial_calls > progress_total:
        raise RuntimeError(
            f"timing file contains {initial_calls} calls but progress total is {progress_total}"
        )
    progress = None
    if progress_total:
        from tqdm.auto import tqdm

        progress = tqdm(
            total=progress_total,
            initial=initial_calls,
            desc=progress_description,
            unit="pass",
            dynamic_ncols=True,
        )
    model, tokenizer, aa_token_ids = _MODEL_CACHE[key]
    return ESM2MaskedSitePredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        aa_token_ids=aa_token_ids,
        torch=torch,
        timings_path=timings_path,
        progress=progress,
        calls=initial_calls,
    )
