# models/esmc_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from tqdm import tqdm

from .base_model import BaseModel


Pooling = Literal["mean", "max", "cls"]
InputType = Literal["dna", "protein", "auto"]


class ESMCModel(BaseModel):
    """ESMC adapter for DNA-translated or direct protein embeddings."""

    _MODEL_ALIASES = {
        "300m": "esmc_300m",
        "esmc_300m": "esmc_300m",
        "esmc-300m": "esmc_300m",
        "esmc-300m-2024-12": "esmc_300m",
        "600m": "esmc_600m",
        "esmc_600m": "esmc_600m",
        "esmc-600m": "esmc_600m",
        "esmc-600m-2024-12": "esmc_600m",
        "6b": "esmc_6b",
        "esmc_6b": "esmc_6b",
        "esmc-6b": "esmc_6b",
        "esmc-6b-2025-03": "esmc_6b",
    }

    # ESMC-6B dims: d_model=2560, n_heads=40, n_layers=80
    _6B_DIMS = (2560, 40, 80)

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        translation_mode: Literal["first_orf", "fixed_frame"] = "first_orf",
        translation_frame: int = 0,
        stop_at_stop: bool = True,
        input_type: InputType = "dna",
        max_length: int = 2048,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.translation_mode = translation_mode
        self.translation_frame = int(translation_frame) % 3
        self.stop_at_stop = stop_at_stop
        self.input_type = input_type
        self.model_max_len = int(max_length)
        self.model = None
        self._logits_config = None
        self._esm_protein_cls = None
        self._codon_table = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
            "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
            "AGT": "S", "AGC": "S", "CCT": "P", "CCC": "P",
            "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T",
            "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
            "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y",
            "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
            "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "TGT": "C", "TGC": "C", "TGG": "W", "CGT": "R",
            "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R",
            "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G",
            "GGG": "G", "TAA": "*", "TAG": "*", "TGA": "*",
        }
        self._load_model()

    def _load_model(self):
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise ImportError(
                "[ESMCModel] The ESMC SDK is required. Install it with `pip install esm`."
            ) from exc
        try:
            from esm.pretrained import register_local_model
        except ImportError:
            register_local_model = None

        model_id = self._resolve_model_id(self.model_path, self.model_name)
        print(f"[ESMCModel] Loading '{model_id}' from {self.model_path} on {self.device} ...")

        if model_id == "esmc_6b":
            self.model = self._build_6b_model(self.model_path, ESMC, self.device)
        elif register_local_model is not None:
            self._register_local_weight_dir(model_id, self.model_path, register_local_model)
            self.model = ESMC.from_pretrained(model_id, device=torch.device(self.device)).eval()
        else:
            local_model = self._build_local_weight_model(model_id, self.model_path, ESMC, self.device)
            if local_model is None:
                self.model = ESMC.from_pretrained(model_id, device=torch.device(self.device)).eval()
            else:
                self.model = local_model

        self._esm_protein_cls = ESMProtein
        self._logits_config = LogitsConfig(sequence=True, return_embeddings=True)
        print(
            f"[ESMCModel] loaded '{self.model_name}' as {model_id}, "
            f"embedding_dim={self.embedding_dim}, max_len={self.model_max_len}"
        )

    @classmethod
    def _resolve_model_id(cls, model_path: str, model_name: str) -> str:
        for raw in (model_path, model_name):
            name = os.path.basename(str(raw).rstrip(os.sep)).lower().replace("_", "-")
            if name in cls._MODEL_ALIASES:
                return cls._MODEL_ALIASES[name]
            if "6b" in name:
                return "esmc_6b"
            if "300m" in name:
                return "esmc_300m"
            if "600m" in name:
                return "esmc_600m"
        raise ValueError("Cannot infer ESMC size; use a name/path containing 300m, 600m, or 6b.")

    @staticmethod
    def _remap_6b_state_dict(state_hf: dict) -> dict:
        """Remap HuggingFace ESMC-6B safetensors keys to ESM SDK ESMC key format."""
        state: dict = {}
        for k, v in state_hf.items():
            k = k.removeprefix("esmc.")
            if "_extra_state" in k:
                continue
            # layernorm_qkv: named attrs → Sequential positional indices
            k = re.sub(r"(attn\.layernorm_qkv)\.layer_norm_weight$", r"\1.0.weight", k)
            k = re.sub(r"(attn\.layernorm_qkv)\.layer_norm_bias$",   r"\1.0.bias",   k)
            k = re.sub(r"(attn\.layernorm_qkv)\.weight$",            r"\1.1.weight", k)
            # ffn: named attrs → Sequential positional indices
            k = re.sub(r"(\.ffn)\.layer_norm_weight$", r"\1.0.weight", k)
            k = re.sub(r"(\.ffn)\.layer_norm_bias$",   r"\1.0.bias",   k)
            k = re.sub(r"(\.ffn)\.fc1_weight$",        r"\1.1.weight", k)
            k = re.sub(r"(\.ffn)\.fc2_weight$",        r"\1.3.weight", k)
            # lm_head → sequence_head
            k = re.sub(r"^lm_head\.", "sequence_head.", k)
            state[k] = v
        return state

    @classmethod
    def _build_6b_model(cls, model_path: str, esmc_cls, device: Union[torch.device, str]):
        """Load ESMC-6B from HuggingFace safetensors shards into the ESM SDK ESMC skeleton."""
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("[ESMCModel] safetensors is required to load ESMC-6B.") from exc
        from esm.tokenization import get_esmc_model_tokenizers

        shards = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        if not shards:
            raise FileNotFoundError(
                f"[ESMCModel] No safetensors shards found in {model_path!r}. "
                "Expected model-*.safetensors files for ESMC-6B."
            )

        print(f"[ESMCModel] Loading ESMC-6B from {len(shards)} safetensors shards ...")
        state_hf: dict = {}
        for shard in shards:
            state_hf.update(load_file(shard, device="cpu"))

        torch_device = torch.device(device)
        d_model, n_heads, n_layers = cls._6B_DIMS
        with torch.device(torch_device):
            model = esmc_cls(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                tokenizer=get_esmc_model_tokenizers(),
            ).eval()

        state = cls._remap_6b_state_dict(state_hf)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ESMCModel] ESMC-6B: {len(missing)} missing keys (first 3): {missing[:3]}")
        if unexpected:
            print(f"[ESMCModel] ESMC-6B: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")

        model = model.to(torch_device)
        if torch_device.type != "cpu":
            model = model.to(torch.bfloat16)
        return model.eval()

    @staticmethod
    def _register_local_weight_dir(model_id: str, model_path: str, register_local_model) -> None:
        if ESMCModel._local_weight_path(model_id, model_path) is None:
            return

        def _builder(device: Union[torch.device, str] = "cpu"):
            from esm.models.esmc import ESMC

            model = ESMCModel._build_local_weight_model(model_id, model_path, ESMC, device)
            if model is None:
                raise FileNotFoundError(f"No local ESMC weights found under {model_path!r}")
            return model

        register_local_model(model_id, _builder)

    @staticmethod
    def _local_weight_path(model_id: str, model_path: str) -> Optional[str]:
        if not os.path.isdir(model_path):
            return None
        weight_names = {
            "esmc_300m": "esmc_300m_2024_12_v0.pth",
            "esmc_600m": "esmc_600m_2024_12_v0.pth",
        }
        weight_name = weight_names.get(model_id)
        if weight_name is None:
            return None
        weight_path = os.path.join(model_path, "data", "weights", weight_name)
        return weight_path if os.path.isfile(weight_path) else None

    @staticmethod
    def _build_local_weight_model(model_id: str, model_path: str, esmc_cls, device: Union[torch.device, str]):
        weight_path = ESMCModel._local_weight_path(model_id, model_path)
        if weight_path is None:
            return None
        from esm.tokenization import get_esmc_model_tokenizers

        dims = {"esmc_300m": (960, 15, 30), "esmc_600m": (1152, 18, 36)}
        d_model, n_heads, n_layers = dims[model_id]
        torch_device = torch.device(device)
        with torch.device(torch_device):
            model = esmc_cls(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                tokenizer=get_esmc_model_tokenizers(),
            ).eval()
            state_dict = torch.load(weight_path, map_location=torch_device)
            model.load_state_dict(state_dict)
        return model.to(torch_device).eval()

    @property
    def embedding_dim(self) -> int:
        if self.model is None:
            return 0
        return int(getattr(self.model.embed, "embedding_dim", 0) or 0)

    @staticmethod
    def _clean_dna(seq: str) -> str:
        s = seq.strip().upper().replace("U", "T")
        return "".join(ch for ch in s if ch in ("A", "C", "G", "T", "N"))

    @staticmethod
    def _revcomp(seq: str) -> str:
        tbl = str.maketrans(
            "ACGTRYMKBDHVNacgtrymkbdhvn",
            "TGCAYRKMVHDBNtgcayrkmvhdbn",
        )
        return seq.translate(tbl)[::-1]

    @staticmethod
    def _looks_like_dna(seq: str) -> bool:
        cleaned = seq.strip().upper().replace("U", "T")
        return bool(cleaned) and all(ch in "ACGTN" for ch in cleaned)

    @staticmethod
    def _clean_protein(seq: str) -> str:
        valid = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
        cleaned = "".join(ch if ch in valid else "X" for ch in seq.strip().upper().replace("*", ""))
        return cleaned or "X"

    def _translate_dna_to_protein(self, dna_seq: str) -> str:
        dna = self._clean_dna(dna_seq)
        if len(dna) < 3:
            return "X"
        dna = dna[dna.find("ATG") if self.translation_mode == "first_orf" and dna.find("ATG") >= 0 else 0:]
        if self.translation_mode == "fixed_frame":
            dna = dna[self.translation_frame:]

        aa_list: List[str] = []
        for i in range(0, len(dna) - 2, 3):
            aa = self._codon_table.get(dna[i: i + 3], "X")
            if aa == "*":
                if self.stop_at_stop:
                    break
                aa = "X"
            aa_list.append(aa)
        return "".join(aa_list) or "X"

    def _prepare_protein(self, seq: str) -> str:
        if self.input_type == "protein":
            aa = self._clean_protein(seq)
        elif self.input_type == "auto":
            aa = self._translate_dna_to_protein(seq) if self._looks_like_dna(seq) else self._clean_protein(seq)
        else:
            aa = self._translate_dna_to_protein(seq)
        return aa[: self.model_max_len] if self.model_max_len > 0 else aa

    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        del batch_size
        scores: List[float] = []
        with torch.no_grad():
            for seq in tqdm(sequences, desc="Scoring sequences (ESMC)"):
                scores.append(self._score_single_protein(self._prepare_protein(seq)))
        return scores

    def _score_single_protein(self, aa_sequence: str) -> float:
        encoded = self.model.encode(self._esm_protein_cls(sequence=aa_sequence))
        output = self.model.logits(encoded, self._logits_config)
        logits = output.logits.sequence
        token_ids = encoded.sequence
        if logits is None or token_ids is None or token_ids.numel() <= 2:
            return 0.0
        if logits.ndim == 3:
            logits = logits[0]
        log_probs = torch.log_softmax(logits[1:-1], dim=-1)
        targets = token_ids[1:-1].to(log_probs.device)
        positions = torch.arange(targets.numel(), device=log_probs.device)
        return float(log_probs[positions, targets].mean().item())

    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 16,
        pooling: Pooling = "mean",
        average_reverse_complement: bool = False,
        truncation: bool = True,
        return_numpy: bool = True,
        input_type: Optional[InputType] = None,
    ):
        return self.get_embedding(
            sequences=sequences,
            batch_size=batch_size,
            pool=pooling,
            average_reverse_complement=average_reverse_complement,
            truncation=truncation,
            return_numpy=return_numpy,
            input_type=input_type,
        )

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,
        batch_size: int = 16,
        pool: Pooling = "mean",
        layer_index: int = -1,
        average_reverse_complement: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
        input_type: Optional[InputType] = None,
        return_tokens: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        del layer_name, layer_index, truncation
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
        old_input_type, old_max_len = self.input_type, self.model_max_len
        if input_type is not None:
            self.input_type = input_type
        if max_length is not None:
            self.model_max_len = int(max_length)
        try:
            outputs: List[torch.Tensor] = []
            for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting ESMC embedding"):
                batch = seq_list[st: st + batch_size]
                if return_tokens:
                    if average_reverse_complement:
                        raise ValueError("return_tokens=True does not support average_reverse_complement.")
                    token_batch = [
                        self._embed_single_tokens(self._prepare_protein(s)).detach().cpu()
                        for s in batch
                    ]
                    outputs.extend(token_batch)
                    continue
                vec_f = torch.stack([self._embed_single(self._prepare_protein(s), pool) for s in batch], dim=0)
                if average_reverse_complement:
                    if self.input_type == "protein":
                        raise ValueError("average_reverse_complement is only meaningful for DNA input.")
                    vec_r = torch.stack(
                        [self._embed_single(self._prepare_protein(self._revcomp(s)), pool) for s in batch],
                        dim=0,
                    )
                    vec_f = 0.5 * (vec_f + vec_r)
                outputs.append(vec_f.detach().cpu())
            if return_tokens:
                if not outputs:
                    return [] if return_numpy else []
                if return_numpy:
                    return [item.numpy() for item in outputs]
                return outputs
            out = torch.cat(outputs, dim=0) if outputs else torch.empty(0, self.embedding_dim)
            return out.numpy() if return_numpy else out
        finally:
            self.input_type, self.model_max_len = old_input_type, old_max_len

    def _embed_single_tokens(self, aa_sequence: str) -> torch.Tensor:
        encoded = self.model.encode(self._esm_protein_cls(sequence=aa_sequence))
        output = self.model.logits(encoded, self._logits_config)
        token_reps = output.embeddings
        if token_reps is None:
            raise RuntimeError("[ESMCModel] ESMC did not return embeddings.")
        if token_reps.ndim == 3:
            token_reps = token_reps[0]
        reps = token_reps[1:-1] if token_reps.size(0) > 2 else token_reps
        return reps.float() if reps.numel() else token_reps.float()

    def _embed_single(self, aa_sequence: str, pool: Pooling) -> torch.Tensor:
        encoded = self.model.encode(self._esm_protein_cls(sequence=aa_sequence))
        output = self.model.logits(encoded, self._logits_config)
        token_reps = output.embeddings
        if token_reps is None:
            raise RuntimeError("[ESMCModel] ESMC did not return embeddings.")
        if token_reps.ndim == 3:
            token_reps = token_reps[0]
        reps = token_reps[1:-1] if token_reps.size(0) > 2 else token_reps
        if reps.numel() == 0:
            reps = token_reps
        if pool == "cls":
            return token_reps[0].float()
        if pool == "mean":
            return reps.float().mean(dim=0)
        if pool == "max":
            return reps.float().max(dim=0).values
        raise ValueError(f"Unknown pool type: {pool}")


ESM2Model = ESMCModel


if __name__ == "__main__":
    model = ESMCModel(
        model_name="ESMC-300M",
        model_path="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/esmc-300m-2024-12",
        input_type="auto",
    )
    embs = model.embed_sequences(["AAAAA", "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"], input_type="auto")
    print("Embedding shape:", embs.shape)
