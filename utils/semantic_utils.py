import gc
import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SemanticAnnotator:
    """Offline LLM-driven semantic annotation for HELMS memory prototypes.

    Paper alignment:
      cluster raw windows -> structured statistics prompt -> lightweight local LLM
      generates tag/description -> Sentence-BERT encodes the description for SR.

    The class is deliberately lazy: the local LLM is loaded only during
    ``annotate_clusters`` and is released afterwards unless keep_llm_loaded=True.
    This avoids keeping Qwen on the GPU during HELMS training.
    """

    DEFAULT_SENTENCE_MODEL_PATH = "/data/wanganna/ICDE27/all-MiniLM-L6-v2/"
    DEFAULT_LLM_MODEL_PATH = "/data/wanganna/ICDE27/qwen2.5-1.5b-instruct"

    def __init__(
        self,
        semantic_dim: int = 384,
        sentence_model_path: Optional[str] = None,
        device: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        use_llm: bool = True,
        llm_device: str = "auto",
        llm_max_new_tokens: int = 96,
        llm_temperature: float = 0.2,
        keep_llm_loaded: bool = False,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        self.semantic_dim = int(semantic_dim)
        self.sentence_model_path = sentence_model_path or self.DEFAULT_SENTENCE_MODEL_PATH
        self.device = device
        self.llm_model_path = llm_model_path or self.DEFAULT_LLM_MODEL_PATH
        self.use_llm = bool(use_llm)
        self.llm_device = str(llm_device or "auto")
        self.llm_max_new_tokens = int(llm_max_new_tokens)
        self.llm_temperature = float(llm_temperature)
        self.keep_llm_loaded = bool(keep_llm_loaded)
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name or "dataset"

        self.model = None
        self.model_dim = None
        self.llm_tokenizer = None
        self.llm_model = None

        if self.sentence_model_path:
            self._try_load_sentence_model(self.sentence_model_path)

    # ------------------------------------------------------------------
    # Sentence-BERT embedding
    # ------------------------------------------------------------------
    def _try_load_sentence_model(self, path: str) -> None:
        path = os.path.abspath(os.path.expanduser(path))
        has_weight = (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "pytorch_model.bin"))
        )
        required = ["config.json", "tokenizer_config.json"]
        missing = [f for f in required if not os.path.exists(os.path.join(path, f))]
        if missing or not has_weight:
            print(
                "[WARN] Local Sentence-BERT directory may be incomplete. "
                f"path={path}, missing={missing}, has_weight={has_weight}. "
                "Use deterministic offline text encoder instead."
            )
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(path, device=self.device)
            try:
                dim = self.model.get_sentence_embedding_dimension()
            except Exception:
                dim = None
            self.model_dim = int(dim) if dim is not None else None
            print(
                f"[INFO] Loaded local Sentence-BERT from {path}. "
                f"model_dim={self.model_dim}, target_semantic_dim={self.semantic_dim}"
            )
        except Exception as e:
            self.model = None
            print(f"[WARN] Failed to load local Sentence-BERT from {path}: {e}. Use deterministic offline text encoder.")

    def _hash_embed(self, text: str) -> torch.Tensor:
        vec = np.zeros(self.semantic_dim, dtype=np.float32)
        tokens = text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.semantic_dim
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        if np.linalg.norm(vec) < 1e-6:
            vec[0] = 1.0
        return F.normalize(torch.tensor(vec, dtype=torch.float32), dim=0)

    @staticmethod
    def _deterministic_projection(in_dim: int, out_dim: int) -> torch.Tensor:
        rng = np.random.RandomState(2026)
        proj = rng.normal(0, 1.0 / np.sqrt(max(1, in_dim)), (in_dim, out_dim)).astype(np.float32)
        return torch.tensor(proj, dtype=torch.float32)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        if self.model is not None:
            emb = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            emb = torch.tensor(emb, dtype=torch.float32)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            if emb.shape[-1] != self.semantic_dim:
                proj = self._deterministic_projection(emb.shape[-1], self.semantic_dim)
                emb = F.normalize(emb @ proj, dim=-1)
            else:
                emb = F.normalize(emb, dim=-1)
            return emb
        return torch.stack([self._hash_embed(t) for t in texts], dim=0)

    # ------------------------------------------------------------------
    # LLM loading/generation
    # ------------------------------------------------------------------
    def _llm_path_ok(self) -> bool:
        path = os.path.abspath(os.path.expanduser(self.llm_model_path))
        if not os.path.isdir(path):
            print(f"[WARN] LLM path does not exist: {path}. Use rule-based semantic tags.")
            return False
        required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        missing = [f for f in required if not os.path.exists(os.path.join(path, f))]
        has_weight = (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "pytorch_model.bin"))
            or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
        )
        if missing or not has_weight:
            print(
                f"[WARN] LLM directory is incomplete: {path}, missing={missing}, has_weight={has_weight}. "
                "Use rule-based semantic tags. Note: Qwen2.5-1.5B normally has a single model.safetensors."
            )
            return False
        return True

    def _load_llm(self) -> bool:
        if self.llm_model is not None and self.llm_tokenizer is not None:
            return True
        if not self.use_llm or not self._llm_path_ok():
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            path = os.path.abspath(os.path.expanduser(self.llm_model_path))
            self.llm_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            kwargs = dict(local_files_only=True, trust_remote_code=True, torch_dtype="auto")
            if self.llm_device == "auto":
                kwargs["device_map"] = "auto"
            self.llm_model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
            if self.llm_device not in ("auto", "none", "None"):
                self.llm_model.to(torch.device(self.llm_device))
            self.llm_model.eval()
            print(f"[INFO] Loaded local LLM from {path} for offline HELMS semantic annotation.")
            return True
        except Exception as e:
            self.llm_tokenizer = None
            self.llm_model = None
            print(f"[WARN] Failed to load local LLM from {self.llm_model_path}: {e}. Use rule-based semantic tags.")
            return False

    def _unload_llm(self):
        if self.keep_llm_loaded:
            return
        self.llm_tokenizer = None
        self.llm_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_with_llm(self, prompt: str) -> Optional[str]:
        if not self._load_llm():
            return None
        try:
            tok = self.llm_tokenizer
            model = self.llm_model
            messages = [
                {"role": "system", "content": "You are a traffic forecasting expert. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ]
            if hasattr(tok, "apply_chat_template"):
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            inputs = tok([text], return_tensors="pt")
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            do_sample = self.llm_temperature > 1e-6
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.llm_max_new_tokens,
                    do_sample=do_sample,
                    temperature=max(self.llm_temperature, 1e-6),
                    top_p=0.9,
                    pad_token_id=tok.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            return tok.decode(gen, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[WARN] LLM generation failed: {e}. Use rule-based semantic tag for this prototype.")
            return None

    # ------------------------------------------------------------------
    # Prompt/statistics/cache
    # ------------------------------------------------------------------
    @staticmethod
    def _stats_from_sequences(seqs: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(seqs)
        if arr.ndim == 4:
            arr = arr[..., 0]
        if arr.ndim == 2:
            arr = arr[:, :, None]
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))
        if arr.ndim == 3 and arr.shape[1] > 0:
            temporal_curve = np.mean(arr, axis=(0, 2))
            peak_pos = int(np.argmax(temporal_curve))
            trough_pos = int(np.argmin(temporal_curve))
            slope = float(np.mean(arr[:, -1, :] - arr[:, 0, :])) if arr.shape[1] > 1 else 0.0
            volatility = float(np.mean(np.abs(np.diff(temporal_curve)))) if arr.shape[1] > 1 else 0.0
            early_mean = float(np.mean(temporal_curve[: max(1, len(temporal_curve)//3)]))
            late_mean = float(np.mean(temporal_curve[-max(1, len(temporal_curve)//3):]))
        else:
            peak_pos, trough_pos, slope, volatility, early_mean, late_mean = 0, 0, 0.0, 0.0, mean_val, mean_val
        return {
            "mean": mean_val, "std": std_val, "max": max_val, "min": min_val,
            "peak_step": peak_pos, "trough_step": trough_pos, "slope": slope,
            "volatility": volatility, "early_mean": early_mean, "late_mean": late_mean,
        }

    @staticmethod
    def _rule_tag_from_stats(stats: Dict[str, float]) -> Tuple[str, str]:
        mean_val = stats["mean"]; std_val = stats["std"]; slope = stats["slope"]
        volatility = stats["volatility"]; peak_pos = int(stats["peak_step"])
        if mean_val > 0.80 and slope > 0.08:
            tag = "growing heavy congestion"
        elif mean_val > 0.80 and slope < -0.08:
            tag = "congestion dissipation"
        elif mean_val > 0.80:
            tag = "persistent high-flow congestion"
        elif mean_val < -0.60 and std_val < 0.90:
            tag = "stable off-peak low-flow"
        elif std_val > 1.35 or volatility > 0.30:
            tag = "incident-induced volatile traffic"
        elif slope > 0.12:
            tag = "rising transition before peak"
        elif slope < -0.12:
            tag = "falling transition after peak"
        elif peak_pos <= 4:
            tag = "early-window traffic peak"
        elif peak_pos >= 8:
            tag = "late-window traffic peak"
        else:
            tag = "regular recurrent traffic pattern"
        desc = (
            f"Traffic memory prototype: {tag}. The cluster has normalized mean {mean_val:.4f}, "
            f"standard deviation {std_val:.4f}, maximum {stats['max']:.4f}, minimum {stats['min']:.4f}, "
            f"peak step {peak_pos}, trough step {int(stats['trough_step'])}, temporal trend {slope:.4f}, "
            f"and volatility {volatility:.4f}. It represents a recurring spatio-temporal traffic pattern "
            "used for memory retrieval, concept-drift adaptation, and semantic regularization."
        )
        return tag, desc

    @staticmethod
    def _build_prompt(stats: Dict[str, float], cluster_id: int) -> str:
        stat_text = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()])
        return (
            "Given the following statistics of one traffic-flow memory prototype, generate one concise semantic tag "
            "and one short human-readable description for traffic prediction. Use traffic concepts such as morning peak, "
            "evening congestion, off-peak low flow, incident spike, congestion dissipation, holiday/event pattern, "
            "or recurrent stable flow when appropriate.\n"
            f"Prototype id: {cluster_id}\nStatistics: {stat_text}\n"
            "Return only JSON with exactly two fields: {\"tag\": \"...\", \"description\": \"...\"}."
        )

    @staticmethod
    def _parse_llm_json(text: Optional[str]) -> Optional[Tuple[str, str]]:
        if not text:
            return None
        m = re.search(r"\{.*\}", text, flags=re.S)
        cand = m.group(0) if m else text
        try:
            obj = json.loads(cand)
            tag = str(obj.get("tag", "")).strip()
            desc = str(obj.get("description", "")).strip()
            if tag and desc:
                return tag[:120], desc[:800]
        except Exception:
            pass
        # Very small fallback parser in case the model emits lines instead of JSON.
        tag_match = re.search(r"tag\s*[:：]\s*([^\n]+)", text, flags=re.I)
        desc_match = re.search(r"description\s*[:：]\s*([^\n]+)", text, flags=re.I)
        if tag_match and desc_match:
            return tag_match.group(1).strip().strip('"'), desc_match.group(1).strip().strip('"')
        return None

    def _cache_path(self, k: int) -> Optional[str]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        safe_ds = re.sub(r"[^A-Za-z0-9_\-]+", "_", self.dataset_name)
        mode = "llm" if self.use_llm else "rule"
        return os.path.join(self.cache_dir, f"semantic_{mode}_{safe_ds}_K{k}.json")

    def _load_cache(self, k: int) -> Optional[Tuple[List[str], List[str]]]:
        path = self._cache_path(k)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            entries = obj.get("entries", [])
            if len(entries) != k:
                return None
            tags = [str(e.get("tag", "regular recurrent traffic pattern")) for e in entries]
            descs = [str(e.get("description", tags[i])) for i, e in enumerate(entries)]
            print(f"[INFO] Loaded cached LLM semantic annotations: {path}")
            return tags, descs
        except Exception as e:
            print(f"[WARN] Failed to read semantic cache: {e}")
            return None

    def _save_cache(self, tags: List[str], descs: List[str], stats_list: List[Dict[str, float]]):
        path = self._cache_path(len(tags))
        if not path:
            return
        try:
            entries = [
                {"id": i, "tag": tags[i], "description": descs[i], "statistics": stats_list[i]}
                for i in range(len(tags))
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"dataset": self.dataset_name, "entries": entries}, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved LLM semantic annotations: {path}")
        except Exception as e:
            print(f"[WARN] Failed to save semantic cache: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def annotate_clusters(self, raw_windows: np.ndarray, labels: torch.Tensor, k: int):
        cached = self._load_cache(k)
        if cached is not None:
            tags, descs = cached
            return tags, descs, self.encode_texts(descs)

        labels_np = labels.detach().cpu().numpy()
        tags, descs, stats_list = [], [], []
        llm_ready = self._load_llm() if self.use_llm else False
        for i in range(k):
            idx = np.where(labels_np == i)[0]
            if len(idx) == 0:
                seqs = raw_windows[:1]
            else:
                idx = idx[: min(128, len(idx))]
                seqs = raw_windows[idx]
            stats = self._stats_from_sequences(seqs)
            stats_list.append(stats)
            fallback_tag, fallback_desc = self._rule_tag_from_stats(stats)
            tag, desc = fallback_tag, fallback_desc
            if llm_ready:
                text = self._generate_with_llm(self._build_prompt(stats, i))
                parsed = self._parse_llm_json(text)
                if parsed is not None:
                    tag, desc = parsed
            tags.append(tag)
            descs.append(desc)
        self._unload_llm()
        self._save_cache(tags, descs, stats_list)
        sem = self.encode_texts(descs)
        return tags, descs, sem

    def default_new_memory_semantic(self) -> Tuple[str, str, torch.Tensor]:
        tag = "new unseen traffic pattern"
        desc = (
            "Online-created memory for an unseen traffic pattern under concept drift. "
            "The prototype is initialized from the current query state and connected to nearby memory prototypes."
        )
        sem = self.encode_texts([desc])[0]
        return tag, desc, sem
