"""The embedding model: its ONNX export, downloaded once into ~/.llmdex/models, run with ONNX Runtime."""

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from llm_index.registry import EMBED_MODEL_NAME

MODELS_DIR = Path.home() / ".llmdex" / "models"
# Where each file may sit in a model's repository, tried in order. model.onnx comes down last, so once it is on disk the rest is too.
SOURCES = {
    "tokenizer.json": ("onnx/tokenizer.json", "tokenizer.json"),
    "pooling.json": ("1_Pooling/config.json",),
    "model.onnx_data": ("onnx/model.onnx_data",),
    "model.onnx": ("onnx/model.onnx",),
}
# Pinned, so every copy of llmdex embeds with the same weights and indexes built on one Mac match queries on another.
REVISIONS = {"intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3"}
MAX_TOKENS = 512
# Batches of similar length pad less: sorted batches of 8 embed 1.8x faster than batches of 32 in file order.
BATCH = 8


def model_dir() -> Path:
    return MODELS_DIR / EMBED_MODEL_NAME.replace("/", "--")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def _describe(url: str) -> tuple[int, str] | None:
    """Size and sha256 of a file, or None when the repository has no such file. Hugging Face sends both for a large file before redirecting to its CDN, and a small one is sized after the redirect."""
    try:
        headers = urllib.request.build_opener(_NoRedirect).open(urllib.request.Request(url, method="HEAD"), timeout=30).headers
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        if e.code not in (301, 302, 303, 307, 308):
            raise
        headers = e.headers
    if "x-linked-size" in headers:
        return int(headers["x-linked-size"]), headers.get("x-linked-etag", "").strip('"')
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
        return int(response.headers["content-length"]), ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def download_model(log=print) -> Path:
    """Fetch the model files that are missing, resuming a partial download, and log progress as "Downloading the model: 212 MB of 487 MB"."""
    out = model_dir()
    if (out / "model.onnx").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{EMBED_MODEL_NAME}/resolve/{REVISIONS.get(EMBED_MODEL_NAME, 'main')}/"
    urls, sizes = {}, {}
    for name, paths in SOURCES.items():
        if (out / name).exists():
            continue
        for source in paths:
            if described := _describe(base + source):
                urls[name], sizes[name] = base + source, described
                break
    for name in ("tokenizer.json", "model.onnx"):
        if name not in urls and not (out / name).exists():
            raise RuntimeError(f"{EMBED_MODEL_NAME} has no {SOURCES[name][0]}, so ONNX Runtime cannot run it")
    missing = list(urls)
    total = sum(size for size, _ in sizes.values())
    finished = 0
    shown = -1

    def report(done: int):
        nonlocal shown
        percent = done * 100 // total
        if percent != shown:
            shown = percent
            log(f"Downloading the model: {done // 1_000_000} MB of {total // 1_000_000} MB")

    for name in missing:
        size, sha = sizes[name]
        part = out / f"{name}.part"
        for _ in range(3):
            have = part.stat().st_size if part.exists() else 0
            if have < size:
                request = urllib.request.Request(urls[name], headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        while block := response.read(1 << 20):
                            f.write(block)
                            have += len(block)
                            report(finished + have)
                if have < size:
                    raise ConnectionError(f"Connection closed after {have // 1_000_000} MB of {size // 1_000_000} MB of {name}")
            if not sha or _sha256(part) == sha:
                break
            part.unlink()
        else:
            raise RuntimeError(f"The model's {name} came down damaged three times")
        os.replace(part, out / name)
        finished += size
    return out


_shared = None


def shared_embedder(log=print) -> "Embedder":
    """One Embedder per process: a split project indexes each subfolder with it, and loading takes about a second."""
    global _shared
    if _shared is None:
        _shared = Embedder(log)
    return _shared


class Embedder:
    """Pooled and normalized sentence vectors, the same as sentence-transformers computes for the model."""

    def __init__(self, log=print):
        import onnxruntime
        from tokenizers import Tokenizer

        path = download_model(log)
        self.counter = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        if not self.tokenizer.padding:
            self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>") or 0)
        self.session = onnxruntime.InferenceSession(str(path / "model.onnx"), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        pooling = path / "pooling.json"
        self.cls = pooling.exists() and json.loads(pooling.read_text()).get("pooling_mode_cls_token", False)
        # e5 models are trained with these prefixes and retrieve noticeably worse without them.
        e5 = "e5" in EMBED_MODEL_NAME.lower()
        self.query_prefix, self.text_prefix = ("query: ", "passage: ") if e5 else ("", "")

    def _run(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        pooled = hidden[:, 0] if self.cls else (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1e-9)
        return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        rows: list[np.ndarray] = [None] * len(texts)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            for i, vector in zip(batch, self._run([self.text_prefix + texts[i] for i in batch])):
                rows[i] = vector
        return np.stack(rows)

    def embed_query(self, text: str) -> np.ndarray:
        return self._run([self.query_prefix + text])[0]

    def token_starts(self, text: str) -> list[int]:
        """Where each of the model's tokens starts in text, so a chunk can be sized in the tokens the model reads."""
        return [start for start, _ in self.counter.encode(text, add_special_tokens=False).offsets]
