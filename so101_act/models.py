"""Download a pinned public model and verify its size and SHA-256 manifest."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

from .checks import validate_checkpoint
from .config import ConfigError, PROJECT_ROOT, read_json

DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "pretrained.json"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "pretrained_model" / "so101-act-pick-and-place"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ConfigError("Invalid model manifest")
    for filename, expected in manifest["files"].items():
        # This release is intentionally flat. Reject path traversal on all OSes.
        if filename in {".", ".."} or any(c in filename for c in "/\\:"):
            raise ConfigError(f"Invalid model filename: {filename}")
        if not isinstance(expected, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(expected.get("sha256", ""))):
            raise ConfigError(f"Invalid checksum for {filename}")
        if type(expected.get("size")) is not int or expected["size"] <= 0:
            raise ConfigError(f"Invalid size for {filename}")
    return manifest


def verify_model(folder: Path, manifest: dict) -> None:
    for filename, expected in manifest["files"].items():
        path = folder / filename
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise ConfigError(f"Missing or incorrect file size: {path}")
        if sha256(path) != expected["sha256"]:
            raise ConfigError(f"SHA-256 mismatch: {path}")
    validate_checkpoint(folder)


def download_model(folder: Path, manifest: dict) -> Path:
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ConfigError("Model publication has no pinned Hub commit yet; see docs/model.md")
    if folder.exists():
        verify_model(folder, manifest)
        return folder
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ConfigError("Install huggingface-hub in your active Python to download the model") from exc
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=folder.name + ".partial-", dir=folder.parent))
    try:
        snapshot_download(
            repo_id=manifest["repo_id"], repo_type="model", revision=revision,
            local_dir=temporary, allow_patterns=list(manifest["files"]),
            token=False,  # Public inference bundle; downloading needs no login.
        )
        verify_model(temporary, manifest)
        temporary.rename(folder)
    except Exception as exc:
        raise ConfigError(f"Download/verification failed; partial files retained in {temporary}: {exc}") from exc
    return folder
