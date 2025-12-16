import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import psycopg

MODEL_EXTS = {
    ".pb": ("pb", "tensorflow"),
    ".pkl": ("pkl", "pytorch"),
    ".pt": ("pt", "pytorch"),
    ".pth": ("pth", "pytorch"),
    ".onnx": ("onnx", "onnx"),
    ".tflite": ("tflite", "tflite"),
    ".dlc": ("dlc", "snpe"),
}

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", ".mypy_cache", ".ruff_cache"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_identity(repo_root: Path, file_path: Path) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Returns: (project, version, variant_path, artifact_name, artifact_stem)
    repo_root/<project>/<version>/<variant_path...>/<artifact>
    """
    rel = file_path.relative_to(repo_root)
    parts = rel.parts
    if len(parts) < 3:
        # Need at least project/version/file
        return None

    project = parts[0]
    version = parts[1]
    artifact_name = parts[-1]
    artifact_stem = Path(artifact_name).stem
    variant_path = "/".join(parts[2:-1])  # may be ''
    return project, version, variant_path, artifact_name, artifact_stem


def should_skip_dir(dirpath: Path) -> bool:
    return any(part in IGNORED_DIRS for part in dirpath.parts)


UPSERT_SQL = """
INSERT INTO modelhub.models (
  source, owner_user_id,
  project, version,
  variant_path, artifact_name, artifact_stem,
  file_path, file_type, framework,
  size_bytes, sha256,
  meta
)
VALUES (
  %(source)s, %(owner_user_id)s,
  %(project)s, %(version)s,
  %(variant_path)s, %(artifact_name)s, %(artifact_stem)s,
  %(file_path)s, %(file_type)s, %(framework)s,
  %(size_bytes)s, %(sha256)s,
  %(meta)s::jsonb
)
ON CONFLICT ON CONSTRAINT uq_modelhub_models_identity
DO UPDATE SET
  file_path   = EXCLUDED.file_path,
  file_type   = EXCLUDED.file_type,
  framework   = EXCLUDED.framework,
  size_bytes  = EXCLUDED.size_bytes,
  sha256      = EXCLUDED.sha256,
  meta        = EXCLUDED.meta,
  updated_at  = now()
RETURNING model_id;
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="postgresql://postgres:postgres@localhost:5432/agentic_suite",
        help="Postgres connection string",
    )
    parser.add_argument(
        "--root",
        default="/data/modelhub/official/repo",
        help="Official repo root on disk",
    )
    parser.add_argument(
        "--source",
        default="official",
        choices=["official", "overlay"],
        help="Index source (official or overlay)",
    )
    parser.add_argument(
        "--owner",
        default="",
        help="owner_user_id (required for overlay; for official leave empty)",
    )
    parser.add_argument(
        "--store-absolute-paths",
        action="store_true",
        help="Store absolute file_path in DB (default: store path relative to root)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    if args.source == "overlay" and not args.owner:
        raise SystemExit("For --source overlay, you must pass --owner <user_id>")

    indexed = 0
    skipped = 0

    with psycopg.connect(args.db) as conn:
        with conn.cursor() as cur:
            for dirpath, dirnames, filenames in os.walk(root):
                dpath = Path(dirpath)
                if should_skip_dir(dpath):
                    dirnames[:] = []
                    continue

                for fn in filenames:
                    p = dpath / fn
                    ext = p.suffix.lower()
                    if ext not in MODEL_EXTS:
                        continue

                    ident = parse_identity(root, p)
                    if ident is None:
                        skipped += 1
                        continue

                    project, version, variant_path, artifact_name, artifact_stem = ident
                    file_type, framework = MODEL_EXTS[ext]
                    size_bytes = p.stat().st_size
                    sha = sha256_file(p)

                    file_path_db = str(p if args.store_absolute_paths else p.relative_to(root))

                    meta = "{}"  # keep as JSON string; add tags later if you want

                    cur.execute(
                        UPSERT_SQL,
                        {
                            "source": args.source,
                            "owner_user_id": args.owner or "",
                            "project": project,
                            "version": version,
                            "variant_path": variant_path,
                            "artifact_name": artifact_name,
                            "artifact_stem": artifact_stem,
                            "file_path": file_path_db,
                            "file_type": file_type,
                            "framework": framework,
                            "size_bytes": size_bytes,
                            "sha256": sha,
                            "meta": meta,
                        },
                    )
                    _model_id = cur.fetchone()[0]
                    indexed += 1

            conn.commit()

    print(f"Done. Indexed={indexed}, skipped_invalid_paths={skipped}")


if __name__ == "__main__":
    main()
