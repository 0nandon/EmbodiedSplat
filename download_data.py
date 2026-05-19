#!/usr/bin/env python3
"""Download sharded dataset archives from Hugging Face and restore the pretrained and dataset folders."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


os.environ["HF_ENDPOINT"] = "https://huggingface.co"


TARGET_DIRS = ("pretrained", "dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download .tar.zst shard archives and shard manifest JSON files from "
            "Hugging Face into a cache folder, extract them, and restore the "
            "'pretrained' and 'dataset' folders under the current path."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="onandon/EmbodiedSplat",
        help="Hugging Face dataset repo id. Default: onandon/EmbodiedSplat.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Where shard archives and manifests are downloaded. Default: ./cache.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Where 'pretrained' and 'dataset' folders will be created. Default: current directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Not required for a public repo.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep the cache folder (downloaded shards and manifests) after extraction.",
    )
    return parser.parse_args()


def download_shards(repo_id: str, cache_dir: Path, token: str | None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.tar.zst",
            "*.json",
            "**/*.tar.zst",
            "**/*.json",
        ],
        token=token,
    )
    return cache_dir


def find_manifests(cache_dir: Path) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for path in sorted(cache_dir.rglob("*-shard-manifest.json")):
        name = path.name[: -len("-shard-manifest.json")]
        if name in TARGET_DIRS:
            manifests[name] = path
    return manifests


def archives_for_manifest(cache_dir: Path, manifest_path: Path) -> list[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    archives: list[Path] = []
    for row in payload.get("shards", []):
        archive_name = row.get("archive")
        if not archive_name:
            continue
        matches = list(cache_dir.rglob(archive_name))
        if not matches:
            raise FileNotFoundError(
                f"archive listed in manifest was not downloaded: {archive_name}"
            )
        archives.append(matches[0])
    return archives


def fallback_archives(cache_dir: Path, target: str) -> list[Path]:
    prefix = f"{target}-shard-"
    return sorted(
        path
        for path in cache_dir.rglob(f"{prefix}*.tar.zst")
        if path.is_file()
    )


def extract_archive(archive_path: Path, output_dir: Path, index: int, total: int) -> None:
    if not archive_path.name.endswith(".tar.zst"):
        raise ValueError(f"unsupported archive type: {archive_path}")
    cmd = ["tar", "--zstd", "-xf", str(archive_path), "-C", str(output_dir)]
    print(f"[{index}/{total}] extracting {archive_path.name} -> {output_dir}")
    subprocess.run(cmd, check=True)


def restore_dataset(
    repo_id: str,
    cache_dir: Path,
    output_root: Path,
    token: str | None,
    keep_cache: bool,
) -> None:
    if shutil.which("tar") is None:
        raise SystemExit("tar is required but was not found in PATH")

    cache_dir = download_shards(repo_id=repo_id, cache_dir=cache_dir, token=token)
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = find_manifests(cache_dir)

    total_extracted = 0
    for target in TARGET_DIRS:
        target_dir = output_root / target
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = manifests.get(target)
        if manifest_path is not None:
            archives = archives_for_manifest(cache_dir, manifest_path)
        else:
            archives = fallback_archives(cache_dir, target)

        if not archives:
            print(f"warning: no shards found for '{target}'")
            continue

        for index, archive_path in enumerate(archives, start=1):
            extract_archive(
                archive_path=archive_path,
                output_dir=target_dir,
                index=index,
                total=len(archives),
            )
            total_extracted += 1

    if total_extracted == 0:
        raise SystemExit(f"no shard archives were extracted from {cache_dir}")

    if not keep_cache:
        print(f"removing cache folder {cache_dir}")
        shutil.rmtree(cache_dir)

    print(f"restored 'pretrained' and 'dataset' under {output_root}")


if __name__ == "__main__":
    args = parse_args()
    restore_dataset(
        repo_id=args.repo_id,
        cache_dir=args.cache_dir.resolve(),
        output_root=args.output_root.resolve(),
        token=args.token,
        keep_cache=args.keep_cache,
    )
