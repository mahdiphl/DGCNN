
import argparse
import logging
import random
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError as exc:
    raise ImportError(
        "huggingface_hub is required. Install it with `pip install huggingface_hub`."
    ) from exc

LOGGER = logging.getLogger("get_donut")
REPO_ID = "LouisM2001/donut"
METADATA_PATTERNS = ["*.csv", "*.md", "*.txt", "*.json"]

# member suffix (after "<key>.") -> destination subfolder
PCD_KIND = "pcd.npy"
OBJ_KINDS = {"vertices.npy", "faces.npy"}
JSON_KIND = "json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the DONUT WebDataset shards from Hugging Face and "
        "extract them into flat pcd/ (and obj/) directories."
    )
    p.add_argument("destination", type=Path)
    p.add_argument("--revision", default="main")
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    g = p.add_argument_group("subset selection")
    g.add_argument("--splits", nargs="+", default=["train", "test"],
                   choices=["train", "test"], help="Which splits' shards to use.")
    g.add_argument("--num_shards", type=int, default=None,
                   help="Number of shards to download (after --splits filtering). "
                        "This is what saves bandwidth.")
    g.add_argument("--pcd_fraction", type=float, default=None,
                   help="Fraction (0-1] of samples to keep from the downloaded shards.")
    g.add_argument("--pcd_count", type=int, default=None,
                   help="Exact number of samples to keep (overrides --pcd_fraction).")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--first", action="store_true",
                   help="Take the first shards / samples (sorted) instead of random ones.")
    g.add_argument("--skip_obj", action="store_true",
                   help="Do not extract vertices/faces mesh files.")
    g.add_argument("--skip_json", action="store_true",
                   help="Do not extract per-sample json files.")
    return p.parse_args()


def validate_args(a: argparse.Namespace) -> None:
    if a.pcd_fraction is not None and not 0 < a.pcd_fraction <= 1:
        raise ValueError("--pcd_fraction must be in (0, 1].")
    if a.pcd_count is not None and a.pcd_count < 1:
        raise ValueError("--pcd_count must be >= 1.")
    if a.num_shards is not None and a.num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")


def split_member(name: str) -> tuple[str, str]:
    """'<key>.pcd.npy' -> ('<key>', 'pcd.npy')."""
    base = PurePosixPath(name).name
    key, _, kind = base.partition(".")
    return key, kind


def choose_shards(a: argparse.Namespace) -> list[str]:
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=a.revision)
    shards = sorted(
        f for f in files if f.endswith(".tar") and f.split("/")[0] in set(a.splits)
    )
    if not shards:
        raise RuntimeError(f"No .tar shards found for splits {a.splits}.")
    if a.num_shards is not None and a.num_shards < len(shards):
        shards = shards[: a.num_shards] if a.first else sorted(
            random.Random(a.seed).sample(shards, a.num_shards)
        )
    LOGGER.info("Using %d shard(s)", len(shards))
    return shards


def list_keys(tar_path: Path) -> list[str]:
    keys = set()
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if m.isfile():
                key, kind = split_member(m.name)
                if kind == PCD_KIND:
                    keys.add(key)
    return sorted(keys)


def extract(tar_path: Path, dest: Path, keep: set[str] | None, a) -> dict[str, int]:
    counts = {"pcd": 0, "obj": 0, "json": 0}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, kind = split_member(m.name)
            if keep is not None and key not in keep:
                continue
            if kind == PCD_KIND:
                sub = "pcd"
            elif kind in OBJ_KINDS and not a.skip_obj:
                sub = "obj"
            elif kind == JSON_KIND and not a.skip_json:
                sub = "json"
            else:
                continue
            target = dest / sub / PurePosixPath(m.name).name
            if target.exists() and not a.overwrite:
                raise FileExistsError(f"{target} exists. Use --overwrite to replace it.")
            target.write_bytes(tf.extractfile(m).read())
            counts[sub] += 1
    return counts


def main() -> int:
    a = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    validate_args(a)

    dest = a.destination.expanduser().resolve()
    for sub in ("pcd", "obj", "json"):
        if sub == "obj" and a.skip_obj or sub == "json" and a.skip_json:
            continue
        (dest / sub).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading metadata (splits, csv, README)")
    snapshot_download(REPO_ID, repo_type="dataset", revision=a.revision,
                      local_dir=dest, allow_patterns=METADATA_PATTERNS,
                      force_download=a.force_download)

    shards = choose_shards(a)

    with TemporaryDirectory(prefix="donut_shards_", dir=dest.parent) as tmp:
        tmp = Path(tmp)
        local = []
        for s in shards:
            LOGGER.info("Downloading %s", s)
            local.append(Path(hf_hub_download(
                REPO_ID, s, repo_type="dataset", revision=a.revision,
                local_dir=tmp, force_download=a.force_download)))

        keep = None
        if a.pcd_count is not None or a.pcd_fraction is not None:
            all_keys = sorted({k for p in local for k in list_keys(p)})
            n = (min(a.pcd_count, len(all_keys)) if a.pcd_count is not None
                 else max(1, round(len(all_keys) * a.pcd_fraction)))
            chosen = all_keys[:n] if a.first else random.Random(a.seed).sample(all_keys, n)
            keep = set(chosen)
            LOGGER.info("Keeping %d of %d samples", n, len(all_keys))

        total = {"pcd": 0, "obj": 0, "json": 0}
        for p in local:
            LOGGER.info("Extracting %s", p.name)
            for k, v in extract(p, dest, keep, a).items():
                total[k] += v
            p.unlink()  # free disk as we go (useful on Kaggle)

    LOGGER.info("Done: %d pcd, %d mesh files, %d json files",
                total["pcd"], total["obj"], total["json"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("Failed to download DONUT: %s", exc)
        raise
