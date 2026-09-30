
import argparse
import logging
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as exc:
    raise ImportError(
        "huggingface_hub is required to download DONUT. "
        "Install it with `pip install huggingface_hub`."
    ) from exc


LOGGER = logging.getLogger("get_donut")
REPO_ID = "LouisM2001/donut"
METADATA_PATTERNS = ["*.csv", "*.md", "*.txt", "*.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the DONUT dataset (or a subset of it) from Hugging Face "
            "and flatten shard directories."
        ),
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Destination directory where the dataset should be materialized.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset revision to download (branch, tag, or commit hash). Default: main.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if files are already cached by huggingface_hub.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files in the destination directory.",
    )

    subset = parser.add_argument_group("subset selection")
    subset.add_argument(
        "--pcd_fraction",
        type=float,
        default=None,
        help="Fraction (0-1] of pcd/*.npy files to download, e.g. 0.05 for 5%%.",
    )
    subset.add_argument(
        "--pcd_count",
        type=int,
        default=None,
        help="Exact number of pcd/*.npy files to download (overrides --pcd_fraction).",
    )
    subset.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed so the same subset is selected each run. Default: 0.",
    )
    subset.add_argument(
        "--first",
        action="store_true",
        help="Take the first N pcd files (sorted by path) instead of a random sample.",
    )
    subset.add_argument(
        "--skip_obj",
        action="store_true",
        help="Do not download any obj/*.npz files.",
    )
    subset.add_argument(
        "--match_obj",
        action="store_true",
        help=(
            "Only download obj/*.npz files whose file stem matches a selected "
            "pcd/*.npy file (assumes obj and pcd share names)."
        ),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_destination(path: Path, skip_obj: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not skip_obj:
        (path / "obj").mkdir(exist_ok=True)
    (path / "pcd").mkdir(exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.pcd_fraction is not None and not 0 < args.pcd_fraction <= 1:
        raise ValueError("--pcd_fraction must be in the range (0, 1].")
    if args.pcd_count is not None and args.pcd_count < 1:
        raise ValueError("--pcd_count must be >= 1.")
    if args.skip_obj and args.match_obj:
        raise ValueError("--skip_obj and --match_obj cannot be used together.")


def select_files(
    revision: str,
    fraction: float | None,
    count: int | None,
    seed: int,
    first: bool,
    skip_obj: bool,
    match_obj: bool,
) -> tuple[list[str] | None, list[str] | None]:
    """Return (pcd_files, obj_files).

    A value of None means "no subsetting requested; download everything
    of that kind using a glob pattern".
    """
    subset_requested = fraction is not None or count is not None
    if not subset_requested and not match_obj:
        return None, None

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=revision)

    pcd_files = sorted(
        f for f in all_files if f.startswith("pcd/") and f.endswith(".npy")
    )
    if not pcd_files:
        raise RuntimeError("No pcd/*.npy files found in the repository.")

    if count is not None:
        k = min(count, len(pcd_files))
    elif fraction is not None:
        k = max(1, round(len(pcd_files) * fraction))
    else:
        k = len(pcd_files)

    if first:
        selected_pcd = pcd_files[:k]
    else:
        selected_pcd = sorted(random.Random(seed).sample(pcd_files, k))
    LOGGER.info("Selected %d of %d pcd files", len(selected_pcd), len(pcd_files))

    selected_obj: list[str] | None = None
    if match_obj:
        stems = {Path(f).stem for f in selected_pcd}
        selected_obj = sorted(
            f
            for f in all_files
            if f.startswith("obj/") and f.endswith(".npz") and Path(f).stem in stems
        )
        LOGGER.info("Matched %d obj files to the selected pcd files", len(selected_obj))
        if not selected_obj:
            LOGGER.warning(
                "No obj files matched the selected pcd stems; check the naming scheme."
            )

    return selected_pcd, selected_obj


def build_allow_patterns(
    pcd_files: list[str] | None,
    obj_files: list[str] | None,
    skip_obj: bool,
) -> list[str]:
    patterns = list(METADATA_PATTERNS)

    if pcd_files is None:
        patterns.append("pcd/**")
    else:
        patterns.extend(pcd_files)

    if not skip_obj:
        if obj_files is None:
            patterns.append("obj/**")
        else:
            patterns.extend(obj_files)

    return patterns


def copy_root_metadata(staging_dir: Path, destination: Path, overwrite: bool) -> None:
    LOGGER.info("Copying repository metadata files")
    for item in staging_dir.iterdir():
        if item.name in {"obj", "pcd", ".cache"}:
            continue

        target = destination / item.name
        if item.is_dir():
            if target.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Refusing to overwrite existing directory: {target}. "
                        "Use --overwrite to replace it."
                    )
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {target}. "
                    "Use --overwrite to replace it."
                )
            shutil.copy2(item, target)


def move_files(
    flat_source_dir: Path, pattern: str, flat_target_dir: Path, overwrite: bool
) -> int:
    moved = 0
    for source_path in sorted(flat_source_dir.rglob(pattern)):
        target_path = flat_target_dir / source_path.name
        if target_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {target_path}. "
                    "Use --overwrite to replace it."
                )
            target_path.unlink()

        shutil.move(str(source_path), str(target_path))
        moved += 1
    return moved


def remove_empty_shard_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def flatten_shards(
    staging_dir: Path, destination: Path, overwrite: bool, skip_obj: bool
) -> None:
    LOGGER.info("Flattening shard directories into %s", destination)

    obj_source_dir = staging_dir / "obj"
    pcd_source_dir = staging_dir / "pcd"
    obj_target_dir = destination / "obj"
    pcd_target_dir = destination / "pcd"

    if not skip_obj and obj_source_dir.exists():
        obj_target_dir.mkdir(exist_ok=True)
        obj_count = move_files(
            obj_source_dir, "*.npz", obj_target_dir, overwrite=overwrite
        )
        LOGGER.info("Moved %s object files into %s", obj_count, obj_target_dir)
        remove_empty_shard_dirs(obj_source_dir)

    if pcd_source_dir.exists():
        pcd_count = move_files(
            pcd_source_dir, "*.npy", pcd_target_dir, overwrite=overwrite
        )
        LOGGER.info("Moved %s point-cloud files into %s", pcd_count, pcd_target_dir)
        remove_empty_shard_dirs(pcd_source_dir)


def main() -> int:
    args = parse_args()
    configure_logging()
    validate_args(args)

    destination = args.destination.expanduser().resolve()
    ensure_destination(destination, skip_obj=args.skip_obj)

    LOGGER.info("Preparing to download dataset %s into %s", REPO_ID, destination)

    pcd_files, obj_files = select_files(
        revision=args.revision,
        fraction=args.pcd_fraction,
        count=args.pcd_count,
        seed=args.seed,
        first=args.first,
        skip_obj=args.skip_obj,
        match_obj=args.match_obj,
    )
    allow_patterns = build_allow_patterns(pcd_files, obj_files, args.skip_obj)
    LOGGER.info("Using %d allow_patterns", len(allow_patterns))

    with TemporaryDirectory(
        prefix="donut_download_", dir=destination.parent
    ) as tmp_dir:
        staging_dir = Path(tmp_dir) / "snapshot"
        LOGGER.info(
            "Downloading dataset snapshot to temporary staging directory %s",
            staging_dir,
        )
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=args.revision,
            local_dir=staging_dir,
            allow_patterns=allow_patterns,
            force_download=args.force_download,
            max_workers=1,
        )

        copy_root_metadata(staging_dir, destination, overwrite=args.overwrite)
        flatten_shards(
            staging_dir, destination, overwrite=args.overwrite, skip_obj=args.skip_obj
        )

    LOGGER.info("DONUT dataset download and unsharding complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("Failed to download DONUT: %s", exc)
        raise
