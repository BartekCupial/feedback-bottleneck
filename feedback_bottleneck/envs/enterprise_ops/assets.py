import os
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

ASSET_DIR_NAME = "Domain Wise DBs and Task-DB Mappings"
DEFAULT_OUTPUT_DIR = "~/.cache/enterprise_ops"
DEFAULT_ARCHIVE_NAME = "gym_dbs.zip"
DEFAULT_ARCHIVE_URL = "https://raw.githubusercontent.com/ServiceNow/EnterpriseOps-Gym/main/gym_dbs.zip"
PREPARE_LOCK_NAME = ".prepare_enterprise_ops_assets.lock"


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _asset_root(seed_db_root: Path) -> Path:
    return seed_db_root / ASSET_DIR_NAME


def _selected_domains(seed_db_root: Path, domains: list[str] | tuple[str, ...] | None) -> list[str]:
    if domains:
        return list(domains)
    return sorted(path.name for path in _asset_root(seed_db_root).iterdir() if path.is_dir())


def _has_seed_dbs(seed_db_root: Path, domains: list[str] | tuple[str, ...] | None) -> bool:
    asset_root = _asset_root(seed_db_root)
    if not asset_root.exists():
        return False

    selected_domains = _selected_domains(seed_db_root, domains)
    return bool(selected_domains) and all(
        any((asset_root / domain / "dbs").glob("*.sql")) for domain in selected_domains
    )


def validate_seed_db_root(
    seed_db_root: str | Path,
    domains: list[str] | tuple[str, ...] | None = None,
) -> dict[str, int]:
    seed_db_root = expand_path(seed_db_root)
    asset_root = _asset_root(seed_db_root)
    if not asset_root.exists():
        raise FileNotFoundError(f"EnterpriseOps asset directory does not exist: {asset_root}")

    selected_domains = _selected_domains(seed_db_root, domains)
    if not selected_domains:
        raise ValueError(f"No EnterpriseOps domains found in {asset_root}")

    counts = {}
    for domain in selected_domains:
        db_dir = asset_root / domain / "dbs"
        if not db_dir.exists():
            raise FileNotFoundError(f"EnterpriseOps DB directory does not exist for domain '{domain}': {db_dir}")

        seed_files = sorted(db_dir.glob("*.sql"))
        if not seed_files:
            raise ValueError(f"No EnterpriseOps SQL seed files found in {db_dir}")
        counts[domain] = len(seed_files)

    return counts


@contextmanager
def prepare_lock(seed_db_root: str | Path):
    seed_db_root = expand_path(seed_db_root)
    seed_db_root.mkdir(parents=True, exist_ok=True)
    lock_path = seed_db_root / PREPARE_LOCK_NAME
    deadline = time.monotonic() + 600

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for EnterpriseOps asset lock: {lock_path}")
            time.sleep(1)

    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def download_archive(seed_db_root: str | Path) -> Path:
    seed_db_root = expand_path(seed_db_root)
    seed_db_root.mkdir(parents=True, exist_ok=True)
    archive_path = seed_db_root / DEFAULT_ARCHIVE_NAME
    temp_path = seed_db_root / f".{DEFAULT_ARCHIVE_NAME}.tmp"
    urllib.request.urlretrieve(DEFAULT_ARCHIVE_URL, temp_path)
    temp_path.replace(archive_path)
    return archive_path


def extract_archive(archive_path: str | Path, seed_db_root: str | Path) -> None:
    with zipfile.ZipFile(expand_path(archive_path)) as archive:
        archive.extractall(expand_path(seed_db_root))


def ensure_seed_db_root(
    seed_db_root: str | Path,
    domains: list[str] | tuple[str, ...] | None = None,
) -> dict[str, int]:
    seed_db_root = expand_path(seed_db_root)
    if not _has_seed_dbs(seed_db_root, domains):
        with prepare_lock(seed_db_root):
            if not _has_seed_dbs(seed_db_root, domains):
                archive_path = download_archive(seed_db_root)
                extract_archive(archive_path, seed_db_root)

    return validate_seed_db_root(seed_db_root, domains=domains)
