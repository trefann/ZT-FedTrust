"""Detects and shells out to a 7z/unrar binary to extract .rar archives.

No Python RAR library is used on purpose — `rarfile` still requires an
external `unrar`/`bsdtar` binary to do the actual decompression, so it adds
a dependency without removing the real requirement: some extraction tool
must be installed on the machine. N-BaIoT's Mirai/Gafgyt attack traffic
ships as `.rar` (not `.zip`) inside its outer zip archive.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

_CANDIDATES = ("7z", "7za", "unrar", r"C:\Program Files\7-Zip\7z.exe")


class RarToolNotFoundError(RuntimeError):
    pass


def find_rar_tool() -> Optional[str]:
    """Return a path to a usable 7z/unrar binary, or None if none is found."""
    for candidate in _CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate
    return None


def extract_rar(archive_path: Path, dest_dir: Path) -> None:
    """Extract `archive_path` into `dest_dir`, creating it if needed."""
    tool = find_rar_tool()
    if tool is None:
        raise RarToolNotFoundError(
            "No RAR extraction tool found (looked for 7z/7za/unrar on PATH). "
            "Install 7-Zip (e.g. `winget install 7zip.7zip`) and retry."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    tool_name = Path(tool).stem.lower()
    if "unrar" in tool_name:
        command = [tool, "x", "-y", str(archive_path), str(dest_dir) + "/"]
    else:
        command = [tool, "x", str(archive_path), f"-o{dest_dir}", "-y"]
    subprocess.run(command, check=True, capture_output=True)
