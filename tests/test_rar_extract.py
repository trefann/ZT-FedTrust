"""Tests for the 7z/unrar detection-and-extraction utility (mocked — no real archives)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from data.rar_extract import RarToolNotFoundError, extract_rar, find_rar_tool


def test_find_rar_tool_returns_first_match_on_path():
    with patch("shutil.which", side_effect=lambda name: "/usr/bin/7z" if name == "7z" else None):
        assert find_rar_tool() == "/usr/bin/7z"


def test_find_rar_tool_returns_none_when_nothing_found():
    with patch("shutil.which", return_value=None), patch("pathlib.Path.exists", return_value=False):
        assert find_rar_tool() is None


def test_extract_rar_raises_clear_error_when_no_tool_found(tmp_path):
    with patch("data.rar_extract.find_rar_tool", return_value=None):
        with pytest.raises(RarToolNotFoundError, match="7-Zip"):
            extract_rar(tmp_path / "archive.rar", tmp_path / "out")


def test_extract_rar_invokes_7z_with_expected_arguments(tmp_path):
    archive = tmp_path / "archive.rar"
    archive.write_bytes(b"fake")
    dest = tmp_path / "out"

    with patch("data.rar_extract.find_rar_tool", return_value="/usr/bin/7z"), patch(
        "subprocess.run"
    ) as mock_run:
        extract_rar(archive, dest)

    mock_run.assert_called_once()
    command = mock_run.call_args[0][0]
    assert command[0] == "/usr/bin/7z"
    assert str(archive) in command
    assert dest.exists()


def test_extract_rar_invokes_unrar_with_expected_arguments(tmp_path):
    archive = tmp_path / "archive.rar"
    archive.write_bytes(b"fake")
    dest = tmp_path / "out"

    with patch("data.rar_extract.find_rar_tool", return_value="/usr/bin/unrar"), patch(
        "subprocess.run"
    ) as mock_run:
        extract_rar(archive, dest)

    command = mock_run.call_args[0][0]
    assert command[0] == "/usr/bin/unrar"
    assert command[1] == "x"
