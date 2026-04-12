import sys
from unittest.mock import MagicMock

import pytest

import run_tests


def test_run_command_returns_exit_code(monkeypatch):
    fake_result = MagicMock(returncode=42)
    monkeypatch.setattr("run_tests.subprocess.run", MagicMock(return_value=fake_result))

    exit_code = run_tests.run_command(["echo", "hello"])

    assert exit_code == 42
    run_tests.subprocess.run.assert_called_once_with(["echo", "hello"])


def test_main_with_unit_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "--unit"])
    monkeypatch.setattr("run_tests.run_command", lambda cmd: 0)

    with pytest.raises(SystemExit) as excinfo:
        run_tests.main()

    assert excinfo.value.code == 0
