"""Inspect actual setup metadata without recompiling extensions in each case."""
import os
from pathlib import Path
import runpy
import sysconfig
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "free_threaded,requested,expected",
    [(False, None, True), (True, None, False), (False, "0", False),
     (True, "0", False), (False, "1", True)],
)
def test_native_abi_selection(free_threaded, requested, expected):
    script = next(
        parent / "setup.py"
        for parent in Path(__file__).resolve().parents
        if (parent / "setup.py").is_file()
    )
    original = sysconfig.get_config_var

    def config(name):
        return int(free_threaded) if name == "Py_GIL_DISABLED" else original(name)

    environment = dict(os.environ, USE_SYSTEM_LIBS="1")
    environment.pop("USE_PY_LIMITED_API", None)
    if requested is not None:
        environment["USE_PY_LIMITED_API"] = requested
    with patch.dict(os.environ, environment, clear=True), patch.object(
        sysconfig, "get_config_var", config
    ), patch("Cython.Build.cythonize", side_effect=lambda extensions, **kw: extensions), patch(
        "setuptools.setup"
    ) as setup:
        runpy.run_path(str(script), run_name="__main__")
    metadata = setup.call_args.kwargs
    assert metadata["ext_modules"]
    for extension in metadata["ext_modules"]:
        assert extension.py_limited_api is expected
        assert ("Py_LIMITED_API" in dict(extension.define_macros)) is expected
    assert ("bdist_wheel" in metadata["options"]) is expected
