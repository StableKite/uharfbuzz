"""Execute setup's ABI selection without requiring build tools in wheel tests."""
import ast
import os
from pathlib import Path
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
    tree = ast.parse(script.read_text(encoding="utf-8"))
    statements = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "bool_from_environ")
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "use_py_limited_api"
            for target in node.targets
        ))
    ]
    namespace = {"os": os, "sysconfig": sysconfig}
    with patch.dict(os.environ, environment, clear=True), patch.object(
        sysconfig, "get_config_var", config
    ):
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(script), "exec"), namespace)
    assert namespace["use_py_limited_api"] is expected
