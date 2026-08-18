"""Tests for the layering rules the README talks about.

The README claims the package is strictly layered and that numpy is kept at the
bottom. That kind of claim goes stale the moment someone adds a convenient
import, so these check it instead of trusting it.

They parse each module's top-level imports with ast. Imports inside a function
body are exempt on purpose, since Function.apply imports Tensor at call time
to break a real circular dependency, and that's the sanctioned way out.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import teenygrad

PACKAGE_ROOT = pathlib.Path(teenygrad.__file__).parent

# A module can only import from a layer at or below its own number. data sits at
# 0 because it's outside the engine entirely. It depends on numpy and one type
# alias and nothing else.
LAYERS = {
    "core": 0,
    "data": 0,
    "autograd": 1,
    "tensor": 2,
    "nn": 3,
    "optim": 3,
    "training": 4,
}


def source_files() -> list[pathlib.Path]:
    """Every module except the top-level __init__, whose whole job is to import
    from every layer and present one flat API."""
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.relative_to(PACKAGE_ROOT).as_posix() != "__init__.py"
    )


def layer_of(path: pathlib.Path) -> int:
    """Work out which layer a file is in from its path."""
    return LAYERS[path.relative_to(PACKAGE_ROOT).parts[0].removesuffix(".py")]


def top_level_imports(path: pathlib.Path) -> set[str]:
    """Modules imported at import time.

    Skips imports nested inside functions (only walks tree.body, not the whole
    tree) and `if TYPE_CHECKING:` blocks, since neither is a real runtime
    dependency.
    """
    tree = ast.parse(path.read_text())
    imported: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                continue
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    return imported


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_modules_only_import_from_their_own_layer_or_below(path):
    """No module reaches upward.

    A failure here means the dependency graph grew an edge that makes the
    diagram in the README a lie. Core reaching up into tensor, say, which would
    create an import cycle and kill the idea that the backend is swappable.
    """
    importer_layer = layer_of(path)

    for module in top_level_imports(path):
        if not module.startswith("teenygrad"):
            continue
        parts = module.split(".")
        if len(parts) < 2:
            continue  # plain `import teenygrad`
        imported_layer = LAYERS[parts[1]]

        assert imported_layer <= importer_layer, (
            f"{path.relative_to(PACKAGE_ROOT)} (layer {importer_layer}) imports "
            f"{module} (layer {imported_layer}), which is above it"
        )


@pytest.mark.parametrize(
    "path",
    [p for p in source_files() if p.parent.name == "autograd"],
    ids=lambda p: p.name,
)
def test_the_autograd_layer_never_touches_numpy(path):
    """Every derivative is written using only LazyBuffer ops.

    This is what makes the backend swappable. Porting to a GPU should mean
    rewriting core/lazybuffer.py and nothing else. A stray np.something in a
    backward pass would quietly break that.
    """
    assert "numpy" not in top_level_imports(path)


def test_numpy_stays_out_of_the_differentiable_path():
    """Pin down exactly which files are allowed to import numpy at all.

    tensor, nn and training do import it, but only for random init, turning
    integer labels into one-hot floats, and array interop, never arithmetic on
    something that needs a gradient. Listing them explicitly means a new numpy
    shortcut in a forward pass has to be a deliberate change, not an accident.
    """
    users = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in source_files()
        if "numpy" in top_level_imports(path)
    }
    assert users == {
        "core/lazybuffer.py",  # the backend, runs every primitive
        "core/types.py",  # type aliases only
        "data/datasets.py",  # outside the engine
        "data/metrics.py",  # outside the engine
        "nn/layers.py",  # weight init
        "nn/losses.py",  # integer labels -> one-hot
        "tensor.py",  # randn, .numpy(), __array__
        "training.py",  # input coercion
    }


def test_every_module_has_a_docstring():
    """Each file should at least say what it's for."""
    for path in source_files():
        assert ast.get_docstring(ast.parse(path.read_text())), f"{path} has no docstring"
