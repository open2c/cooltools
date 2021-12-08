import importlib
import pathlib

__all__ = [
    f.stem
    for f in pathlib.Path(__file__).parent.glob("*.py")
    if f.is_file() and not f.name == "__init__.py"
]

for _ in __all__:
    importlib.import_module("." + _, "cooltools.api")

del pathlib
del importlib
