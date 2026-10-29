"""Patch the pinned pycolmap (used by gsplat's COLMAP loader) for NumPy 2 and Windows.

- NumPy 2 rejects np.uint64(-1); replace it with the max uint64 value.
- On Windows 'L' is 4 bytes, so the binary readers misread COLMAP files; force 8-byte little-endian.
Safe to run more than once.
"""

import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("pycolmap")
if spec is None:
    raise SystemExit("pycolmap is not installed")
path = Path(spec.origin).parent / "scene_manager.py"
text = path.read_text()
patched = (
    text.replace("np.uint64(-1)", "np.iinfo(np.uint64).max")
    .replace("struct.unpack('IiLL', f.read(24))", "struct.unpack('<IiQQ', f.read(24))")
    .replace("struct.unpack('L', f.read(8))", "struct.unpack('<Q', f.read(8))")
)
if patched != text:
    path.write_text(patched)
    print(f"Patched {path}")
else:
    print("pycolmap already patched")
