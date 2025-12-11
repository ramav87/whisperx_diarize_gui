# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

# --- 1. VERIFY PATHS (Stop the build if files are missing) ---
project_dir = os.getcwd()
ollama_src = os.path.join(project_dir, 'resources', 'ollama')
pyannote_src = os.path.join(project_dir, 'resources', 'pyannote')

if not os.path.exists(ollama_src):
    raise FileNotFoundError(f"\n\nCRITICAL ERROR: Could not find Ollama binary at:\n{ollama_src}\n\nMake sure the file exists and is named exactly 'ollama' (no extension, not a folder).")

if not os.path.exists(pyannote_src):
    raise FileNotFoundError(f"\n\nCRITICAL ERROR: Could not find Pyannote folder at:\n{pyannote_src}")

print(f"--- FOUND OLLAMA AT: {ollama_src} ---")
# -------------------------------------------------------------

datas = []
binaries = []
hiddenimports = []

# Collect dependencies
tmp_ret = collect_all('whisperx'); datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyannote.audio'); datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchaudio'); datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lightning_fabric'); datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pytorch_lightning'); datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('speechbrain'); datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

hiddenimports += [
    'scipy.special.cython_special', 
    'sklearn.utils._typedefs', 
    'sklearn.neighbors._partition_nodes',
    'fsspec.implementations.memory'
]

# --- 2. ADD RESOURCES (Using Absolute Paths) ---
# We put Ollama in a folder named 'deps'. 
# PyInstaller logic: ('source_path', 'dest_folder_name')
# This creates: Contents/MacOS/deps/ollama
# We will handle resource copying in build_app.sh manually
datas += []
# -----------------------------------------------

block_cipher = None

a = Analysis(
    ['app_launcher.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DiarizeApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DiarizeApp',
)

app = BUNDLE(
    coll,
    name='DiarizeApp.app',
    icon=None,
    bundle_identifier='com.rama.diarizegui',
)