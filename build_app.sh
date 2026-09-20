#!/bin/bash
set -euo pipefail
export COPYFILE_DISABLE=1

if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: build_app.sh creates the macOS application and must run on macOS."
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
APP_NAME="Language Learning Mobile Assistant"
BUNDLE_PATH="dist/${APP_NAME}.app"
DMG_PATH="dist/Language-Learning-Mobile-Assistant-macOS.dmg"
if ! "$PYTHON_BIN" -c 'import PyInstaller' >/dev/null 2>&1; then
    echo "ERROR: PyInstaller is missing from $PYTHON_BIN. Activate the build environment and install pyinstaller."
    exit 1
fi

for required in resources/pyannote/config.yaml resources/icons/diarize.icns resources/icons/diarize_logo.png; do
    if [ ! -e "$required" ]; then
        echo "ERROR: Missing required build resource: $required"
        exit 1
    fi
done

# 1. Clean previous builds
echo "Cleaning..."
rm -rf build dist

# 2. Run PyInstaller (Builds the code only)
echo "Building App..."
"$PYTHON_BIN" -m PyInstaller --clean --noconfirm diarize.spec

# 3. MANUALLY Copy Resources (The Fix)
BASE_DIR="$BUNDLE_PATH/Contents/MacOS"

# --- A. Copy Ollama Binary ---
echo "Injecting Ollama binary..."
mkdir -p "$BASE_DIR/deps"
OLLAMA_RUNTIME_SOURCE="${OLLAMA_RUNTIME_SOURCE:-/Applications/Ollama.app/Contents/Resources}"
OLLAMA_BIN_SOURCE="${OLLAMA_BIN_SOURCE:-$OLLAMA_RUNTIME_SOURCE/ollama}"
if [ ! -x "$OLLAMA_BIN_SOURCE" ]; then
    echo "ERROR: Missing current Ollama executable at $OLLAMA_BIN_SOURCE"
    echo "Install the current Ollama macOS app before building the installer."
    exit 1
fi
cp "$OLLAMA_BIN_SOURCE" "$BASE_DIR/deps/ollama"
chmod +x "$BASE_DIR/deps/ollama"

# Ollama's standalone executable is not sufficient for MLX-native models. Copy
# the runtime shipped by the official macOS app so Gemma MLX works on machines
# that do not already have Ollama installed separately.
if [ ! -d "$OLLAMA_RUNTIME_SOURCE/mlx_metal_v4" ]; then
    echo "ERROR: Missing Ollama MLX runtime at $OLLAMA_RUNTIME_SOURCE/mlx_metal_v4"
    echo "Install the current Ollama macOS app before building the installer."
    exit 1
fi
echo "Injecting Ollama MLX runtime..."
cp -R "$OLLAMA_RUNTIME_SOURCE/mlx_metal_v4" "$BASE_DIR/deps/"
if [ -d "$OLLAMA_RUNTIME_SOURCE/mlx_metal_v3" ]; then
    cp -R "$OLLAMA_RUNTIME_SOURCE/mlx_metal_v3" "$BASE_DIR/deps/"
fi
find "$OLLAMA_RUNTIME_SOURCE" -maxdepth 1 -type f -name '*.dylib' -exec cp {} "$BASE_DIR/deps/" \;
for license in MLX_LICENSE MLX_C_LICENSE; do
    if [ -f "$OLLAMA_RUNTIME_SOURCE/$license" ]; then
        cp "$OLLAMA_RUNTIME_SOURCE/$license" "$BASE_DIR/deps/"
    fi
done

# --- A2. Copy FluidAudio Apple-native diarization helper when available ---
if [ -f resources/fluidaudiocli ]; then
    echo "Injecting FluidAudio diarization helper..."
    cp resources/fluidaudiocli "$BASE_DIR/deps/fluidaudiocli"
    chmod +x "$BASE_DIR/deps/fluidaudiocli"
fi

# --- B. Copy Pyannote Models (New) ---
echo "Injecting Pyannote models..."
# Create the parent 'models' folder
mkdir -p "$BASE_DIR/models"
# Copy the 'pyannote' folder INTO 'models'
# This creates .../MacOS/models/pyannote/config.yaml
cp -r resources/pyannote "$BASE_DIR/models/"

# --- C. COPY FFMPEG ---
echo "Injecting FFmpeg..."
mkdir -p "$BASE_DIR/deps/ffmpeg"
FFMPEG_SOURCE="resources/ffmpeg/ffmpeg"
HOST_ARCH="$(uname -m)"
if [ ! -x "$FFMPEG_SOURCE" ] || ! file "$FFMPEG_SOURCE" | grep -q "$HOST_ARCH"; then
    FFMPEG_SOURCE="$(command -v ffmpeg || true)"
fi
if [ -z "$FFMPEG_SOURCE" ] || [ ! -x "$FFMPEG_SOURCE" ] || ! file "$FFMPEG_SOURCE" | grep -q "$HOST_ARCH"; then
    echo "ERROR: A native $HOST_ARCH ffmpeg binary is required. Install it with: brew install ffmpeg"
    exit 1
fi
cp "$FFMPEG_SOURCE" "$BASE_DIR/deps/ffmpeg/ffmpeg"
chmod +x "$BASE_DIR/deps/ffmpeg/ffmpeg"

# --- D. COPY UI ASSETS ---
echo "Injecting UI assets..."
mkdir -p "$BASE_DIR/resources"
cp -r resources/icons "$BASE_DIR/resources/"

# NEW: Remove quarantine from the copied binary inside the app
xattr -cr "$BASE_DIR/deps/ffmpeg/ffmpeg" 2>/dev/null || true

# 4. Verify-A
if [ -f "$BASE_DIR/deps/ffmpeg/ffmpeg" ]; then
    echo "SUCCESS: FFmpeg injected successfully."
else
    echo "ERROR: FFmpeg injection failed."
    exit 1
fi

# 4. Verify-B
if [ -f "$BASE_DIR/deps/ollama" ] \
    && [ -f "$BASE_DIR/deps/mlx_metal_v4/libmlx.dylib" ] \
    && [ -f "$BASE_DIR/models/pyannote/config.yaml" ] \
    && [ -f "$BASE_DIR/resources/icons/diarize_logo.png" ] \
    && [ -f "$BASE_DIR/resources/icons/language_learning_mobile_assistant_lockup.png" ]; then
    echo "SUCCESS: All resources injected successfully."
else
    echo "ERROR: Resource injection failed."
    ls -R "$BASE_DIR"
    exit 1
fi


# 5. Fix Gatekeeper & Apply Permissions
echo "Applying permissions and signing..."

# Clear attributes
xattr -cr "$BUNDLE_PATH" 2>/dev/null || echo "WARNING: Some external-drive metadata could not be cleared."

# Sign the App with Entitlements
# We sign the main binary inside the app
codesign --force --deep --sign - --entitlements entitlements.plist "$BUNDLE_PATH"

echo "Creating drag-to-install disk image..."
hdiutil create -volname "$APP_NAME" -srcfolder "$BUNDLE_PATH" -ov -format UDZO "$DMG_PATH"

echo "Done: $BUNDLE_PATH and $DMG_PATH"
