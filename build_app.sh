#!/bin/bash

# 1. Clean previous builds
echo "Cleaning..."
rm -rf build dist

# 2. Run PyInstaller (Builds the code only)
echo "Building App..."
pyinstaller --clean --noconfirm diarize.spec

# 3. MANUALLY Copy Resources (The Fix)
BASE_DIR="dist/DiarizeApp.app/Contents/MacOS"

# --- A. Copy Ollama Binary ---
echo "Injecting Ollama binary..."
mkdir -p "$BASE_DIR/deps"
cp resources/ollama "$BASE_DIR/deps/ollama"
chmod +x "$BASE_DIR/deps/ollama"

# --- B. Copy Pyannote Models (New) ---
echo "Injecting Pyannote models..."
# Create the parent 'models' folder
mkdir -p "$BASE_DIR/models"
# Copy the 'pyannote' folder INTO 'models'
# This creates .../MacOS/models/pyannote/config.yaml
cp -r resources/pyannote "$BASE_DIR/models/"

# --- C. COPY FFMPEG (NEW) ---
echo "Injecting FFmpeg..."
# Create 'deps/ffmpeg' folder
mkdir -p "$BASE_DIR/deps/ffmpeg"
# Copy the binary
cp resources/ffmpeg/ffmpeg "$BASE_DIR/deps/ffmpeg/"
# Make it executable
chmod +x "$BASE_DIR/deps/ffmpeg/ffmpeg"

# 4. Verify-A
if [ -f "$BASE_DIR/deps/ffmpeg/ffmpeg" ]; then
    echo "SUCCESS: FFmpeg injected successfully."
else
    echo "ERROR: FFmpeg injection failed."
    exit 1
fi

# 4. Verify-B
if [ -f "$BASE_DIR/deps/ollama" ] && [ -f "$BASE_DIR/models/pyannote/config.yaml" ]; then
    echo "SUCCESS: All resources injected successfully."
else
    echo "ERROR: Resource injection failed."
    ls -R "$BASE_DIR"
    exit 1
fi


# 5. Fix Gatekeeper & Apply Permissions
echo "Applying permissions and signing..."

# Clear attributes
xattr -cr dist/DiarizeApp.app

# Sign the App with Entitlements
# We sign the main binary inside the app
codesign --force --deep --sign - --entitlements entitlements.plist dist/DiarizeApp.app

echo "Done! You can run the app now."