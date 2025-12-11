#!/bin/bash

# 1. Clean previous builds
echo "Cleaning..."
rm -rf build dist

# 2. Run PyInstaller
echo "Building App..."
pyinstaller --clean --noconfirm diarize.spec

# 3. MANUALLY Copy Ollama (The Fix)
# We create the folder ourselves and copy the file
echo "Forcing copy of Ollama binary..."
mkdir -p dist/DiarizeApp.app/Contents/MacOS/deps
cp resources/ollama dist/DiarizeApp.app/Contents/MacOS/deps/ollama
chmod +x dist/DiarizeApp.app/Contents/MacOS/deps/ollama

# 4. Verify
if [ -f "dist/DiarizeApp.app/Contents/MacOS/deps/ollama" ]; then
    echo "SUCCESS: Ollama binary injected successfully."
else
    echo "ERROR: Failed to inject Ollama binary."
    exit 1
fi

# 5. Fix Gatekeeper
echo "Fixing Gatekeeper..."
xattr -cr dist/DiarizeApp.app

echo "Done! You can run the app now."