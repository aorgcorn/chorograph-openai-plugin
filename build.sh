#!/bin/bash
# Builds the WASM plugin, assembles and publishes a .bundle.zip to GitHub Releases,
# then prints the SHA256 of the file GitHub actually serves.
#
# Usage:
#   ./build.sh          — build WASM + bundle, upload as the version in version.txt, print SHA256
#
# After running, paste the printed SHA256 into registry.json.
set -e

NAME=ChorographOpenAIPlugin
BUNDLE_NAME="${NAME}.bundle"
WASM_CRATE=chorograph-openai-plugin-rust
WASM_OUT=chorograph_openai_plugin_rust.wasm
REPO=aorgcorn/chorograph-openai-plugin
BUILD_DIR=".build/release"

VERSION=$(cat version.txt)
TAG="v${VERSION}"

# ── Build WASM ─────────────────────────────────────────────────────────────────
echo "Building WASM (wasm32-unknown-unknown)..."
cargo build --release --target wasm32-unknown-unknown
WASM_SRC="target/wasm32-unknown-unknown/release/${WASM_CRATE//-/_}.wasm"
cp "${WASM_SRC}" "${WASM_OUT}"
echo "WASM built: ${WASM_OUT} ($(du -sh ${WASM_OUT} | cut -f1))"

# ── Build Swift bundle ─────────────────────────────────────────────────────────
echo "Building Swift bundle ${BUNDLE_NAME}..."
swift build -c release

rm -rf "${BUNDLE_NAME}"
mkdir -p "${BUNDLE_NAME}/Contents/MacOS"

cp "${BUILD_DIR}/lib${NAME}.dylib" "${BUNDLE_NAME}/Contents/MacOS/${NAME}"

cat > "${BUNDLE_NAME}/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>ChorographOpenAIPlugin</string>
    <key>CFBundleIdentifier</key>
    <string>com.aorgcorn.chorograph.plugin.openai-compatible</string>
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
</dict>
</plist>
PLIST

echo "Packaging ${BUNDLE_NAME}.zip..."
rm -f "${BUNDLE_NAME}.zip"
zip -r "${BUNDLE_NAME}.zip" "${BUNDLE_NAME}"
rm -rf "${BUNDLE_NAME}"

# ── Publish to GitHub Releases ────────────────────────────────────────────────
DOWNLOAD_URL_WASM="https://github.com/${REPO}/releases/download/${TAG}/${WASM_OUT}"
DOWNLOAD_URL_BUNDLE="https://github.com/${REPO}/releases/download/${TAG}/${BUNDLE_NAME}.zip"

echo "Publishing ${TAG} to ${REPO}..."
gh release delete "${TAG}" --repo "${REPO}" --yes 2>/dev/null || true
git tag -d "${TAG}" 2>/dev/null || true
git push origin ":refs/tags/${TAG}" 2>/dev/null || true

git tag "${TAG}"
git push origin "${TAG}"
gh release create "${TAG}" "${WASM_OUT}" "${BUNDLE_NAME}.zip" \
    --repo "${REPO}" \
    --title "${TAG}" \
    --notes "Release ${TAG}"

# ── Hash what GitHub actually serves ─────────────────────────────────────────
echo "Fetching published WASM to compute canonical SHA256..."
VERIFIED=$(mktemp /tmp/${NAME}-verify-XXXXXX.wasm)
PREV_SHA=""
SHA=""
for i in 1 2 3 4 5; do
    sleep 3
    curl -L -s -o "${VERIFIED}" "${DOWNLOAD_URL_WASM}"
    SHA=$(shasum -a 256 "${VERIFIED}" | awk '{print $1}')
    if [ "${SHA}" = "${PREV_SHA}" ]; then
        break
    fi
    PREV_SHA="${SHA}"
done
rm -f "${VERIFIED}"

echo ""
echo "Done!"
echo "WASM URL    : ${DOWNLOAD_URL_WASM}"
echo "Bundle URL  : ${DOWNLOAD_URL_BUNDLE}"
echo "SHA256      : ${SHA}"
echo ""
echo "Paste into registry.json:"
echo "  \"version\": \"${VERSION}\","
echo "  \"wasm_url\": \"${DOWNLOAD_URL_WASM}\","
echo "  \"sha256\": \"${SHA}\""
