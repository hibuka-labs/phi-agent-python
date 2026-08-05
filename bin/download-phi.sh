#!/bin/bash
# Download the phi binary for the current platform.
# Run by pip post-install, or manually for development.
#
# Usage:  ./download-phi.sh [version]

set -euo pipefail

VERSION="${1:-0.2.5}"
BASE_URL="https://github.com/hibuka-labs/phi-agent/releases/download/v${VERSION}"

case "$(uname -s)" in
    Darwin)  OS="darwin" ;;
    Linux)   OS="linux" ;;
    *)       echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac

case "$(uname -m)" in
    arm64|aarch64) ARCH="arm64" ;;
    x86_64)        ARCH="x86_64" ;;
    *)             echo "Unsupported arch: $(uname -m)"; exit 1 ;;
esac

TARBALL="phi-${OS}-${ARCH}.tar.gz"
URL="${BASE_URL}/${TARBALL}"
DEST_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Downloading phi v${VERSION} for ${OS}/${ARCH}..."
curl -fsSL "$URL" -o "/tmp/${TARBALL}"

echo "Extracting to ${DEST_DIR}..."
tar -xzf "/tmp/${TARBALL}" -C "$DEST_DIR"
chmod +x "${DEST_DIR}/phi"

rm "/tmp/${TARBALL}"
echo "Done — phi binary installed at ${DEST_DIR}/phi"
