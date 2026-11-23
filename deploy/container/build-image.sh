#!/bin/bash
# Build fastasfuck's release container image on Arch Linux for AMD Strix Halo (gfx1151).
#
# Usage:
#   build-image.sh [FASTASFUCK_ROOT] [TAG]
#
# Example:
#   ./deploy/container/build-image.sh . latest
set -eo pipefail

root=${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
tag=${2:-latest}
here=$(cd "$(dirname "$0")" && pwd)

ENGINE="podman"
if ! command -v podman > /dev/null 2>&1; then
    if command -v docker > /dev/null 2>&1; then
        ENGINE="docker"
    else
        echo "build-image: neither podman nor docker found on PATH"
        exit 1
    fi
fi

commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "build-image: building fastasfuck:$tag ($commit) using $ENGINE on Arch Linux base..."

$ENGINE build \
    --label org.opencontainers.image.revision="$commit" \
    --label org.opencontainers.image.version="$tag" \
    --label org.opencontainers.image.source=https://github.com/Cheblan/fastasfuck \
    --label org.opencontainers.image.title=fastasfuck \
    --label org.opencontainers.image.licenses=AGPL-3.0-only \
    --label org.opencontainers.image.description="fastasfuck: High-throughput LLM inference engine on AMD Strix Halo (gfx1151)" \
    -t "fastasfuck:$tag" \
    -f "$here/Containerfile" \
    "$root"

echo "build-image: successfully built fastasfuck:$tag"
if [ "$ENGINE" = "podman" ]; then
    echo "Image size: $(podman image inspect "fastasfuck:$tag" --format '{{.Size}}' | numfmt --to=iec 2>/dev/null || true)"
fi
