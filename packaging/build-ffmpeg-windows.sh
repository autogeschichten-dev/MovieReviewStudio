#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="$repo_root/build/ffmpeg"
workspace="$(mktemp -d)"
trap 'rm -rf -- "$workspace"' EXIT
prefix="$(cygpath -m "$workspace/prefix")"

# Pin both source trees so the included source archive matches the binaries.
x264_commit=31e19f92f00c7003fa115047ce50978bc98c3a0d
ffmpeg_commit=3a0867c2bfda4a4d4309ca1a8cbdc6175e67f587

fetch_source() {
    local name="$1" remote="$2" commit="$3"
    git init -q "$workspace/$name"
    git -C "$workspace/$name" fetch --depth=1 "$remote" "$commit"
    git -C "$workspace/$name" checkout --quiet --detach FETCH_HEAD
    test "$(git -C "$workspace/$name" rev-parse HEAD)" = "$commit"
}

fetch_source x264 https://github.com/mirror/x264.git "$x264_commit"
fetch_source ffmpeg https://github.com/FFmpeg/FFmpeg.git "$ffmpeg_commit"

(
    cd "$workspace/x264"
    ./configure --host=x86_64-w64-mingw32 --prefix="$prefix" --enable-static --disable-cli --disable-opencl
    make -j"$(nproc)"
    make install
)
(
    cd "$workspace/ffmpeg"
    PKG_CONFIG_PATH="$prefix/lib/pkgconfig" ./configure \
        --arch=x86_64 --target-os=mingw32 --disable-autodetect \
        --disable-shared --enable-static --disable-doc --disable-ffplay \
        --enable-gpl --enable-libx264 --pkg-config-flags=--static \
        --extra-ldflags=-static
    make -j"$(nproc)"
)

mkdir -p "$output/bin" "$output/source"
cp "$workspace/ffmpeg/"{ffmpeg,ffprobe}.exe "$output/bin/"
git -C "$workspace/ffmpeg" archive --format=zip --prefix=ffmpeg/ \
    -o "$output/source/ffmpeg.zip" HEAD
git -C "$workspace/x264" archive --format=zip --prefix=x264/ \
    -o "$output/source/x264.zip" HEAD
cp "$workspace/ffmpeg/COPYING.GPLv2" "$output/source/"
cp "$repo_root/packaging/build-ffmpeg-windows.sh" "$output/source/"
