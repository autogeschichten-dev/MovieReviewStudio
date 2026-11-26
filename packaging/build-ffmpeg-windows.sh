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

encoder_flags=()
link_flags="-static"
if [[ ${1:-} == --with-lame ]]; then
    lame_archive="$workspace/lame-3.100.tar.gz"
    curl --fail --location --retry 3 \
        https://deb.debian.org/debian/pool/main/l/lame/lame_3.100.orig.tar.gz -o "$lame_archive"
    printf '%s  %s\n' ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e "$lame_archive" | sha256sum -c -
    tar -xzf "$lame_archive" -C "$workspace"
    (
        cd "$workspace/lame-3.100"
        ./configure --host=x86_64-w64-mingw32 --prefix="$prefix" --enable-static \
            --disable-shared --disable-frontend --disable-nasm
        make -j"$(nproc)"
        make install
    )
    encoder_flags=(--enable-libmp3lame "--extra-cflags=-I$prefix/include")
    link_flags+=" -L$prefix/lib"
fi

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
        --extra-ldflags="$link_flags" "${encoder_flags[@]}"
    make -j"$(nproc)"
)

mkdir -p "$output/bin" "$output/source"
cp "$workspace/ffmpeg/"{ffmpeg,ffprobe}.exe "$output/bin/"
git -C "$workspace/ffmpeg" archive --format=zip --prefix=ffmpeg/ \
    -o "$output/source/ffmpeg.zip" HEAD
git -C "$workspace/x264" archive --format=zip --prefix=x264/ \
    -o "$output/source/x264.zip" HEAD
cp "$workspace/ffmpeg/COPYING.GPLv2" "$output/source/"
if [[ -n ${lame_archive:-} ]]; then
    cp "$lame_archive" "$output/source/"
    cp "$workspace/lame-3.100/COPYING" "$output/source/LAME-COPYING"
fi
cp "$repo_root/packaging/build-ffmpeg-windows.sh" "$output/source/"
