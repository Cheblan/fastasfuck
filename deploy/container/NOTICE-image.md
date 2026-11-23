# Third-party software in this image

fastasfuck is licensed under the AGPL-3.0 (`/opt/fastasfuck/share/LICENSE`); third-party dependencies and build libraries are listed in `/opt/fastasfuck/share/THIRD_PARTY.md`.

The image is built on official Arch Linux packages. Runtime GPU execution relies on AMD ROCm packages from the Arch Linux official repositories:
- `hip-runtime-amd` (MIT)
- `hsa-rocr` (NCSA)
- `comgr` (NCSA)
- `rocm-llvm-libs` (Apache-2.0 with LLVM Exception)
- `rocprofiler-register` (MIT)

Underlying system dependencies (`glibc`, `numactl`, `libdrm`, `libelf`, `ncurses`) are provided by Arch Linux and licensed under their respective open-source licenses.
