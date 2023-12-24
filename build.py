import subprocess
import os
from packaging.version import parse, Version
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

# package name managed by pip, which can be remove by `pip uninstall tiny_pkg`
PACKAGE_NAME = "tiny_pkg"
FORCE_SINGLE_THREAD = os.getenv("FLASH_ATTENTION_FORCE_SINGLE_THREAD", "FALSE") == "TRUE"

ext_modules = []
generator_flag = []
cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")


# helper function to get cuda version
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    if not FORCE_SINGLE_THREAD:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# cuda module
ext_modules.append(
    CUDAExtension(
        # package name for import
        name="tiny_api_cuda",
        sources=[
            "csrc/cuda_api.cu",
        ],
        extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "--use_fast_math",
                    # "--ptxas-options=-v",
                    # "--ptxas-options=-O2",
                    # "-lineinfo",
                ]
                + generator_flag
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "include",
            # Path(this_dir) / "some" / "thing" / "more",
        ],
    )
)

# cpp module
ext_modules.append(
    CppExtension(
        # package name for import
        name="tiny_api_c",
        sources=[
            "csrc/c_api.cpp",
        ],
        extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # no nvcc flags needed
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "include",
            # Path(this_dir) / "some" / "thing" / "more",
        ],
    )
)


setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    description="Tiny cuda and c api binding for pytorch.",
    ext_modules=ext_modules,
    cmdclass={ "build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)




