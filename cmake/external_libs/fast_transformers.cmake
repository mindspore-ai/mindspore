set(REQ_URL "https://github.com/NVIDIA/FasterTransformer/archive/refs/tags/release/v5.0_tag.tar.gz")
set(SHA256 "7adffe2d53b3c1544295a6b7d1887e59b044eba25dd3e150bc909168d5e99081")
set(ft_libs "transformer-shared")

if(DEFINED ENV{MSLITE_GPU_ARCH})
  set(arch_opt -DSM=$ENV{MSLITE_GPU_ARCH})
endif()

mindspore_add_pkg(fast_transformers
        VER 0.5.0
        URL ${REQ_URL}
        SHA256 ${SHA256}
        LIBS ${ft_libs}
        LIB_PATH lib
        PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/fast_transformer/001-fast_transformer.patch
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release ${arch_opt} -DEXAMPLES=off)
include_directories(${fast_transformers_INC})

add_library(mindspore::fast_transformers ALIAS fast_transformers::transformer-shared)
