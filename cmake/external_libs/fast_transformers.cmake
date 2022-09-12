set(REQ_URL "https://github.com/NVIDIA/FasterTransformer/archive/refs/tags/release/v5.0_tag.tar.gz")
set(MD5 "f2e06ec43f3b5b83017bd87b0427524f")
set(ft_libs "transformer-shared")


mindspore_add_pkg(fast_transformers
        VER 0.5.0
        URL ${REQ_URL}
        MD5 ${MD5}
        LIBS ${ft_libs}
        LIB_PATH lib
        PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/fast_transformer/001-fast_transformer.patch
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DEXAMPLES=off)
include_directories(${fast_transformers_INC})

add_library(mindspore::fast_transformers ALIAS fast_transformers::transformer-shared)
