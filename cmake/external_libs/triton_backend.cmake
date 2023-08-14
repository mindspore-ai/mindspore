set(COMMON_GIT_REPOSITORY "https://gitee.com/mirrors_triton-inference-server/common.git")
set(COMMON_GIT_TAG "1df32b982a6ed11ead3271a55b04bf6e7abc1cf9")
set(COMMON_SHA256 "1df32b982a6ed11ead3271a55b04bf6e7abc1cf9")  #unused
mindspore_add_pkg(triton_common
        LIBS tritonasyncworkqueue
        GIT_REPOSITORY ${COMMON_GIT_REPOSITORY}
        GIT_TAG ${COMMON_GIT_TAG}
        SHA256 ${COMMON_SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/triton/triton_common.patch001
        CMAKE_OPTION
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DTRITON_COMMON_ENABLE_JSON:BOOL=off
        )

set(CORE_GIT_REPOSITORY "https://gitee.com/mirrors_triton-inference-server/core.git")
set(CORE_GIT_TAG "9714cd666d512c4a2c687d028cd4ebf3d354a9dc")
set(CORE_SHA256 "9714cd666d512c4a2c687d028cd4ebf3d354a9dc")  #unused
mindspore_add_pkg(triton_core
        LIBS tritonserver
        GIT_REPOSITORY ${CORE_GIT_REPOSITORY}
        GIT_TAG ${CORE_GIT_TAG}
        SHA256 ${CORE_SHA256}
        LIB_SUFFIXES_PATH stubs
        PATCHES ${TOP_DIR}/third_party/patch/triton/triton_core.patch001
        CMAKE_OPTION
        -DCMAKE_BUILD_TYPE:STRING=Release
        )

set(BACKEND_GIT_REPOSITORY "https://gitee.com/mirrors_triton-inference-server/backend.git")
set(BACKEND_GIT_TAG "2f3f44eba661af9b4fcd62160eafd9012db32652")
set(BACKEND_SHA256 "2f3f44eba661af9b4fcd62160eafd9012db32652")  #unused
mindspore_add_pkg(triton_backend
        LIBS tritonbackendutils
        GIT_REPOSITORY ${BACKEND_GIT_REPOSITORY}
        GIT_TAG ${BACKEND_GIT_TAG}
        SHA256 ${BACKEND_SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/triton/triton_backend.patch001
        CMAKE_OPTION
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
        -DTRITON_RAPID_JSON_PATH:STRING=${TRITON_RAPID_JSON_PATH}
        -DTRITON_CORE_PATH=${triton_core_INC}
        -DTRITON_COMMON_PATH=${triton_common_INC}
        )

include_directories(${triton_backend_INC})
include_directories(${triton_common_INC})
add_library(mindspore::tritonasyncworkqueue ALIAS triton_common::tritonasyncworkqueue)
add_library(mindspore::tritonserver ALIAS triton_core::tritonserver)
add_library(mindspore::tritonbackendutils ALIAS triton_backend::tritonbackendutils)
