# The CUDA toolkit 11 and higher versions include cub library,
# but the lower version (CUDA 10) doesn't support.
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    set(cub_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    find_path(CUB_INCLUDE_DIRS
            HINTS "${CUDA_INCLUDE_DIRS}"
            NAMES cub/cub.cuh
            DOC "The directory where cub library reside.")
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(cub
            FOUND_VAR CUB_FOUND
            REQUIRED_VARS CUB_INCLUDE_DIRS)
    if(CUB_FOUND)
        include_directories(${CUB_INCLUDE_DIRS})
    else()
        set(REQ_URL "https://github.com/NVlabs/cub/archive/1.8.0.zip")
        set(SHA256 "6bfa06ab52a650ae7ee6963143a0bbc667d6504822cbd9670369b598f18c58c3")
        set(INCLUDE "cub")
        mindspore_add_pkg(cub
                VER 1.8.0
                HEAD_ONLY ${INCLUDE}
                URL ${REQ_URL}
                SHA256 ${SHA256})
        include_directories(${cub_INC}/../)
    endif()
endif()
