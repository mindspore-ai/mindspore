set(REQ_URL "https://github.com/NVIDIA/cuCollections/archive/d6ba69b1fdab90ae144301e77eb93a2f130ede1d.tar.gz")
set(MD5 "196a453e5db52e904a906b13b2b8771c")
set(INCLUDE "include")

mindspore_add_pkg(cucollections
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${TOP_DIR}/third_party/patch/cucollections/0001-refine-bitwise-compare.patch
        PATCHES ${TOP_DIR}/third_party/patch/cucollections/0002-add-get-api-of-dynamic_map.patch
        PATCHES ${TOP_DIR}/third_party/patch/cucollections/0003-add-erase-and-export-api.patch
        )
include_directories(${cucollections_INC})
