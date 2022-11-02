set(REQ_URL "https://github.com/NVIDIA/cuCollections/archive/d6ba69b1fdab90ae144301e77eb93a2f130ede1d.tar.gz")
set(SHA256 "8968d3a426db5f48ece1dafd10d51e77c163a53a514055efe96b42b88b938a87")
set(INCLUDE "include")

mindspore_add_pkg(cucollections
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/cucollections/0001-refine-bitwise-compare.patch
        PATCHES ${TOP_DIR}/third_party/patch/cucollections/0002-add-get-api-of-dynamic_map.patch
        PATCHES ${TOP_DIR}/third_party/patch/cucollections/0003-add-erase-and-export-api.patch
        )
include_directories(${cucollections_INC})
