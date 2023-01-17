if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/robin-hood-hashing/repository/archive/3.11.5.zip")
    set(SHA256 "8d1f5d5ee447e5827032d1eb8b1609134618b1cc5c5bcadfcbfed99a2d3583d4")
else()
    set(REQ_URL "https://github.com/martinus/robin-hood-hashing/archive/3.11.5.zip")
    set(SHA256 "7aa183252527ded7f46186c1e2f4efe7d6139a3b7c0869c1b6051bd7260587ed")
endif()
set(INCLUDE "./src")

mindspore_add_pkg(robin_hood_hashing
        VER 3.11.5
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        SHA256 ${SHA256})

include_directories(${robin_hood_hashing_INC})
