if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/robin-hood-hashing/repository/archive/3.11.5.zip")
    set(MD5 "b1f36d958f0bd75671b43ccf4685a5be")
else()
    set(REQ_URL "https://github.com/martinus/robin-hood-hashing/archive/3.11.5.zip")
    set(MD5 "35154dc71e47762d9b56b2114bc906ca")
endif()
set(INCLUDE "./src")

mindspore_add_pkg(robin_hood_hashing
        VER 3.11.5
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        MD5 ${MD5})

include_directories(${robin_hood_hashing_INC})
