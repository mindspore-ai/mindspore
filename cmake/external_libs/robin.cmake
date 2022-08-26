set(REQ_URL "https://github.com/martinus/robin-hood-hashing/archive/3.11.5.zip")
set(MD5 "35154dc71e47762d9b56b2114bc906ca")
set(INCLUDE "./src")

mindspore_add_pkg(robin_hood_hashing
        VER 3.11.5
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        MD5 ${MD5})

include_directories(${robin_hood_hashing_INC})
