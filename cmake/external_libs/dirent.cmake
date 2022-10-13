if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/dirent/repository/archive/1.23.2.zip")
    set(MD5 "9a1d076cd5fe6272e9f5078c67a7ca4c")
else()
    set(REQ_URL "https://github.com/tronkko/dirent/archive/refs/tags/1.23.2.zip")
    set(MD5 "43514791ab73ef5ac7c490afc7c3bab2")
endif()


if(MSVC)
    mindspore_add_pkg(dirent
        VER 1.23.2
        HEAD_ONLY ./include
        RELEASE on
        URL ${REQ_URL}
        MD5 ${MD5})
    include_directories(${dirent_INC})
endif()


