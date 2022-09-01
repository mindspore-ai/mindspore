set(REQ_URL "https://github.com/tronkko/dirent/archive/refs/tags/1.23.2.zip")
set(MD5 "43514791ab73ef5ac7c490afc7c3bab2")

if(MSVC)
    mindspore_add_pkg(dirent
        VER 1.23.2
        HEAD_ONLY ./include
        RELEASE on
        URL ${REQ_URL}
        MD5 ${MD5})
    include_directories(${dirent_INC})
endif()


