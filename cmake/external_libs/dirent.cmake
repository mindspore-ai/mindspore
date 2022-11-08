if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/dirent/repository/archive/1.23.2.zip")
    set(SHA256 "8a442ab85f670ed9db275b4f3e62f61872afbd8acbcbd1bb4cf212b7658e8838")
else()
    set(REQ_URL "https://github.com/tronkko/dirent/archive/refs/tags/1.23.2.zip")
    set(SHA256 "4bcf07266f336bcd540fec5f75e90f027bd5081d3752f9ea5d408ef6ae30a897")
endif()


if(MSVC)
    mindspore_add_pkg(dirent
        VER 1.23.2
        HEAD_ONLY ./include
        RELEASE on
        URL ${REQ_URL}
        SHA256 ${SHA256})
    include_directories(${dirent_INC})
endif()


