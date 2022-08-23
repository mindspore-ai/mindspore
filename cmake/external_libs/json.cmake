if(MSVC)
    set(flatbuffers_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_CFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(nlohmann_json3101_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(nlohmann_json3101_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.10.1.zip")
    set(MD5 "a402ee7412bd6c7fcb994946ff977c44")
    set(INCLUDE "./include")
else()

    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.10.1/include.zip")
    set(MD5 "990b11a4fd9e1b05be02959a439fd402")
    set(INCLUDE "./include")
endif()

mindspore_add_pkg(nlohmann_json3101
        VER 3.10.1
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        MD5 ${MD5})
include_directories(${nlohmann_json3101_INC})
add_library(mindspore::json ALIAS nlohmann_json3101)
