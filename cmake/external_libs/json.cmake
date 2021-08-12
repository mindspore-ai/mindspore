if(MSVC)
    set(flatbuffers_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_CFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(nlohmann_json_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(nlohmann_json_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.6.1.zip")
    set(MD5 "36ea0d9a709c6667b2798a62f6b197ae")
    set(INCLUDE "./include")
else()
    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip")
    set(MD5 "0dc903888211db3a0f170304cd9f3a89")
    set(INCLUDE "./")
endif()

mindspore_add_pkg(nlohmann_json
        VER 3.6.1
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        MD5 ${MD5})
include_directories(${nlohmann_json_INC})
add_library(mindspore::json ALIAS nlohmann_json)
