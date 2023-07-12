if(MSVC)
    set(flatbuffers_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_CFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(nlohmann_json3101_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(nlohmann_json3101_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
endif()

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.10.1.zip")
    set(SHA256 "5c7d0a0542431fef628f8dc4c34fd022fe8747ccb577012d58f38672d8747e0d")
    set(INCLUDE "./include")
else()

    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.10.1/include.zip")
    set(SHA256 "144268f7f85afb0f0fbea7c796723c849724c975f9108ffdadde9ecedaa5f0b1")
    set(INCLUDE "./include")
endif()

set(ENABLE_NATIVE_JSON "off")
if(EXISTS ${TOP_DIR}/mindspore/lite/providers/json/native_json.cfg)
    set(ENABLE_NATIVE_JSON "on")
endif()
if(ENABLE_NATIVE_JSON)
    file(STRINGS ${TOP_DIR}/mindspore/lite/providers/json/native_json.cfg native_json_path)
    mindspore_add_pkg(nlohmann_json3101
            VER 3.10.1
            HEAD_ONLY ${INCLUDE}
            DIR ${native_json_path})
    add_library(mindspore::json ALIAS nlohmann_json3101)
else()
    mindspore_add_pkg(nlohmann_json3101
            VER 3.10.1
            HEAD_ONLY ${INCLUDE}
            URL ${REQ_URL}
            SHA256 ${SHA256})
    include_directories(${nlohmann_json3101_INC})
    add_library(mindspore::json ALIAS nlohmann_json3101)
endif()