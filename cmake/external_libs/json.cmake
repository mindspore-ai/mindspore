if(MSVC)
    set(flatbuffers_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_CFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(nlohmann_json3120_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(nlohmann_json3120_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
endif()

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.12.0.zip")
    set(SHA256 "ed098b2314ebd9b92cc4113bbdec15d570be2db1d7d8a796493432df115b4821")
    set(INCLUDE "./include")
else()
    ## The nlohmann_json3120 naming convention is used to distinguish different versions of the include.zip file.
    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.12.0/include.zip")
    set(SHA256 "b8cb0ef2dd7f57f18933997c9934bb1fa962594f701cd5a8d3c2c80541559372")
    set(INCLUDE "./include")
endif()

set(ENABLE_NATIVE_JSON "off")
if(EXISTS ${TOP_DIR}/mindspore/lite/providers/json/native_json.cfg)
    set(ENABLE_NATIVE_JSON "on")
endif()
if(ENABLE_NATIVE_JSON)
    file(STRINGS ${TOP_DIR}/mindspore/lite/providers/json/native_json.cfg native_json_path)
    mindspore_add_pkg(nlohmann_json3120
            VER 3.12.0
            HEAD_ONLY ${INCLUDE}
            DIR ${native_json_path})
    add_library(mindspore::json ALIAS nlohmann_json3120)
else()
    mindspore_add_pkg(nlohmann_json3120
            VER 3.12.0
            HEAD_ONLY ${INCLUDE}
            URL ${REQ_URL}
            SHA256 ${SHA256})
    include_directories(${nlohmann_json3120_INC})
    add_library(mindspore::json ALIAS nlohmann_json3120)
endif()