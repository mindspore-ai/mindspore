if(MSVC)
    set(flatbuffers_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_CFLAGS "${CMAKE_CXX_FLAGS}")
    set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
else()
    set(nlohmann_json373_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(nlohmann_json373_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.7.3.zip")
    set(MD5 "b758acca4f3e133bacf919e31ca302e3")
    set(INCLUDE "./include")
else()
    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip")
    set(MD5 "fb96f95cdf609143e998db401ca4f324")
    set(INCLUDE "./include")
endif()

mindspore_add_pkg(nlohmann_json373
        VER 3.7.3
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        MD5 ${MD5})
include_directories(${nlohmann_json373_INC})
add_library(mindspore::json ALIAS nlohmann_json373)
