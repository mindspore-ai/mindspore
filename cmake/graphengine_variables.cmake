# path variables for graphengine submodule, it has to be included after mindspore/core
# and minspore/ccsrc to prevent conflict of op headers
if(ENABLE_D OR ENABLE_ACL OR ENABLE_TESTCASES)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/inc)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/inc/external)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/inc/framework)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/base)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/third_party/fwkacllib/inc)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/third_party/fwkacllib/inc/aicpu)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/third_party/fwkacllib/inc/toolchain)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/metadef/inc)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/metadef/inc/external)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/metadef/inc/external/graph)
endif()