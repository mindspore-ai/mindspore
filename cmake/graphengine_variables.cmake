# path variables for graphengine submodule, it has to be included after mindspore/core
# and minspore/ccsrc to prevent conflict of op headers
if(ENABLE_D OR ENABLE_ACL OR ENABLE_TESTCASES)
    include_directories(${GRAPHENGINE_PATH}/inc)
    include_directories(${GRAPHENGINE_PATH}/inc/external)
    include_directories(${GRAPHENGINE_PATH}/inc/framework)
    include_directories(${GRAPHENGINE_PATH}/base)
    include_directories(${GRAPHENGINE_PATH}/third_party/fwkacllib/inc)
    include_directories(${GRAPHENGINE_PATH}/third_party/fwkacllib/inc/aicpu)
    include_directories(${GRAPHENGINE_PATH}/third_party/fwkacllib/inc/toolchain)
    include_directories(${GRAPHENGINE_PATH}/metadef/inc)
    include_directories(${GRAPHENGINE_PATH}/metadef/inc/external)
    include_directories(${GRAPHENGINE_PATH}/metadef/inc/external/graph)
endif()