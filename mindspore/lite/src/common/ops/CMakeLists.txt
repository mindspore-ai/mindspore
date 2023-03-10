include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CCSRC_DIR}/plugin/device/cpu/kernel)
if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions -Wno-shorten-64-to-32 \
        -fno-aligned-allocation -DTARGET_OS_OSX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -ffunction-sections \
        -fdata-sections -ffast-math -fno-rtti -fno-exceptions -Wno-shorten-64-to-32 \
        -fno-aligned-allocation -DTARGET_OS_OSX")
endif()
file(GLOB OPS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/populate/*.cc
    )
if(NOT MSLITE_ENABLE_RUNTIME_PASS)
  list(REMOVE_ITEM OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/populate/custom_populate.cc)
endif()
if(MSLITE_ENABLE_STRING_KERNEL)
    file(GLOB OPS_SRC_STRING
            ${CMAKE_CURRENT_SOURCE_DIR}/populate/string/*.cc
            )
    set(OPS_SRC
            ${OPS_SRC}
            ${OPS_SRC_STRING}
            )
endif()
if(MSLITE_ENABLE_CONTROLFLOW)
    file(GLOB OPS_SRC_CONTROL_TENSORLIST
            ${CMAKE_CURRENT_SOURCE_DIR}/populate/control/*.cc
            )
    set(OPS_SRC
            ${OPS_SRC}
            ${OPS_SRC_CONTROL_TENSORLIST}
            )
endif()
if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    file(GLOB OPS_SRC_CORE_OPERATOR_POPULATE
            ${CMAKE_CURRENT_SOURCE_DIR}/operator_populate/*.cc
            )
    set(OPS_SRC
            ${OPS_SRC}
            ${OPS_SRC_CORE_OPERATOR_POPULATE}
            )
endif()

add_library(cpu_ops_mid OBJECT ${OPS_SRC})
add_dependencies(cpu_ops_mid fbs_src fbs_inner_src)

