include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB OPS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/populate/*.cc
    )
if(ENABLE_V0)
    file(GLOB_RECURSE COMPAT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/compat/*.cc)
    file(GLOB OPS_SRC_V0 ${CMAKE_CURRENT_SOURCE_DIR}/populate/v0/*.cc)
    set(OPS_SRC ${OPS_SRC} ${COMPAT_SRC} ${OPS_SRC_V0})
endif()

add_library(cpu_ops_mid OBJECT ${OPS_SRC})
add_dependencies(cpu_ops_mid fbs_src)
