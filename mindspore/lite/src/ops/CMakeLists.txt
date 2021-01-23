include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB OPS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/populate/*.cc
    )
if(ENABLE_V0)
    file(GLOB_RECURSE COMPAT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/compat/*.cc)
    set(OPS_SRC ${OPS_SRC} ${COMPAT_SRC})
endif()

add_library(cpu_ops_mid OBJECT ${OPS_SRC})
add_dependencies(cpu_ops_mid fbs_src)
