include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB OPS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/populate/*.cc
    )
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
if(MSLITE_ENABLE_V0)
    file(GLOB_RECURSE COMPAT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/compat/*.cc)
    file(GLOB OPS_SRC_V0 ${CMAKE_CURRENT_SOURCE_DIR}/populate/v0/*.cc)
    if(MSLITE_ENABLE_STRING_KERNEL)
        file(GLOB OPS_SRC_STRING_V0
                ${CMAKE_CURRENT_SOURCE_DIR}/populate/v0/string/*.cc
                )
        set(OPS_SRC_V0
                ${OPS_SRC_V0}
                ${OPS_SRC_STRING_V0}
                )
    endif()
    if(MSLITE_ENABLE_CONTROLFLOW)
        file(GLOB OPS_SRC_CONTROL_TENSORLIST_V0
                ${CMAKE_CURRENT_SOURCE_DIR}/populate/v0/control/*.cc
                )
        set(OPS_SRC_V0
                ${OPS_SRC_V0}
                ${OPS_SRC_CONTROL_TENSORLIST_V0}
                )
    endif()
    set(OPS_SRC ${OPS_SRC} ${COMPAT_SRC} ${OPS_SRC_V0})
endif()

add_library(cpu_ops_mid OBJECT ${OPS_SRC})
add_dependencies(cpu_ops_mid fbs_src)
