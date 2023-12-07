include(CMakeCommonLanguageInclude)

function(product_dir str newstr)
if("x${str}" STREQUAL "xascend610")
set(${newstr} "Ascend610")
elseif("x${str}" STREQUAL "xascend910")
set(${newstr} "Ascend910A")
elseif("x${str}" STREQUAL "xascend310")
set(${newstr} "Ascend310")
elseif("x${str}" STREQUAL "xascend310p")
set(${newstr} "Ascend310P1")
elseif("x${str}" STREQUAL "xascend920")
set(${newstr} "Ascend920A")
elseif("x${str}" STREQUAL "xascend910b")
set(${newstr} "Ascend910B1")
  else()
    string(SUBSTRING ${str} 0 1 _headlower)
    string(SUBSTRING ${str} 1 -1 _leftstr)
    string(TOUPPER ${_headlower} _headupper)
    set(${newstr} "${_headupper}${_leftstr}")
  endif()
endfunction()

message(STATUS "ASCEND_PRODUCT_TYPE:\n" "  ${ASCEND_PRODUCT_TYPE}")
message(STATUS "ASCEND_CORE_TYPE:\n" "  ${ASCEND_CORE_TYPE}")
message(STATUS "ASCEND_INSTALL_PATH:\n" "  ${ASCEND_INSTALL_PATH}")

set(CMAKE_C_COMPILER ${CMAKE_CCE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_CCE_COMPILER})
set(CMAKE_C_LINK ${CMAKE_CCE_COMPILER})
set(CMAKE_C_LINK_SHARED ${CMAKE_CCE_COMPILER})


if(DEFINED ASCEND_INSTALL_PATH)
    set(_CMAKE_ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH})
else()
    message(FATAL_ERROR
        "no, installation path found, should passing -DASCEND_INSTALL_PATH=<PATH_TO_ASCEND_INSTALLATION> in cmake"
    )
    set(_CMAKE_ASCEND_INSTALL_PATH)
endif()

if(DEFINED ASCEND_PRODUCT_TYPE)
    set(_CMAKE_CCE_COMMON_COMPILE_OPTIONS "--cce-auto-sync -std=c++17 -O2 -DTILING_KEY_VAR=0")
    if(ASCEND_PRODUCT_TYPE STREQUAL "")
        message(FATAL_ERROR "ASCEND_PRODUCT_TYPE must be non-empty if set.")
    elseif(ASCEND_PRODUCT_TYPE AND NOT ASCEND_PRODUCT_TYPE MATCHES "^ascend[0-9][0-9][0-9][a-zA-Z]?[1-9]?$")
        message(FATAL_ERROR
            "ASCEND_PRODUCT_TYPE: ${ASCEND_PRODUCT_TYPE}\n"
            "is not one of the following: ascend910, ascend310p, ascend910B1"
        )
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "ascend910")
        if(ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c100")
        else()
            message(FATAL_ERROR, "only AiCore inside")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS)
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "ascend310p")
        if(ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-m200")
        elseif(ASCEND_CORE_TYPE STREQUAL "VectorCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-m200-vec")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS
            "-mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-fp-ceiling=2 "
            )
        set(_CMAKE_CCE_COMPILE_OPTIONS "${_CMAKE_CCE_COMPILE_OPTIONS} -mllvm -cce-aicore-record-overflow=false")
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "ascend910B1")
        if(ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c220-cube")
        elseif(ASCEND_CORE_TYPE STREQUAL "VectorCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c220-vec")
        elseif(ASCEND_CORE_TYPE STREQUAL "MixCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c220")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS "-g -mllvm -cce-aicore-function-stack-size=16000 ")
        set(_CMAKE_CCE_COMPILE_OPTIONS "${_CMAKE_CCE_COMPILE_OPTIONS} -mllvm -cce-aicore-record-overflow=false ")
        set(_CMAKE_CCE_COMPILE_OPTIONS "${_CMAKE_CCE_COMPILE_OPTIONS} -mllvm -cce-aicore-addr-transform")
    endif()
endif()
set(CMAKE_CXX_FLAGS "-xcce ${_CMAKE_CCE_COMMON_COMPILE_OPTIONS} ${_CMAKE_CCE_COMPILE_OPTIONS} ")
string(APPEND CMAKE_CXX_FLAGS ${_CMAKE_COMPILE_AS_CCE_FLAG})

if(ASCEND_RUN_MODE STREQUAL "ONBOARD")
    target_link_libraries(${ASCEND_TARGET} PUBLIC runtime)
elseif(ASCEND_RUN_MODE STREQUAL "SIMULATOR")
    if(ASCEND_PRODUCT_TYPE STREQUAL "ascend910")
    target_link_libraries(${ASCEND_TARGET} PUBLIC pam_davinci)
    endif()
    target_link_libraries(${ASCEND_TARGET} PUBLIC runtime_camodel)
else()
    message(FATAL_ERROR
        "ASCEND_RUN_MODE: ${ASCEND_RUN_MODE}\n"
        "ASCEND_RUN_MODE must be one of the following: ONBOARD or SIMULATOR"
    )
endif()
target_link_libraries(${ASCEND_TARGET} PUBLIC ascendcl)

target_include_directories(${ASCEND_TARGET} PRIVATE
    ${_CMAKE_ASCEND_INSTALL_PATH}/acllib/include
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw/impl
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw/interface
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/include
    ${COMMON_INCLUDES}
)
product_dir(${ASCEND_PRODUCT_TYPE} PRODUCT_UPPER)
target_link_directories(${ASCEND_TARGET} PRIVATE
    ${_CMAKE_ASCEND_INSTALL_PATH}/runtime/lib64
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/simulator/${PRODUCT_UPPER}/lib
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${PRODUCT_UPPER}
)

if(UNIX)
  set(CMAKE_CCE_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_CCE_OUTPUT_EXTENSION .obj)
endif()
set(CMAKE_CCE_OUTPUT_EXTENSION_REPLACE 1)

target_link_options(${ASCEND_TARGET} PRIVATE --cce-fatobj-link ${_CMAKE_COMPILE_AS_CCE_FLAG})
set_target_properties(${ASCEND_TARGET} PROPERTIES LINKER_LANGUAGE C)
