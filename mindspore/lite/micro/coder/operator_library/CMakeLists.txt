option(MICRO_CMSIS_X86 "build for CMSIS x86" ON)
option(ENABLE_ASAN "enable asan" OFF)
option(CMAKE_BUILD_TYPE "debug or release" Debug)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DEBUG_MODE "Debug")
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "-DDebug -g -fvisibility=default ${CMAKE_C_FLAGS}")
else()
    set(CMAKE_C_FLAGS "-fPIC -fPIE -Werror -O3 -fstack-protector-strong -fomit-frame-pointer ${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS_Release "${CMAKE_C_FLAGS_Release} -O3 -ffunction-sections -fdata-sections")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

if(ENABLE_ASAN)
    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS} -lasan")
endif()

set(MICRO_CMAKE_PATH ${MICRO_DIR}/cmake)
set(OPERATOR_LIBRARY_PATH ${CMAKE_BINARY_DIR}/operator_library)
set(HEADER_PATH "${OPERATOR_LIBRARY_PATH}/include")

message("===========>start to pack operators' head file")
file(REMOVE_RECURSE ${OPERATOR_LIBRARY_PATH})
file(MAKE_DIRECTORY ${OPERATOR_LIBRARY_PATH})
file(GLOB NNACL_FILES GLOB ${LITE_DIR}/nnacl/*.h)
file(COPY ${NNACL_FILES} DESTINATION ${HEADER_PATH}/nnacl/)
file(COPY ${LITE_DIR}/nnacl/base
        ${LITE_DIR}/nnacl/int8
        ${LITE_DIR}/nnacl/fp32
        ${LITE_DIR}/nnacl/intrinsics
        DESTINATION ${HEADER_PATH}/nnacl FILES_MATCHING PATTERN "*.h")
file(COPY ${MICRO_DIR}/coder/operator_library/wrapper DESTINATION ${HEADER_PATH} FILES_MATCHING PATTERN "*.h")
file(COPY ${CMAKE_BINARY_DIR}/cmsis/CMSIS/Core/Include
        ${CMAKE_BINARY_DIR}/cmsis/CMSIS/DSP/Include
        ${CMAKE_BINARY_DIR}/cmsis/CMSIS/NN/Include
        DESTINATION ${HEADER_PATH}/CMSIS/Core)

if(PLATFORM_ARM64)
    set(MICRO_BUILD_ARM64 ON)
endif()
if(PLATFORM_ARM32)
    set(MICRO_BUILD_ARM32A ON)
endif()

include(${MICRO_CMAKE_PATH}/package_android.cmake)
include(${MICRO_CMAKE_PATH}/package_nnacl.cmake)
include(${MICRO_CMAKE_PATH}/package_cmsis.cmake)
include(${MICRO_CMAKE_PATH}/package_wrapper.cmake)

list(APPEND OP_FILES ${NNACL_OPS} ${WRAPPER_SRC} ${RUNTIME_SRC})

if(PLATFORM_ARM64)
    set(LIB_PATH "${OPERATOR_LIBRARY_PATH}/lib/arm64")
elseif(PLATFORM_ARM32)
    set(LIB_PATH "${OPERATOR_LIBRARY_PATH}/lib/arm32a")
else()
    set(LIB_PATH "${OPERATOR_LIBRARY_PATH}/lib/x86")
    list(APPEND OP_FILES ${CMSIS_OPS})
endif()

# generate static library
add_library(ops STATIC ${OP_FILES})
target_compile_definitions(ops PRIVATE NOT_USE_STL)
install(TARGETS ops ARCHIVE DESTINATION ${LIB_PATH})
