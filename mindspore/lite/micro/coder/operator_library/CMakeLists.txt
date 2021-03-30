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

include_directories(${LITE_DIR})
set(MICRO_CMAKE_PATH ${MICRO_DIR}/cmake)
set(OPERATOR_LIBRARY_PATH ${CMAKE_BINARY_DIR}/operator_library)
set(HEADER_PATH "${OPERATOR_LIBRARY_PATH}/include")
set(LIB_PATH "${OPERATOR_LIBRARY_PATH}/lib")
set(NNACL_DIR ${LITE_DIR}/nnacl)

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
    add_compile_definitions(ENABLE_ARM64)
endif()

if(PLATFORM_ARM32)
    add_compile_definitions(ENABLE_ARM32)
    add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    add_compile_definitions(ENABLE_NEON)
    add_compile_definitions(ENABLE_ARM)
endif()

file(GLOB KERNEL_SRC
        ${NNACL_DIR}/*.c
        ${NNACL_DIR}/base/*.c
        ${NNACL_DIR}/fp32/*.c
        ${NNACL_DIR}/int8/*.c
        )

if(PLATFORM_ARM64)
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/assembly/arm64/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

if(PLATFORM_ARM32)
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/assembly/arm32/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

set(NNACL_OPS ${KERNEL_SRC} ${ASSEMBLY_SRC})
set(RUNTIME_SRC ${LITE_DIR}/src/runtime/thread_pool.c)

include(${MICRO_CMAKE_PATH}/package_cmsis.cmake)
include(${MICRO_CMAKE_PATH}/package_wrapper.cmake)

list(APPEND OP_FILES ${NNACL_OPS} ${WRAPPER_SRC} ${RUNTIME_SRC})

if(NOT PLATFORM_ARM64 AND NOT PLATFORM_ARM32)
    list(APPEND OP_FILES ${CMSIS_OPS})
endif()


# generate static library
add_library(ops STATIC ${OP_FILES})

function(create_library)
    string(CONCAT library_name "lib" ops ".a")
    string(CONCAT opt_name "lib" opt_ops ".a")
    set(ANDROID_AR
            ${ANDROID_NDK}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-ar)
    add_custom_command(TARGET ops
            POST_BUILD
            COMMAND rm -rf tmp
            COMMAND mkdir tmp
            COMMAND mv ${opt_name} ./tmp && cd ./tmp && ${ANDROID_AR} -x ${opt_name}
            COMMAND mv ${library_name} ./tmp && cd ./tmp && ${ANDROID_AR} -x ${library_name}
            COMMENT "unzip raw static library ${library_name}"
            COMMAND ${ANDROID_AR} cr ${library_name} ./tmp/*.o
            COMMAND rm -rf tmp
            COMMENT "generate specified static library ${library_name}"
            )
endfunction()

if(PLATFORM_ARM64)
    file(GLOB OPT_SRC ${NNACL_DIR}/assembly/opt/*.S)
    set_property(SOURCE ${OPT_SRC} PROPERTY LANGUAGE C)
    add_library(opt_ops STATIC ${OPT_SRC})

    add_dependencies(ops opt_ops)
    set_target_properties(opt_ops PROPERTIES COMPILE_FLAGS -march=armv8.2-a+dotprod)
    create_library()
endif()

install(TARGETS ops ARCHIVE DESTINATION ${LIB_PATH})
