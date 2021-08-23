project(optimize)

set(NNACL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(NNACL_DIR)

########################### optimized files ###########################
file(GLOB FP16_C_SRC ${NNACL_DIR}/fp16/*.c)
if(PLATFORM_ARM32)
    file(GLOB FP16_NEON_SRC ${NNACL_DIR}/assembly/arm82_aarch32_fp16/*.S)
else()
    file(GLOB FP16_NEON_SRC ${NNACL_DIR}/assembly/fp16/*.S)
    file(GLOB SDOT_SRC ${NNACL_DIR}/assembly/opt/*.S)
    set_property(SOURCE ${SDOT_SRC} PROPERTY LANGUAGE C)
endif()

set_property(SOURCE ${FP16_C_SRC} PROPERTY LANGUAGE C)
set_property(SOURCE ${FP16_NEON_SRC} PROPERTY LANGUAGE C)

if(APPLE)
    set_source_files_properties(${SDOT_SRC} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    set_source_files_properties(${FP16_NEON_SRC} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
endif()
########################### share library build ########################
list(APPEND FP16_FILES ${FP16_C_SRC})
list(APPEND FP16_FILES ${FP16_NEON_SRC})

if(SUPPORT_TRAIN)
    file(GLOB FP16_TRAIN_SRC ${NNACL_DIR}/fp16_grad/*.c)
    list(APPEND FP16_FILES ${FP16_TRAIN_SRC})
endif()

string(REPLACE "-fvisibility=hidden" "-fvisibility=default" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

if(NOT PLATFORM_ARM32)
    list(APPEND SDOT_FILES ${SDOT_SRC})
    add_library(nnacl_optimize_mid OBJECT ${SDOT_FILES})
    add_dependencies(nnacl_optimize_mid fbs_src)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod+fp16")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+fp16")
endif()

if(MSLITE_ENABLE_FP16)
    add_library(nnacl_fp16_mid OBJECT ${FP16_FILES})
    add_dependencies(nnacl_fp16_mid fbs_src)
    if(PLATFORM_ARM32)
        target_compile_options(nnacl_fp16_mid PRIVATE -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8 -mfloat-abi=softfp)
    endif()
endif()