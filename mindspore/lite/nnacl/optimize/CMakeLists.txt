project(optimize)

set(NNACL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(NNACL_DIR)

########################### optimized files ###########################
file(GLOB SDOT_SRC ${NNACL_DIR}/assembly/opt/*.S)
file(GLOB FP16_C_SRC ${NNACL_DIR}/fp16/*.c)
file(GLOB FP16_NEON_SRC ${NNACL_DIR}/assembly/fp16/*.S)

set_property(SOURCE ${SDOT_SRC} PROPERTY LANGUAGE C)
set_property(SOURCE ${FP16_C_SRC} PROPERTY LANGUAGE C)
set_property(SOURCE ${FP16_NEON_SRC} PROPERTY LANGUAGE C)

if(APPLE)
    set_source_files_properties(${SDOT_SRC} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    set_source_files_properties(${FP16_NEON_SRC} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
endif()
########################### share library build ########################
list(APPEND SDOT_FILES ${SDOT_SRC})
list(APPEND FP16_FILES ${FP16_C_SRC})
list(APPEND FP16_FILES ${FP16_NEON_SRC})

if(SUPPORT_TRAIN)
    file(GLOB FP16_TRAIN_SRC ${NNACL_DIR}/fp16_grad/*.c)
    list(APPEND FP16_FILES ${FP16_TRAIN_SRC})
endif()

string(REPLACE "-fvisibility=hidden" "-fvisibility=default" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod+fp16")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+fp16")

add_library(nnacl_optimize_mid OBJECT ${SDOT_FILES})
add_dependencies(nnacl_optimize_mid fbs_src)

if(ENABLE_FP16)
    add_library(nnacl_fp16_mid OBJECT ${FP16_FILES})
    add_dependencies(nnacl_fp16_mid fbs_src)
endif()