## cross compile
# find ccec compiler
find_program(CMAKE_CCE_COMPILER NAMES "ccec" PATHS "$ENV{PATH}" DOC "CCE Compiler")
find_file(ASCEND_C ccec REQUIRED)
message(STATUS "ccec is in " ${ASCEND_C} " " ${MSLITE_ASCEND_TARGET} " ascend1 " ${ASCEND_CUSTOM_PATH})
get_filename_component(ASCEND_C_DIR ${ASCEND_C} DIRECTORY)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CCE_FLAGS "${CMAKE_CCE_FLAGS} -g")
endif()

set(COMMON_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})
if(DEFINED ENV{MSLITE_ASCEND_RUN_MODE})
set(ASCEND_RUN_MODE $ENV{MSLITE_ASCEND_RUN_MODE})
else()
set(ASCEND_RUN_MODE "ONBOARD")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vector_core)

set(TILING_FILES
${CMAKE_CURRENT_SOURCE_DIR}/tiling_if.cc
${CMAKE_CURRENT_SOURCE_DIR}/tiling.cc
)

set(HOST_FILES
${CMAKE_CURRENT_SOURCE_DIR}/encoder_ps.cc
${CMAKE_CURRENT_SOURCE_DIR}/gemm.cc
${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
)

if(ASCEND_RUN_MODE STREQUAL "ONBOARD")
set(TILING_OBJ tiling_mid)
add_library(${TILING_OBJ} OBJECT ${TILING_FILES})
else()
add_executable(tilingx ${TILING_FILES})
target_compile_definitions(tilingx PRIVATE PIPELINE)
set(TILING_OBJ tilingx)
endif()

target_include_directories(${TILING_OBJ} PRIVATE
${ASCEND_INSTALL_PATH}/include/
${ASCEND_INSTALL_PATH}/runtime/include/
${ASCEND_INSTALL_PATH}/runtime/include/tiling/
)

target_link_directories(${TILING_OBJ} PUBLIC
${ASCEND_INSTALL_PATH}/lib64
)

target_link_libraries(${TILING_OBJ} PRIVATE
tiling_api
)
if(ASCEND_RUN_MODE STREQUAL "SIMULATOR")
target_link_libraries(${TILING_OBJ} PRIVATE
-Wl,--no-as-needed
c_sec
graph
graph_base
register
-Wl,--as-needed
)
endif()

add_library(ascend_native_impl_mid OBJECT ${HOST_FILES})
if(ASCEND_RUN_MODE STREQUAL "ONBOARD")
    target_link_libraries(ascend_native_impl_mid PUBLIC tiling_mid)
endif()

if(MS_ENABLE_ASCEND_DISTRIBUTION STREQUAL "on")
    target_compile_definitions(ascend_native_impl_mid PUBLIC MS_ENABLE_ASCEND_DISTRIBUTION)
    include(${TOP_DIR}/cmake/external_libs/ompi.cmake)
    add_library(mindspore::ompi ALIAS ompi::mpi)
    add_library(hccl_adapter_mid OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/hccl_adapter.cc)
    target_link_libraries(ascend_native_impl_mid PUBLIC hccl_adapter_mid)
endif()

target_include_directories(ascend_native_impl_mid PRIVATE
${ASCEND_INSTALL_PATH}/include
${ASCEND_INSTALL_PATH}/runtime/include
)
