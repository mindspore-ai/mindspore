set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2)
cmake_minimum_required(VERSION 3.5.1)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")

set(ASCEND_HOME $ENV{ASCEND_HOME})

project(custom_aicore_ops)
set(VENDOR_NAME "custom_aicore_ops")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
endif()

execute_process(COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR})

if(IS_DIRECTORY "${OPS_DIR}/op_proto")
  add_subdirectory(${OPS_DIR}/op_proto op_proto)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tbe")
  add_subdirectory(tbe)
endif()

message(STATUS "operation system is ${CMAKE_HOST_SYSTEM_NAME}")

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/op_proto")
  set(ALL_MODULES ${OP_PROTO_TARGET})
endif()


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tbe/CMakeLists.txt")
  set(ALL_MODULES ${ALL_MODULES} config_json)
endif()

message(STATUS "ALL_MODULES=${ALL_MODULES}")
add_custom_target(${RUN_TARGET} ALL DEPENDS ${ALL_MODULES})
message("RUN_TARGET ${RUN_TARGET}")

set(VENDOR_IMPL "${VENDOR_NAME}_impl")
add_custom_command(
  TARGET ${RUN_TARGET}
  PRE_BUILD
  COMMAND
    mkdir -p
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_DIR}/op_impl/ai_core/tbe/${VENDOR_IMPL}
  COMMAND
    mkdir -p
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_DIR}/op_impl/vector_core/tbe/${VENDOR_IMPL}
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/gen_impl_and_merge_json.sh
          ${OPS_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${VENDOR_NAME}
)

# specify output dir
set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/vendors/${VENDOR_NAME})
set(INSTALL_LIBRARY_DIR lib/plugin/ascend)
install(DIRECTORY ${OUTPUT_DIR} DESTINATION ${INSTALL_LIBRARY_DIR})

