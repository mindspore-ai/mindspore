# set compile option -std=c++11
set(CMAKE_CXX_STANDARD 11)

# set compile option -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if(NOT DEFINED ASCEND_TENSOR_COMPILER_INCLUDE)
  if(NOT "x$ENV{ASCEND_TENSOR_COMPILER_INCLUDE}" STREQUAL "x")
    set(ASCEND_TENSOR_COMPILER_INCLUDE $ENV{ASCEND_TENSOR_COMPILER_INCLUDE})
    set(ASCEND_COMPILER_LIB ${ASCEND_TENSOR_COMPILER_INCLUDE}/../lib64)
  else()
    if(DEFINED ENV{ASCEND_CUSTOM_PATH})
      set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
      set(ASCEND_PATH /usr/local/Ascend)
    endif()
    set(ASCEND_TENSOR_COMPILER_INCLUDE ${ASCEND_PATH}/latest/include)
    set(ASCEND_COMPILER_LIB ${ASCEND_PATH}/latest/lib64)
  endif()
endif()
message(
  STATUS "ASCEND_TENSOR_COMPILER_INCLUDE=${ASCEND_TENSOR_COMPILER_INCLUDE}")

set(ASCEND_INC ${ASCEND_TENSOR_COMPILER_INCLUDE})

if(UNIX)
  if(NOT DEFINED SYSTEM_INFO)
    if(NOT "x$ENV{SYSTEM_INFO}" STREQUAL "x")
      set(SYSTEM_INFO $ENV{SYSTEM_INFO})
    else()
      execute_process(COMMAND grep -i ^id= /etc/os-release
                      OUTPUT_VARIABLE SYSTEM_NAME_INFO)
      string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${SYSTEM_NAME_INFO})
      message(STATUS "SYSTEM_NAME=${SYSTEM_NAME}")
      set(SYSTEM_INFO ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR})
    endif()
  endif()

  message(STATUS "SYSTEM_INFO=${SYSTEM_INFO}")
elseif(WIN32)
  message(STATUS "System is Windows. Only for pre-build.")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
endif()

set(RUN_TARGET "custom_opp_${SYSTEM_INFO}.run")
message(STATUS "RUN_TARGET=${RUN_TARGET}")
set(VENDOR_NAME "custom_aicore_ops")
set(PROJECT_DIR "vendors/${VENDOR_NAME}")

set(OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "OUT_DIR=${OUT_DIR}")

set(OP_PROTO_TARGET "op_proto")
set(OP_PROTO_TARGET_OUT_DIR ${OUT_DIR}/${PROJECT_DIR}/op_proto/)

set(AIC_OP_INFO_CFG_OUT_DIR ${OUT_DIR}/${PROJECT_DIR}/op_impl/ai_core/tbe/config)
set(AIV_OP_INFO_CFG_OUT_DIR ${OUT_DIR}/${PROJECT_DIR}/op_impl/vector_core/tbe/config/)

set(OPS_DIR ${CMAKE_SOURCE_DIR}/mindspore/lite/tools/kernel_builder/ascend/tbe_and_aicpu/)
set(INI_2_JSON_PY "${OPS_DIR}/cmake/util/parse_ini_to_json.py")

set(CMAKE_SKIP_RPATH TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all")

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,relro,-z,now")
set(CMAKE_EXECUTABLE_LINKER_FLAGS
    "${CMAKE_EXECUTABLE_LINKER_FLAGS} -Wl,-z,relro,-z,now")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack")
set(CMAKE_EXECUTABLE_LINKER_FLAGS
    "${CMAKE_EXECUTABLE_LINKER_FLAGS} -Wl,-z,noexecstack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpie")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pie")
set(CMAKE_EXECUTABLE_LINKER_FLAGS "${CMAKE_EXECUTABLE_LINKER_FLAGS} -pie")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2 -O2")

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -fvisibility=hidden")
set(CMAKE_EXECUTABLE_LINKER_FLAGS
    "${CMAKE_EXECUTABLE_LINKER_FLAGS} -fvisibility=hidden")
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -fvisibility-inlines-hidden")
set(CMAKE_EXECUTABLE_LINKER_FLAGS
    "${CMAKE_EXECUTABLE_LINKER_FLAGS} -fvisibility-inlines-hidden")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -ftrapv")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fstack-check")

set(CMAKE_C_FLAGS_DEBUG
    "-fPIC -pthread -Wfloat-equal -Wshadow -Wformat=2 -Wno-deprecated -fstack-protector-strong -Wall -Wextra"
)

include_directories(${ASCEND_INC})
