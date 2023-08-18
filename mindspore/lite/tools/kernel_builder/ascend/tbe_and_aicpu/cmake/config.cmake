# set compile option -std=c++11
set(CMAKE_CXX_STANDARD 11)

# set compile option -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(OPP_CUSTOM_VENDOR "$ENV{OPP_CUSTOM_VENDOR}")
# set(TOP_DIR ${CMAKE_SOURCE_DIR}/../..)

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
set(PROJECT_DIR "vendors/mslite_tbe_and_aicpu")

set(OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/makepkg)
message(STATUS "OUT_DIR=${OUT_DIR}")

set(TF_PLUGIN_TARGET "cust_tf_parsers")
set(TF_PLUGIN_TARGET_OUT_DIR
    ${OUT_DIR}/packages/${PROJECT_DIR}/framework/tensorflow/)

set(OP_PROTO_TARGET "cust_op_proto")
set(OP_PROTO_TARGET_OUT_DIR ${OUT_DIR}/packages/${PROJECT_DIR}/op_proto/)

set(AIC_FUSION_PASS_TARGET "cust_aic_fusion_pass")
set(AIC_FUSION_PASS_TARGET_OUT_DIR
    ${OUT_DIR}/packages/${PROJECT_DIR}/fusion_pass/ai_core)

set(AIV_FUSION_PASS_TARGET "cust_aiv_fusion_pass")
set(AIV_FUSION_PASS_TARGET_OUT_DIR
    ${OUT_DIR}/packages/${PROJECT_DIR}/fusion_pass/vector_core)

set(AIC_OP_INFO_CFG_OUT_DIR
    ${OUT_DIR}/packages/${PROJECT_DIR}/op_impl/ai_core/tbe/config)
set(AIV_OP_INFO_CFG_OUT_DIR
    ${OUT_DIR}/packages/${PROJECT_DIR}/op_impl/vector_core/tbe/config/)

set(AICPU_CONFIG_JSON_TARGET "aicpu_config_json")
set(AICPU_OP_INFO_CFG_OUT_DIR
    ${OUT_DIR}/packages/${PROJECT_DIR}/op_impl/cpu/config)
set(AICPU_OP_IMPL_OUT_DIR
    ${OUT_DIR}/packages/${PROJECT_DIR}/op_impl/cpu/aicpu_kernel/impl/)

set(INI_2_JSON_PY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/parse_ini_to_json.py")
set(AICPU_INI_2_JSON_PY
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/aicpu_parser_ini.py")

set(AICPU_KERNEL_TARGET $ENV{AICPU_KERNEL_TARGET})

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
if("x${AICPU_KERNEL_TARGET}" STREQUAL "x")
  set(AICPU_KERNEL_TARGET "cust_aicpu_kernels")
endif()

include_directories(${ASCEND_INC})
