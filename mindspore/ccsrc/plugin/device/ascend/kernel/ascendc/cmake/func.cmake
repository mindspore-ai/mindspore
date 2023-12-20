function(get_system_info SYSTEM_INFO)
  if(UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO}
        ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}
        PARENT_SCOPE)
  elseif(WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;INC_DIR;SEC_INC"
                        "OPS_SRC" ${ARGN})
  execute_process(
    COMMAND
      ${CMAKE_CXX_COMPILER} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC}
      -D_GLIBCXX_USE_CXX11_ABI=0 -I ${ASCEND_CANN_PACKAGE_PATH}/include
      -I ${OPBUILD_INC_DIR} -I ${OPBUILD_SEC_INC}
      -I ${TOP_DIR}/graphengine/910/metadef/inc
      -I ${TOP_DIR}/graphengine/910/inc
      -I ${TOP_DIR}/graphengine/910/inc/external
      -I ${TOP_DIR}/graphengine/910/third_party/fwkacllib
      -I ${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc
      -I ${TOP_DIR}/graphengine/910b/metadef/inc
      -I ${TOP_DIR}/graphengine/910b/inc
      -I ${TOP_DIR}/graphengine/910b/inc/external
      -I ${TOP_DIR}/graphengine/910b/third_party/fwkacllib
      -I ${TOP_DIR}/graphengine/910b/third_party/fwkacllib/inc
      -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api -o
      ${OPBUILD_OUT_DIR}/libascend_all_ops.so
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR)
  if(${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if(NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if(NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()
  set(ENV{LD_LIBRARY_PATH} "${ASCEND_CANN_PACKAGE_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")
  execute_process(
    COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
      ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR)
  if(${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

set(UTIL_DIR ${CMAKE_SOURCE_DIR}/mindspore/lite/tools/kernel_builder/ascend/ascendc/cmake/util/)
function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" ""
                        ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(
    OUTPUT ${OPINFO_OUTPUT}
    COMMAND mkdir -p ${opinfo_file_path}
    COMMAND mkdir -p ${OPINFO_INSTALL_DIR}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE}
      ${UTIL_DIR}/parse_ini_to_json.py ${OPINFO_OPS_INFO}
      ${OPINFO_OUTPUT})
  add_custom_target(${OPINFO_TARGET} ALL DEPENDS ${OPINFO_OUTPUT})
  install(FILES ${OPINFO_OUTPUT} DESTINATION ${OPINFO_INSTALL_DIR})
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  file(APPEND ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
      "${OP_TYPE},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n")
endfunction()

function(add_ops_impl_target)
  cmake_parse_arguments(
    OPIMPL "" "TARGET;OPS_INFO;IMPL_DIR;OUT_DIR;INSTALL_DIR"
    "OPS_BATCH;OPS_ITERATE" ${ARGN})
  add_custom_command(
    OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
    COMMAND mkdir -p ${OPIMPL_INSTALL_DIR}
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE}
      ${UTIL_DIR}/ascendc_impl_build.py ${OPIMPL_OPS_INFO}
      \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\" ${OPIMPL_IMPL_DIR}
      ${OPIMPL_OUT_DIR}/dynamic
      ${ASCEND_AUTOGEN_PATH}

    COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp
    DEPENDS ${OPIMPL_OPS_INFO}
            ${UTIL_DIR}/ascendc_impl_build.py)
  add_custom_target(${OPIMPL_TARGET} ALL
                    DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp)
  if(${ENABLE_SOURCE_PACKAGE})
    install(DIRECTORY ${OPIMPL_OUT_DIR}/dynamic
            DESTINATION ${OPIMPL_INSTALL_DIR})
  endif()
endfunction()
