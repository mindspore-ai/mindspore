if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python")
else()
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# generate operation definition code, include python/mindspore/ops/auto_generate/gen_ops_def.py
# and core/ops/ops_generate/gen_ops_def.cc
execute_process(COMMAND "${PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/python/mindspore/ops_generate/gen_ops.py"
        RESULT_VARIABLE STATUS)
if(NOT STATUS EQUAL "0")
    message(FATAL_ERROR "Generate operator python/c++ definitions FAILED.")
else()
    message("Generate operator python/c++ definitions SUCCESS!")
endif()

add_custom_target(generated_code DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/python/mindspore/ops/auto_generate/gen_ops_def.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/python/mindspore/ops/auto_generate/gen_arg_handler.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/python/mindspore/ops/auto_generate/gen_arg_dtype_cast.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/python/mindspore/ops/auto_generate/cpp_create_prim_instance_helper.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/core/ops/auto_generate/gen_ops_def.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/core/ops/auto_generate/gen_lite_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/core/ops/auto_generate/gen_lite_ops.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/core/ops/auto_generate/gen_ops_name.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/core/ops/auto_generate/gen_ops_primitive.h")
