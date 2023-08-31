if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python3")
else()
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# generate operation definition code, include python/mindspore/ops/auto_generate/gen_ops_def.py
# and core/ops/ops_generate/gen_ops_def.cc
execute_process(COMMAND "${PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_ops.py")
execute_process(COMMAND "${PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_enum.py")

add_custom_target(generated_code DEPENDS
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops/auto_generate/gen_ops_def.py"
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops/auto_generate/gen_labels.py"
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops/auto_generate/arg_handler.py"
        "${CMAKE_SOURCE_DIR}/mindspore/core/ops/gen_ops_def.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/core/ops/gen_lite_ops.h"
        "${CMAKE_SOURCE_DIR}/mindspore/core/ops/gen_ops_name.h"
        "${CMAKE_SOURCE_DIR}/mindspore/core/ops/gen_ops_primitive.h")
