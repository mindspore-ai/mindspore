
if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python3")
else()
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

set(yaml_file "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops.yaml")
#generate operation definition code, include python/mindspore/gen_ops_def.py and core/ops/gen_ops_def.cc
add_custom_command(
                OUTPUT "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/gen_ops_def.py"
                "${CMAKE_SOURCE_DIR}/mindspore/core/ops/gen_ops_def.cc"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND "${PYTHON_EXECUTABLE}"
                "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/gen_ops.py" ${CMAKE_SOURCE_DIR}
                DEPENDS ${yaml_file}
                COMMENT "Generate operator python/c++ definitions" VERBATIM)
add_custom_target(generated_code DEPENDS
            "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/gen_ops_def.py"
            "${CMAKE_SOURCE_DIR}/mindspore/core/ops/gen_ops_def.cc")