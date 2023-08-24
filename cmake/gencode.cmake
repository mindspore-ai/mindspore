set(C_GEN_OPS ${CMAKE_BINARY_DIR}/gen_ops_def.cc CACHE STRING "FILE:gen_ops_def.cc")
set(yaml_file "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/ops.yaml")
# generate operation definition code, include python/mindspore/ops_generate/gen_ops_def.py
# and core/ops/ops_generate/gen_ops_def.cc
add_custom_command(
                OUTPUT "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_ops_def.py"
                ${C_GEN_OPS}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND "${PYTHON_EXECUTABLE}"
                "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_ops.py" ${CMAKE_SOURCE_DIR}
                DEPENDS ${yaml_file} "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_ops.py"
                COMMENT "Generate operator python/c++ definitions" VERBATIM)

set(enum_yaml_file "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/enum.yaml")
add_custom_command(
        OUTPUT "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_enum_def.py"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND "${PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_enum.py" ${CMAKE_SOURCE_DIR}
        DEPENDS ${enum_yaml_file}
        COMMENT "Generate enum python definitions" VERBATIM)

add_custom_target(generated_code DEPENDS
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_enum_def.py"
        "${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate/gen_ops_def.py"
        ${C_GEN_OPS})
set_source_files_properties(${C_GEN_OPS} PROPERTIES GENERATED TRUE)
add_library(c_gen_ops OBJECT ${C_GEN_OPS})
