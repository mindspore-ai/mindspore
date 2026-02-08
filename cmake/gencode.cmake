if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(PYTHON_EXECUTABLE "python")
else()
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# generate operation definition code, include python/mindspore/ops/auto_generate/gen_ops_def.py
# and ops/op_def/auto_generate/gen_ops_def.cc (source files)
# and ops/include/primitive/auto_generate/gen_ops_def.h (header files)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/python/mindspore/ops/auto_generate/cpp_create_prim_instance_helper.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/op_def/auto_generate/gen_deprecated_ops_def.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/op_def/auto_generate/gen_lite_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_lite_ops.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_a.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_b.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_c.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_d.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_e.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_f.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_g.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_h.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_i.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_k.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_l.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_m.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_n.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_o.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_p.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_q.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_r.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_s.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_t.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_u.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_v.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_w.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_x.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_name_z.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_a.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_b.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_c.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_d.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_e.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_f.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_g.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_h.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_i.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_k.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_l.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_m.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_n.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_o.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_p.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_q.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_r.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_s.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_t.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_u.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_v.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_w.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_x.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_primitive_z.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore/ops/include/primitive/auto_generate/gen_ops_def.h")
