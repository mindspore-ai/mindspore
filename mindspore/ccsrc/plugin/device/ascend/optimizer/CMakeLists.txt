if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_D OR ENABLE_ACL)
    file(GLOB_RECURSE _PREACTIVATE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
    )
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

file(GLOB_RECURSE MS_OPTIMIZER_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./mindir/*.cc"
        "./ge/*.cc"
        "./ge_backend_optimization.cc"
        "./ge_optimization.cc"
        "./backend_common_unify_mindir.cc"
        "./format_type/insert_cast.cc"
        "./format_type/set_fracz_group_attr.cc"
        "./format_type/utils.cc"
        "./format_type/deal_ref_output.cc"
        "./ir_fission/cdist_fission.cc"
        "./ir_fission/bn_split.cc"
        "./ir_fission/bn_grad_split.cc"
        "./ir_fission/seed_adapter.cc"
        "./ir_fission/tensor_scatter_fission.cc"
        "./ir_fission/adam_weight_decay_fission.cc"
        "./ir_fission/batch_norm_grad_infer_fission.cc"
        "./ir_fission/bn_split.cc"
        "./ir_fission/bn_grad_split.cc"
        "./ir_fusion/batchnorm_to_bninfer.cc"
        "./ir_fission/ascend_convert_tuple_input_to_dynamic_input.cc"
        "./ir_fusion/batchnormgrad_to_bninfergrad.cc"
        "./ir_fusion/histogram_fixed_width_fusion.cc"
        "./ir_fusion/adaptive_max_pool2d_fusion.cc"
        "./enhancer/add_placeholder_for_dynamic_rnn.cc"
        "./enhancer/add_placeholder_for_dynamic_gru.cc"
        )
list(REMOVE_ITEM MS_OPTIMIZER_910B
        "mindir/ascend_vm_op_adapter.cc"
        "mindir/conv2d_unify_mindir.cc"
        "ir_fission/ascend_clip_by_norm_fission.cc"
        "ir_fission/batch_norm_bert_fission.cc"
        "ir_fission/batch_norm_grad_split.cc"
        "ir_fusion/fused_batch_norm_fusion.cc"
        "ir_fusion/stateless_dropout_genmask_replace.cc"
        "ge/dropout_for_ge.cc"
        "ge/convert_attr_to_input.cc"
        "ge/convert_resize_nearest_neighbor_x_dtype.cc"
        "ge/batchnorm_transform.cc"
        "ge/avg_pool_grad_for_ge.cc"
        )

set_property(SOURCE ${_PREACTIVATE_SRC_LIST} ${MS_OPTIMIZER_910B} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_PRE_ACT)

if(ENABLE_D OR ENABLE_ACL)
    add_library(_mindspore_plugin_device_ascend_optimizer_910b_obj OBJECT ${MS_OPTIMIZER_910B})
    add_library(_mindspore_plugin_device_ascend_optimizer_obj OBJECT ${_PREACTIVATE_SRC_LIST})
endif()
