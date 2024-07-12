if(ENABLE_D OR ENABLE_ACL)
    set(op_proto_src_gen_script "${CMAKE_CURRENT_BINARY_DIR}/op_proto/generate_op_proto.cmake")
    file(WRITE ${op_proto_src_gen_script} ""
            "get_filename_component(op_inc_file_name \${OP_PROTO_INC} NAME_WE) \n"
            "set(OP_PROTO_INCLUDE_FILE \${OP_PROTO_INC}) \n"
            "configure_file(${CMAKE_CURRENT_SOURCE_DIR}/op_declare/op_proto.cc.in \n"
            "    \${WORKSPACE_PATH}/\${op_inc_file_name}_op_proto.cc @ONLY) \n"
            )
endif()
function(op_proto_generate path c_var op_proto_include_file)
    set(${c_var})
    get_filename_component(abs_file ${op_proto_include_file} ABSOLUTE)
    get_filename_component(file_name ${op_proto_include_file} NAME_WE)
    get_filename_component(file_dir ${abs_file} PATH)
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${file_dir})

    list(APPEND ${c_var} "${path}/${file_name}_op_proto.cc")
    add_custom_command(
            OUTPUT "${path}/${file_name}_op_proto.cc"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${CMAKE_COMMAND}
                      -DOP_PROTO_INC=\"${op_proto_include_file}\"
                      -DWORKSPACE_PATH=${CMAKE_CURRENT_BINARY_DIR}/op_proto -P ${op_proto_src_gen_script}
            DEPENDS ${op_proto_include_file}
                    ${CMAKE_CURRENT_SOURCE_DIR}/op_declare/op_proto.cc.in
            COMMENT "Generating op proto source file: ${${c_var}}" VERBATIM)

    set_source_files_properties(${${c_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
endfunction()

if(ENABLE_D OR ENABLE_ACL)
    set(OPS_INC_DIR ${ASCEND_PATH}/latest/opp/built-in/op_proto/inc/)
    file(GLOB_RECURSE CUSTOM_OPS_INC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/custom_op_proto/*.h)
    message("CANN ops include path: " ${OPS_INC_DIR})
    file(GLOB_RECURSE OPS_INC_LIST ${OPS_INC_DIR}/*.h)
    # remove god.h
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/all_ops.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/nn.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/nn_math.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/tensor.h")
    # remove repeated header
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/outfeed_ops.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/nn_pooling_ops.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/nn_norm.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/selection.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/transformation.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/experiment_ops.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/flash_attention_score_proto.h")
    list(REMOVE_ITEM OPS_INC_LIST "${OPS_INC_DIR}/flash_attention_score_grad_proto.h")
    #append custom op
    list(APPEND OPS_INC_LIST ${CUSTOM_OPS_INC_LIST})
    set(OPS_PROTO_OBJECTS)
    foreach(op ${OPS_INC_LIST})
        get_filename_component(op_inc_file_name ${op} NAME_WE)
        op_proto_generate(${CMAKE_CURRENT_BINARY_DIR}/op_proto ${op_inc_file_name}_SRC ${op})
        list(APPEND OPS_PROTO_OBJECTS ${${op_inc_file_name}_SRC})
    endforeach()
    file(GLOB_RECURSE _TRANSFORM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    if(BUILD_LITE)
        list(REMOVE_ITEM _TRANSFORM_SRC_LIST "callbacks_ge.cc")
    endif()
    set_property(SOURCE ${_TRANSFORM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    add_library(_mindspore_transform_graph_ir_obj OBJECT ${_TRANSFORM_SRC_LIST} ${OPS_PROTO_OBJECTS})
endif()
