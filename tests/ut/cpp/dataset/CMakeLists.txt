include(GoogleTest)

SET(DE_UT_SRCS
        affine_op_test.cc
        execute_test.cc
        album_op_test.cc
        arena_test.cc
        auto_contrast_op_test.cc
        batch_op_test.cc
        bit_functions_test.cc
        bounding_box_augment_op_test.cc
        btree_test.cc
        buddy_test.cc
        build_vocab_test.cc
        c_api_cache_test.cc
        c_api_dataset_album_test.cc
        c_api_dataset_cifar_test.cc
        c_api_dataset_clue_test.cc
        c_api_dataset_coco_test.cc
        c_api_dataset_config_test.cc
        c_api_dataset_csv_test.cc
        c_api_dataset_iterator_test.cc
        c_api_dataset_manifest_test.cc
        c_api_dataset_minddata_test.cc
        c_api_dataset_ops_test.cc
        c_api_dataset_randomdata_test.cc
        c_api_dataset_save.cc
        c_api_dataset_textfile_test.cc
        c_api_dataset_tfrecord_test.cc
        c_api_dataset_voc_test.cc
        c_api_datasets_test.cc
        c_api_epoch_ctrl_test.cc
        c_api_pull_based_test.cc
        c_api_repeat_test.cc
        c_api_samplers_test.cc
        c_api_text_sentence_piece_vocab_test.cc
        c_api_text_vocab_test.cc
        c_api_transforms_test.cc
        c_api_vision_a_to_q_test.cc
        c_api_vision_affine_test.cc
        c_api_vision_bounding_box_augment_test.cc
        c_api_vision_random_subselect_policy_test.cc
        c_api_vision_random_test.cc
        c_api_vision_r_to_z_test.cc
        c_api_vision_soft_dvpp_test.cc
        c_api_vision_uniform_aug_test.cc
        celeba_op_test.cc
        center_crop_op_test.cc
        channel_swap_test.cc
        cifar_op_test.cc
        circular_pool_test.cc
        client_config_test.cc
        clue_op_test.cc
        coco_op_test.cc
        common/bboxop_common.cc
        common/common.cc
        common/cvop_common.cc
        concat_op_test.cc
        concatenate_op_test.cc
        connector_test.cc
        csv_op_test.cc
        cut_out_op_test.cc
        cutmix_batch_op_test.cc
        cyclic_array_test.cc
        data_helper_test.cc
        datatype_test.cc
        decode_op_test.cc
        distributed_sampler_test.cc
        equalize_op_test.cc
        execution_tree_test.cc
        fill_op_test.cc
        global_context_test.cc
        gnn_graph_test.cc
        image_folder_op_test.cc
        image_process_test.cc
        interrupt_test.cc
        ir_callback_test.cc
        ir_sampler_test.cc
        ir_tensor_op_fusion_pass_test.cc
        ir_tree_adapter_test.cc
        ir_vision_random_test.cc
        ir_vision_test.cc
        jieba_tokenizer_op_test.cc
        main_test.cc
        map_op_test.cc
        mask_test.cc
        memory_pool_test.cc
        mind_record_op_test.cc
        mixup_batch_op_test.cc
        mnist_op_test.cc
        normalize_op_test.cc
        one_hot_op_test.cc
        optimization_pass_test.cc
        pad_end_op_test.cc
        pad_op_test.cc
        path_test.cc
        perf_data_test.cc
        project_op_test.cc
        queue_test.cc
        random_affine_op_test.cc
        random_color_adjust_op_test.cc
        random_color_op_test.cc
        random_crop_and_resize_op_test.cc
        random_crop_and_resize_with_bbox_op_test.cc
        random_crop_decode_resize_op_test.cc
        random_crop_op_test.cc
        random_crop_with_bbox_op_test.cc
        random_horizontal_flip_op_test.cc
        random_horizontal_flip_with_bbox_test.cc
        random_resize_op_test.cc
        random_resize_op_test.cc
        random_resize_with_bbox_op_test.cc
        random_rotation_op_test.cc
        random_solarize_op_test.cc
        random_vertical_flip_op_test.cc
        random_vertical_flip_with_bbox_op_test.cc
        rename_op_test.cc
        rescale_op_test.cc
        resize_op_test.cc
        resize_with_bbox_op_test.cc
        rgba_to_bgr_op_test.cc
        rgba_to_rgb_op_test.cc
        schema_test.cc
        sentence_piece_vocab_op_test.cc
        shuffle_op_test.cc
        skip_op_test.cc
        slice_op_test.cc
        sliding_window_op_test.cc
        solarize_op_test.cc
        stand_alone_samplers_test.cc
        status_test.cc
        storage_container_test.cc
        subset_random_sampler_test.cc
        subset_sampler_test.cc
        swap_red_blue_test.cc
        take_op_test.cc
        task_manager_test.cc
        tensor_row_test.cc
        tensor_string_test.cc
        tensor_test.cc
        tensorshape_test.cc
        text_file_op_test.cc
        tfReader_op_test.cc
        to_float16_op_test.cc
        tokenizer_op_test.cc
        treap_test.cc
        tree_modifying_function_test.cc
        trucate_pair_test.cc
        type_cast_op_test.cc
        weighted_random_sampler_test.cc
        zip_op_test.cc
        )

if(ENABLE_PYTHON)
    set(DE_UT_SRCS
            ${DE_UT_SRCS}
            filter_op_test.cc
            manifest_op_test.cc
            voc_op_test.cc
            sentence_piece_vocab_op_test.cc
            )
endif()

if(ENABLE_ACL)
    set(DE_UT_SRCS
            ${DE_UT_SRCS}
            dvpp_decode_jpeg_test.cc)
endif()

add_executable(de_ut_tests ${DE_UT_SRCS})

set_target_properties(de_ut_tests PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib64")

target_link_libraries(de_ut_tests PRIVATE
        _c_dataengine pybind11::embed
        ${GTEST_LIBRARY}
        ${SECUREC_LIBRARY}
        ${SLOG_LIBRARY}
        )

gtest_discover_tests(de_ut_tests WORKING_DIRECTORY ${Project_DIR}/tests/dataset)

install(TARGETS de_ut_tests
        RUNTIME DESTINATION test)

# For internal testing only.
install(DIRECTORY ${Project_DIR}/tests/dataset/data/
        DESTINATION test/data)
