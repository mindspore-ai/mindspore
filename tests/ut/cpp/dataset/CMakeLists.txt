include(GoogleTest)

SET(DE_UT_SRCS
    common/common.cc
    common/cvop_common.cc
    batch_op_test.cc
    bit_functions_test.cc
    storage_container_test.cc
    treap_test.cc
    interrupt_test.cc
    image_folder_op_test.cc
    buddy_test.cc
    arena_test.cc
    btree_test.cc
    center_crop_op_test.cc
    channel_swap_test.cc
    circular_pool_test.cc
    client_config_test.cc
    connector_test.cc
    datatype_test.cc
    decode_op_test.cc
    execution_tree_test.cc
    global_context_test.cc
    main_test.cc
    map_op_test.cc
    mind_record_op_test.cc
    memory_pool_test.cc
    normalize_op_test.cc
    one_hot_op_test.cc
    path_test.cc
    project_op_test.cc
    queue_test.cc
    random_crop_op_test.cc
    random_crop_decode_resize_op_test.cc
    random_crop_and_resize_op_test.cc
    random_color_adjust_op_test.cc
    random_horizontal_flip_op_test.cc
    random_resize_op_test.cc
    random_rotation_op_test.cc
    random_vertical_flip_op_test.cc
    rename_op_test.cc
    repeat_op_test.cc
    skip_op_test.cc
    rescale_op_test.cc
    resize_bilinear_op_test.cc
    resize_op_test.cc
    shuffle_op_test.cc
    stand_alone_samplers_test.cc
    status_test.cc
    storage_op_test.cc
    task_manager_test.cc
    tensor_test.cc
    tensor_string_test.cc
    tensorshape_test.cc
    tfReader_op_test.cc
    to_float16_op_test.cc
    type_cast_op_test.cc
    zip_op_test.cc
    random_resize_op_test.cc
    subset_random_sampler_test.cc
    weighted_random_sampler_test.cc
    mnist_op_test.cc
    manifest_op_test.cc
    voc_op_test.cc
    cifar_op_test.cc
    celeba_op_test.cc
    take_op_test.cc
    text_file_op_test.cc
    filter_op_test.cc
    concat_op_test.cc
    jieba_tokenizer_op_test.cc
    tokenizer_op_test.cc
    gnn_graph_test.cc
    coco_op_test.cc
    fill_op_test.cc
    )

add_executable(de_ut_tests ${DE_UT_SRCS})

set_target_properties(de_ut_tests PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib64")

target_link_libraries(de_ut_tests PRIVATE _c_dataengine pybind11::embed ${GTEST_LIBRARY} ${SECUREC_LIBRARY} ${SLOG_LIBRARY})

gtest_discover_tests(de_ut_tests WORKING_DIRECTORY ${Project_DIR}/tests/dataset)

install(TARGETS de_ut_tests
    RUNTIME DESTINATION test)

# For internal testing only.
install(DIRECTORY ${Project_DIR}/tests/dataset/data/
    DESTINATION test/data)
