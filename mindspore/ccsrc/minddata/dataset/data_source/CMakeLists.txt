add_subdirectory(operation)
add_subdirectory(sampler)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
    ag_news_op.cc
    album_op.cc
    amazon_review_op.cc
    clue_op.cc
    conll2000_op.cc
    csv_op.cc
    dbpedia_op.cc
    en_wik9_op.cc
    image_folder_op.cc
    imdb_op.cc
    iwslt_op.cc
    io_block.cc
    mappable_leaf_op.cc
    multi30k_op.cc
    nonmappable_leaf_op.cc
    penn_treebank_op.cc
    random_data_op.cc
    sogou_news_op.cc
    squad_op.cc
    sst2_op.cc
    text_file_op.cc
    udpos_op.cc
    wiki_text_op.cc
    yahoo_answers_op.cc
    yelp_review_op.cc
    )

set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
    ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
    mindrecord_op.cc
    tf_reader_op.cc
    )

if(ENABLE_PYTHON)
    set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
        ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
        generator_op.cc
        manifest_op.cc
        )
endif()

add_library(data_source OBJECT ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES})
