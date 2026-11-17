file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_subdirectory(samplers)

set(DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES
        ag_news_node.cc
        album_node.cc
        amazon_review_node.cc
        clue_node.cc
        conll2000_node.cc
        csv_node.cc
        dbpedia_node.cc
        en_wik9_node.cc
        image_folder_node.cc
        imdb_node.cc
        iwslt2016_node.cc
        iwslt2017_node.cc
        manifest_node.cc
        minddata_node.cc
        multi30k_node.cc
        penn_treebank_node.cc
        random_node.cc
        sogou_news_node.cc
        squad_node.cc
        sst2_node.cc
        text_file_node.cc
        tf_record_node.cc
        udpos_node.cc
        wiki_text_node.cc
        yahoo_answers_node.cc
        yelp_review_node.cc
        )

if(ENABLE_PYTHON)
    set(DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES
            ${DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES}
            generator_node.cc
            )
endif()

add_library(data_source-operation OBJECT ${DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES})
