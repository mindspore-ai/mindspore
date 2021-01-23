file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
if(NOT (CMAKE_SYSTEM_NAME MATCHES "Windows"))
        set(ICU_DEPEND_FILES
                basic_tokenizer_op.cc
                bert_tokenizer_op.cc
                case_fold_op.cc
                normalize_utf8_op.cc
                regex_replace_op.cc
                regex_tokenizer_op.cc
                unicode_script_tokenizer_op.cc
                whitespace_tokenizer_op.cc)
endif()
add_library(text-kernels OBJECT
        data_utils.cc
        lookup_op.cc
        jieba_tokenizer_op.cc
        unicode_char_tokenizer_op.cc
        ngram_op.cc
        sliding_window_op.cc
        wordpiece_tokenizer_op.cc
        truncate_sequence_pair_op.cc
        to_number_op.cc
        sentence_piece_tokenizer_op.cc
        ${ICU_DEPEND_FILES}
        )
