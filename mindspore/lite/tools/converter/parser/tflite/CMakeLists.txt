file(GLOB_RECURSE TFLITE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
set_property(SOURCE ${TFLITE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
if (NOT WIN32)
    ADD_DEFINITIONS(-DFLATBUFFERS_LOCALE_INDEPENDENT=1)
    find_library(FLATBUFFERS_LIBRARY flatbuffers HINTS ${TOP_DIR}/third_party/flatbuffers/build)
endif ()
add_library(tflite_parser_mid OBJECT
        ${TFLITE_SRC_LIST}
        )
if (WIN32)
    target_link_libraries(tflite_parser_mid mindspore::flatbuffers)
else ()
    target_link_libraries(tflite_parser_mid ${FLATBUFFERS_LIBRARY})
endif ()
