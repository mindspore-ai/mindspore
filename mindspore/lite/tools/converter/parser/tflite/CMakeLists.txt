file(GLOB_RECURSE TFLITE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
ADD_DEFINITIONS(-DFLATBUFFERS_LOCALE_INDEPENDENT=1)
find_library(FLATBUFFERS_LIBRARY flatbuffers HINTS ${TOP_DIR}/third_party/flatbuffers/build)
add_library(tflite_parser_mid OBJECT
        ${TFLITE_SRC_LIST}
        )
target_link_libraries(tflite_parser_mid ${FLATBUFFERS_LIBRARY})
