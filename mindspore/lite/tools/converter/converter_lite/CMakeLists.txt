remove_definitions(-DUSE_GLOG)

link_directories(${opencv_INC}/../lib)

add_executable(converter_lite main.cc converter_flags.cc
              ${TOP_DIR}/mindspore/lite/src/common/log.cc
              ${TOP_DIR}/mindspore/lite/src/common/utils.cc
              ${TOP_DIR}/mindspore/core/utils/status.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/../../common/flag_parser.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/../../common/string_util.cc
              ${TOP_DIR}/mindspore/lite/src/common/file_utils.cc)

target_link_libraries(converter_lite mindspore_converter)
