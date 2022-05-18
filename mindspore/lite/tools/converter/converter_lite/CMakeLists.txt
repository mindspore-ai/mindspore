remove_definitions(-DUSE_GLOG)

add_executable(converter_lite main.cc converter_flags.cc
              ${TOP_DIR}/mindspore/lite/src/common/log.cc
              ${TOP_DIR}/mindspore/core/utils/status.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/../../common/flag_parser.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/../../common/string_util.cc)
target_link_libraries(converter_lite PRIVATE mindspore-converter)
