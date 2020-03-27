file(GLOB_RECURSE _UTILS_ALL_SRC_FILES *.cc)
#TODO :  "utils/node_utils.cc" 
add_library(_mindspore_utils_obj OBJECT ${_UTILS_ALL_SRC_FILES})
