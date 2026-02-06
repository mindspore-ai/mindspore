include(${OPS_DIR}/cmake/merge_ops.cmake)

file(GLOB_RECURSE PYBOOST_SRC_LIST "*.cc")

# merge customize pyboost kernel files
file(GLOB_RECURSE PYBOOST_MERGE_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/customize/*.cc)
list(REMOVE_ITEM PYBOOST_SRC_LIST ${PYBOOST_MERGE_SRC_LIST})

set(ASCEND_PYBOOST_MERGE_DIR ${CMAKE_BINARY_DIR}/merge/mindspore/ops/kernel/ascend/aclnn/pyboost_impl)
merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/customize ${ASCEND_PYBOOST_MERGE_DIR} ascend_pyboost_merge "")
file(GLOB_RECURSE ASCEND_PYBOOST_MERGE_FILES ${ASCEND_PYBOOST_MERGE_DIR}/*.cc)

set(PYBOOST_IMPL_SRCS ${PYBOOST_SRC_LIST} ${ASCEND_PYBOOST_MERGE_FILES} PARENT_SCOPE)
