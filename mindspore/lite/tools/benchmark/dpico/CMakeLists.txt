cmake_minimum_required(VERSION 3.14)
project(DPICO_Custom)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cmake/utils.cmake)
__download_pkg(34xx_sdk
        http://mindspore-repo.csi.rnd.huawei.com/mindspore/enterprise/dpico/34xx_sdk.tar.gz
        f64a9129615b3b41b63debe17c6785af)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${34xx_sdk_SOURCE_DIR})
include_directories(${34xx_sdk_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/runtime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/runtime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/runtime/include/third_party)
link_directories(${34xx_sdk_SOURCE_DIR}/lib)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src COMMON_SRC3)

set(LINK_LOCAT_LIB ${34xx_sdk_SOURCE_DIR}/lib/libsvp_acl.so
        pthread ${34xx_sdk_SOURCE_DIR}/lib/libsecurec.so dl
        ${34xx_sdk_SOURCE_DIR}/lib/libprotobuf-c.so.1 stdc++)

add_library(dpico_acl_adapter SHARED
        ${COMMON_SRC3})
target_link_libraries(dpico_acl_adapter ${LINK_LOCAT_LIB} securec)