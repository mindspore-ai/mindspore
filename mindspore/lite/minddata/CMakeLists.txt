set(MINDDATA_DIR ${CCSRC_DIR}/minddata/dataset)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS} -s")

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/core MINDDATA_CORE_SRC_FILES)
list(REMOVE_ITEM MINDDATA_CORE_SRC_FILES "${MINDDATA_DIR}/core/client.cc")

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels MINDDATA_KERNELS_SRC_FILES)
list(REMOVE_ITEM MINDDATA_KERNELS_SRC_FILES "${MINDDATA_DIR}/kernels/py_func_op.cc")

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels/image MINDDATA_KERNELS_IMAGE_SRC_FILES)

AUX_SOURCE_DIRECTORY(${MINDDATA_DIR}/kernels/data MINDDATA_KERNELS_DATA_SRC_FILES)

add_library(minddata-eager OBJECT 
        ${MINDDATA_DIR}/api/de_tensor.cc
        ${MINDDATA_DIR}/api/execute.cc
        )

add_library(minddata-lite SHARED
        ${MINDDATA_CORE_SRC_FILES}
        ${MINDDATA_KERNELS_SRC_FILES}
        ${MINDDATA_KERNELS_IMAGE_SRC_FILES}
        ${MINDDATA_KERNELS_DATA_SRC_FILES}
        ${MINDDATA_DIR}/util/status.cc
        ${MINDDATA_DIR}/util/memory_pool.cc
        ${MINDDATA_DIR}/util/path.cc
        ${MINDDATA_DIR}/api/transforms.cc
        ${CORE_DIR}/utils/log_adapter.cc
        ${CORE_DIR}/gvar/logging_level.cc
        )

target_link_libraries(minddata-lite
        securec
        jpeg-turbo
        jpeg
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        mindspore::json
        )