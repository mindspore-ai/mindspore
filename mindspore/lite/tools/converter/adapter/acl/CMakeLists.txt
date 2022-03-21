include_directories(${TOP_DIR}/graphengine/metadef/inc/external)
include_directories(${TOP_DIR}/graphengine/inc)
include_directories(${TOP_DIR}/graphengine/inc/external)
include_directories(${TOP_DIR}/graphengine/ge)
include_directories(${TOP_DIR}/graphengine/metadef/inc)
include_directories(${TOP_DIR}/graphengine/inc/framework)
include_directories(${TOP_DIR}/graphengine/third_party/fwkacllib/inc)
include_directories(${TOP_DIR}/graphengine/third_party/fwkacllib/inc/aicpu)
include_directories(${TOP_DIR}/graphengine/third_party/fwkacllib/inc/toolchain)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    add_compile_definitions(ENABLE_ONLINE_MODEL_INFER)
endif()

file(GLOB ACL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mapper/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cc
        )

set(ENABLE_ACL on)
set(MODE_ASCEND_ACL off)

add_subdirectory(${TOP_DIR}/mindspore/ccsrc/transform/graph_ir _mindspore_transform_graph_ir_obj)
add_subdirectory(${TOP_DIR}/mindspore/ccsrc/cxx_api mindspore_shared_lib)

set_property(SOURCE ${ACL_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(lite_acl_mid OBJECT ${ACL_SRC})
target_link_libraries(lite_acl_mid mindspore_shared_lib)
add_dependencies(lite_acl_mid fbs_inner_src)

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "cloud")
    set(MSLITE_ENABLE_CLOUD_MIND_DATA ON)
    # find python
    include(${TOP_DIR}/mindspore/lite/cmake/python.cmake)
    if(Python3_FOUND)
        set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
        set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
    else()
        find_python_package(py_inc py_lib)
        set(PYTHON_INCLUDE_DIRS "${py_inc}")
        set(PYTHON_LIBRARIES "${py_lib}")
    endif()
    message("Python version: ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
    message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    # MindData
    add_compile_definitions(ENABLE_LITE_TENSOR)
    add_compile_definitions(ENABLE_PYTHON)
    set(ENABLE_PYTHON ON)

    include(${TOP_DIR}/cmake/external_libs/sentencepiece.cmake)
    include(${TOP_DIR}/cmake/external_libs/jpeg_turbo.cmake)
    include(${TOP_DIR}/cmake/external_libs/tinyxml2.cmake)
    include(${TOP_DIR}/cmake/external_libs/icu4c.cmake)
    include(${TOP_DIR}/cmake/external_libs/sqlite.cmake)
    include(${TOP_DIR}/cmake/external_libs/pybind11.cmake)
    include(${TOP_DIR}/cmake/external_libs/cppjieba.cmake)
    include(${TOP_DIR}/cmake/external_libs/re2.cmake)
    include(${TOP_DIR}/cmake/external_libs/c-ares.cmake)
    include(${TOP_DIR}/cmake/external_libs/absl.cmake)
    include(${TOP_DIR}/cmake/external_libs/grpc.cmake)

    set(MINDDATA_DIR ${CCSRC_DIR}/minddata/dataset)
    include_directories("${MINDDATA_DIR}/kernels/image")

    add_subdirectory(${CCSRC_DIR}/minddata/mindrecord _c_mindrecord)
    add_subdirectory(${CCSRC_DIR}/minddata/dataset _c_dataengine)
endif()
