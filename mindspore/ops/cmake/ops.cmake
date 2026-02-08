# include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gencode.cmake)
set(INFER_DIR "infer")
set(VIEW_DIR "view")
set(UTILS_DIR "ops_utils")

# ------- compiler flags ------
if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(NOT "${CMAKE_C_FLAGS}" STREQUAL "")
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-exceptions" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()
if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

# ------- CORE_OPS_LIST, CORE_SYMBOL_OPS_LIST ------
include(${OPS_DIR}/cmake/merge_ops.cmake)
if(CMAKE_SIZEOF_VOID_P EQUAL 4 OR NOT BUILD_LITE)
    file(GLOB_RECURSE INFER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${INFER_DIR}/*.cc")

    # merge infer/ops_func_impl/*.cc and infer/*.cc(no recursive)
    file(GLOB OLD_INFER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${INFER_DIR}/*.cc")
    file(GLOB_RECURSE INFER_FUNC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${INFER_DIR}/ops_func_impl/*.cc")
    list(REMOVE_ITEM INFER_LIST ${INFER_FUNC_LIST} ${OLD_INFER_LIST})
    set(INFER_MERGE_DIR ${CMAKE_BINARY_DIR}/merge/mindspore/ops/infer)
    merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/infer/ops_func_impl/ ${INFER_MERGE_DIR} infer_func_merge "")
    merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/infer/ ${INFER_MERGE_DIR} old_infer_merge "" TRUE) # no recursive
    file(GLOB_RECURSE INFER_MERGE_LIST "${INFER_MERGE_DIR}/*.cc")

    file(GLOB_RECURSE CORE_SYMBOL_OPS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${INFER_DIR}/symbol_ops_impl/*.cc")
    file(GLOB_RECURSE VIEW_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${VIEW_DIR}/*.cc")
    file(GLOB_RECURSE UTILS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${UTILS_DIR}/*.cc")
    set_property(SOURCE ${CORE_SYMBOL_OPS_LIST} PROPERTY COMPILE_DEFINITIONS
                SUBMODULE_ID=mindspore::SubModuleId::SM_SYMBOLIC_SHAPE)

    list(APPEND CORE_OPS_LIST ${VIEW_LIST} ${UTILS_LIST} ${OP_DEF_SRC} ${INFER_LIST} ${INFER_MERGE_LIST})
else()
    # ------- LITE merge_files -----
    merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/infer/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops infer_merge "py_func.cc")
    merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/view/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops view_merge "")
    merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/ops_utils/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops utils_merge "")
    file(GLOB_RECURSE CORE_OPS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_BINARY_DIR}/merge/mindspore/ops/*.cc")
    list(APPEND CORE_OPS_LIST ${OP_DEF_SRC})
endif()



# -------- CORE_SRC_LIST, SYMBOLIC_SHAPE_SRC_LIST ----------
if(TARGET_AOS_ARM)
    string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-Werror" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()

if(ANDROID_NDK)
    add_definitions(-w)
    add_compile_definitions(KERNEL_EXECUTOR_ANDROID)
    set(TARGET_AOS_ARM ON)
endif()

# ------------------ COMPILE_DEFINITIONS -------------------

if(NOT MSLITE_TARGET_SITEAI)
    set(MSLITE_DEPS_OPENSSL on)
    set(MSLITE_SIMPLEST_CLOUD_INFERENCE off)
endif()

if(MSLITE_DEPS_OPENSSL)
    add_compile_definitions(ENABLE_OPENSSL)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
  endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wno-overloaded-virtual -Wno-user-defined-warnings -Winconsistent-missing-override -Wno-delete-non-virtual-dtor \
    -Wno-unused-private-field")
endif()

# ------------------- COMPILE, ADD_TARGET ---------------------
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mlong-calls" SUPPORT_MLONG_CALLS)
set(OPS_OBJECT_COUNT 1)
src_separate_compile(
    OBJECT_NAME ops_obj
    OBJECT_SIZE OPS_OBJECT_COUNT
    SRC_LIST ${CORE_OPS_LIST})
foreach(number RANGE 1 ${OPS_OBJECT_COUNT})
    list(APPEND OPS_OBJECT_LIST $<TARGET_OBJECTS:ops_obj_${number}>)
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(ops_obj_${number} PRIVATE OPS_DLL)
    endif()
    if(SUPPORT_MLONG_CALLS AND (${CMAKE_BUILD_TYPE} MATCHES "Debug"))
        target_compile_options(ops_obj_${number} PRIVATE -mlong-calls)
    endif()
endforeach()

set(OPS_OBJECT_COUNT "${OPS_OBJECT_COUNT}" PARENT_SCOPE)
add_library(mindspore_ops SHARED ${OPS_OBJECT_LIST})
add_dependencies(mindspore_ops generated_code)

if(SUPPORT_MLONG_CALLS AND (${CMAKE_BUILD_TYPE} MATCHES "Debug"))
    target_compile_options(mindspore_ops PRIVATE -mlong-calls)
endif()

# ------------------ LINK, SET_PROPERTY ---------------

target_link_libraries(mindspore_ops PRIVATE mindspore::securec)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(mindspore_ops PRIVATE -Wl,-init,mindspore_log_init)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_ops PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_ops PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_ops PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

if(USE_GLOG)
    target_link_libraries(mindspore_ops PRIVATE mindspore::glog)
endif()

if((${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE) AND (NOT TARGET_AOS_ARM) AND (NOT ANDROID_NDK) AND
        (NOT MSLITE_SIMPLEST_CLOUD_INFERENCE))
    target_link_libraries(mindspore_ops PRIVATE -pthread)
endif()

if(ANDROID_NDK)
    target_link_libraries(mindspore_ops PRIVATE -llog)
endif()

target_link_libraries(mindspore_ops PRIVATE mindspore_core)
if(NOT BUILD_LITE)
    target_link_libraries(mindspore_ops PRIVATE mindspore::pybind11_module)
endif()
