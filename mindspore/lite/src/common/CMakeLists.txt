set(LITE_SRC_COMMON_MID_SRC "")
set(LITE_SRC_TRAIN_COMMON_MID_SRC "")

set(LITE_SRC_COMMON_MID_SRC
        ${LITE_SRC_COMMON_MID_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/context_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mmap_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/log.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/prim_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/config_infos.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/helper/external_tensor/file_helper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/helper/external_tensor/memory_helper.cc
        )

if(NOT MSLITE_ENABLE_RUNTIME_CONVERT OR MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    set(LITE_SRC_COMMON_MID_SRC
            ${LITE_SRC_COMMON_MID_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/config_file.cc)
endif()

if(NOT ANDROID_NDK_TOOLCHAIN_INCLUDED)
    set(LITE_SRC_COMMON_MID_SRC
            ${LITE_SRC_COMMON_MID_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_library_loader.cc
            )
endif()

if(MSLITE_ENABLE_MODEL_PRE_INFERENCE)
    set(LITE_SRC_COMMON_MID_SRC
            ${LITE_SRC_COMMON_MID_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/thread_utils.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/random_data_generator.cc
            )
endif()

if(MSLITE_ENABLE_MODEL_ENCRYPTION)
    set(LITE_SRC_COMMON_MID_SRC
            ${LITE_SRC_COMMON_MID_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/decrypt.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_library_loader.cc
            )
endif()

if(MSLITE_ENABLE_STRING_KERNEL)
    set(LITE_SRC_COMMON_MID_SRC
            ${LITE_SRC_COMMON_MID_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/string_util.cc
            )
endif()

if(MSLITE_ENABLE_GRAPH_KERNEL)
    set(LITE_SRC_COMMON_MID_SRC
            ${LITE_SRC_COMMON_MID_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_library_loader.cc
            )
endif()

if(MSVC)
    set(LITE_SRC_COMMON_MID_SRC
            ${LITE_SRC_COMMON_MID_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/storage.cc)
endif()

set(LITE_SRC_TRAIN_COMMON_MID_SRC
        ${LITE_SRC_TRAIN_COMMON_MID_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/storage.cc
        )

add_library(lite_src_common_mid OBJECT ${LITE_SRC_COMMON_MID_SRC})
add_dependencies(lite_src_common_mid fbs_src fbs_inner_src)
add_library(lite_src_train_common_mid OBJECT ${LITE_SRC_TRAIN_COMMON_MID_SRC})
add_dependencies(lite_src_train_common_mid fbs_src fbs_inner_src)
