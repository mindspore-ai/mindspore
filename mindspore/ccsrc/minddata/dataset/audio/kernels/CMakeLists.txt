file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        include(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/cmake/dataset.cmake)
        set(AUDIO_KERNELS ${CMAKE_BINARY_DIR}/merge/mindspore/minddata/audio/kernels)
        merge_minddata_files(${CMAKE_CURRENT_SOURCE_DIR}/ ${AUDIO_KERNELS} dataset_audio_kernels "" "" TRUE)
        file(GLOB AUDIO_MERGE_LIST "${AUDIO_KERNELS}/*.cc")
        add_library(audio-kernels OBJECT ${AUDIO_MERGE_LIST})
else()
        add_library(audio-kernels OBJECT
                audio_utils.cc
        )
endif()

