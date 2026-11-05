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
                allpass_biquad_op.cc
                audio_utils.cc
                band_biquad_op.cc
                bandpass_biquad_op.cc
                bandreject_biquad_op.cc
                bass_biquad_op.cc
                biquad_op.cc
                deemph_biquad_op.cc
                equalizer_biquad_op.cc
                filtfilt_op.cc
                highpass_biquad_op.cc
                lfilter_op.cc
                lowpass_biquad_op.cc
                riaa_biquad_op.cc
                treble_biquad_op.cc
        )
endif()

