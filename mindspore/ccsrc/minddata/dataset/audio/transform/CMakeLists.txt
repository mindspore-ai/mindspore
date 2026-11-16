file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        include(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/cmake/dataset.cmake)
        set(AUDIO_IR ${CMAKE_BINARY_DIR}/merge/mindspore/minddata/audio/transform)
        merge_minddata_files(${CMAKE_CURRENT_SOURCE_DIR}/ ${AUDIO_IR} dataset_audio_transform "" "" TRUE)
        file(GLOB AUDIO_IR_MERGE_LIST "${AUDIO_IR}/*.cc")
        add_library(audio-transform OBJECT ${AUDIO_IR_MERGE_LIST})
else()
        add_library(audio-transform OBJECT
                validators.cc
                allpass_biquad_ir.cc
                band_biquad_ir.cc
                bandpass_biquad_ir.cc
                bandreject_biquad_ir.cc
                bass_biquad_ir.cc
                biquad_ir.cc
                deemph_biquad_ir.cc
                equalizer_biquad_ir.cc
                filtfilt_ir.cc
                highpass_biquad_ir.cc
                lfilter_ir.cc
                lowpass_biquad_ir.cc
                riaa_biquad_ir.cc
                treble_biquad_ir.cc
                )
endif()
