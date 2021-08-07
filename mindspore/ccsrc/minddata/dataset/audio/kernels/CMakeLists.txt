file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

add_library(audio-kernels OBJECT
        allpass_biquad_op.cc
        amplitude_to_db_op.cc
        angle_op.cc
        audio_utils.cc
        band_biquad_op.cc
        bandpass_biquad_op.cc
        bandreject_biquad_op.cc
        bass_biquad_op.cc
        time_stretch_op.cc
        )
