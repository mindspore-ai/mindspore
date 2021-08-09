file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

add_library(audio-ir-kernels OBJECT
        allpass_biquad_ir.cc
        amplitude_to_db_ir.cc
        angle_ir.cc
        band_biquad_ir.cc
        bandpass_biquad_ir.cc
        bandreject_biquad_ir.cc
        bass_biquad_ir.cc
        time_stretch_ir.cc
        )
