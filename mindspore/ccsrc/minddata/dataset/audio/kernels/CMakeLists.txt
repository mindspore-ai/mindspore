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
        biquad_op.cc
        complex_norm_op.cc
        contrast_op.cc
        dc_shift_op.cc
        deemph_biquad_op.cc
        equalizer_biquad_op.cc
        fade_op.cc
        frequency_masking_op.cc
        highpass_biquad_op.cc
        lfilter_op.cc
        lowpass_biquad_op.cc
        magphase_op.cc
        mu_law_decoding_op.cc
        time_masking_op.cc
        time_stretch_op.cc
        vol_op.cc
        )

