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
        biquad_ir.cc
        complex_norm_ir.cc
        compute_deltas_ir.cc
        contrast_ir.cc
        db_to_amplitude_ir.cc
        dc_shift_ir.cc
        deemph_biquad_ir.cc
        detect_pitch_frequency_ir.cc
        dither_ir.cc
        equalizer_biquad_ir.cc
        fade_ir.cc
        filtfilt_ir.cc
        flanger_ir.cc
        frequency_masking_ir.cc
        gain_ir.cc
        griffin_lim_ir.cc
        highpass_biquad_ir.cc
        inverse_mel_scale_ir.cc
        inverse_spectrogram_ir.cc
        lfcc_ir.cc
        lfilter_ir.cc
        lowpass_biquad_ir.cc
        magphase_ir.cc
        mask_along_axis_iid_ir.cc
        mask_along_axis_ir.cc
        mel_scale_ir.cc
        mel_spectrogram_ir.cc
        mfcc_ir.cc
        mu_law_decoding_ir.cc
        mu_law_encoding_ir.cc
        overdrive_ir.cc
        phase_vocoder_ir.cc
        phaser_ir.cc
        pitch_shift_ir.cc
        resample_ir.cc
        riaa_biquad_ir.cc
        sliding_window_cmn_ir.cc
        spectral_centroid_ir.cc
        spectrogram_ir.cc
        time_masking_ir.cc
        time_stretch_ir.cc
        treble_biquad_ir.cc
        vad_ir.cc
        vol_ir.cc
        )

