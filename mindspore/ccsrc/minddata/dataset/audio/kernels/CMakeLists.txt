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
        compute_deltas_op.cc
        contrast_op.cc
        db_to_amplitude_op.cc
        dc_shift_op.cc
        deemph_biquad_op.cc
        detect_pitch_frequency_op.cc
        dither_op.cc
        equalizer_biquad_op.cc
        fade_op.cc
        filtfilt_op.cc
        flanger_op.cc
        frequency_masking_op.cc
        gain_op.cc
        griffin_lim_op.cc
        highpass_biquad_op.cc
        inverse_mel_scale_op.cc
        inverse_spectrogram_op.cc
        lfcc_op.cc
        lfilter_op.cc
        lowpass_biquad_op.cc
        magphase_op.cc
        mask_along_axis_iid_op.cc
        mask_along_axis_op.cc
        mel_scale_op.cc
        mel_spectrogram_op.cc
        mfcc_op.cc
        mu_law_decoding_op.cc
        mu_law_encoding_op.cc
        overdrive_op.cc
        phase_vocoder_op.cc
        phaser_op.cc
        pitch_shift_op.cc
        resample_op.cc
        riaa_biquad_op.cc
        sliding_window_cmn_op.cc
        spectral_centroid_op.cc
        spectrogram_op.cc
        time_masking_op.cc
        time_stretch_op.cc
        treble_biquad_op.cc
        vad_op.cc
        vol_op.cc
        )

