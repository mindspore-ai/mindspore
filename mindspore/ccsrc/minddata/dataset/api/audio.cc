/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "minddata/dataset/include/dataset/audio.h"

#include "minddata/dataset/audio/ir/kernels/allpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/amplitude_to_db_ir.h"
#include "minddata/dataset/audio/ir/kernels/angle_ir.h"
#include "minddata/dataset/audio/ir/kernels/band_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bandpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bandreject_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/complex_norm_ir.h"
#include "minddata/dataset/audio/ir/kernels/compute_deltas_ir.h"
#include "minddata/dataset/audio/ir/kernels/contrast_ir.h"
#include "minddata/dataset/audio/ir/kernels/db_to_amplitude_ir.h"
#include "minddata/dataset/audio/ir/kernels/dc_shift_ir.h"
#include "minddata/dataset/audio/ir/kernels/deemph_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/detect_pitch_frequency_ir.h"
#include "minddata/dataset/audio/ir/kernels/dither_ir.h"
#include "minddata/dataset/audio/ir/kernels/equalizer_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/fade_ir.h"
#include "minddata/dataset/audio/ir/kernels/filtfilt_ir.h"
#include "minddata/dataset/audio/ir/kernels/flanger_ir.h"
#include "minddata/dataset/audio/ir/kernels/frequency_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/gain_ir.h"
#include "minddata/dataset/audio/ir/kernels/griffin_lim_ir.h"
#include "minddata/dataset/audio/ir/kernels/highpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/inverse_mel_scale_ir.h"
#include "minddata/dataset/audio/ir/kernels/inverse_spectrogram_ir.h"
#include "minddata/dataset/audio/ir/kernels/lfcc_ir.h"
#include "minddata/dataset/audio/ir/kernels/lfilter_ir.h"
#include "minddata/dataset/audio/ir/kernels/lowpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/magphase_ir.h"
#include "minddata/dataset/audio/ir/kernels/mask_along_axis_iid_ir.h"
#include "minddata/dataset/audio/ir/kernels/mask_along_axis_ir.h"
#include "minddata/dataset/audio/ir/kernels/mel_scale_ir.h"
#include "minddata/dataset/audio/ir/kernels/mel_spectrogram_ir.h"
#include "minddata/dataset/audio/ir/kernels/mfcc_ir.h"
#include "minddata/dataset/audio/ir/kernels/mu_law_decoding_ir.h"
#include "minddata/dataset/audio/ir/kernels/mu_law_encoding_ir.h"
#include "minddata/dataset/audio/ir/kernels/overdrive_ir.h"
#include "minddata/dataset/audio/ir/kernels/phase_vocoder_ir.h"
#include "minddata/dataset/audio/ir/kernels/phaser_ir.h"
#include "minddata/dataset/audio/ir/kernels/pitch_shift_ir.h"
#include "minddata/dataset/audio/ir/kernels/resample_ir.h"
#include "minddata/dataset/audio/ir/kernels/riaa_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/sliding_window_cmn_ir.h"
#include "minddata/dataset/audio/ir/kernels/spectral_centroid_ir.h"
#include "minddata/dataset/audio/ir/kernels/spectrogram_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_stretch_ir.h"
#include "minddata/dataset/audio/ir/kernels/treble_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/vad_ir.h"
#include "minddata/dataset/audio/ir/kernels/vol_ir.h"
#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"

namespace mindspore {
namespace dataset {
namespace audio {
// AllpassBiquad Transform Operation.
struct AllpassBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
};

AllpassBiquad::AllpassBiquad(int32_t sample_rate, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q)) {}

std::shared_ptr<TensorOperation> AllpassBiquad::Parse() {
  return std::make_shared<AllpassBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_);
}

// AmplitudeToDB Transform Operation.
struct AmplitudeToDB::Data {
  Data(ScaleType stype, float ref_value, float amin, float top_db)
      : stype_(stype), ref_value_(ref_value), amin_(amin), top_db_(top_db) {}
  ScaleType stype_;
  float ref_value_;
  float amin_;
  float top_db_;
};

AmplitudeToDB::AmplitudeToDB(ScaleType stype, float ref_value, float amin, float top_db)
    : data_(std::make_shared<Data>(stype, ref_value, amin, top_db)) {}

std::shared_ptr<TensorOperation> AmplitudeToDB::Parse() {
  return std::make_shared<AmplitudeToDBOperation>(data_->stype_, data_->ref_value_, data_->amin_, data_->top_db_);
}

// Angle Transform Operation.
Angle::Angle() = default;

std::shared_ptr<TensorOperation> Angle::Parse() { return std::make_shared<AngleOperation>(); }

// BandBiquad Transform Operation.
struct BandBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q, bool noise)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), noise_(noise) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
  bool noise_;
};

BandBiquad::BandBiquad(int32_t sample_rate, float central_freq, float Q, bool noise)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q, noise)) {}

std::shared_ptr<TensorOperation> BandBiquad::Parse() {
  return std::make_shared<BandBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_, data_->noise_);
}

// BandpassBiquad Transform Operation.
struct BandpassBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q, bool const_skirt_gain)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), const_skirt_gain_(const_skirt_gain) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
  bool const_skirt_gain_;
};

BandpassBiquad::BandpassBiquad(int32_t sample_rate, float central_freq, float Q, bool const_skirt_gain)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q, const_skirt_gain)) {}

std::shared_ptr<TensorOperation> BandpassBiquad::Parse() {
  return std::make_shared<BandpassBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_,
                                                   data_->const_skirt_gain_);
}

// BandrejectBiquad Transform Operation.
struct BandrejectBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
};

BandrejectBiquad::BandrejectBiquad(int32_t sample_rate, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q)) {}

std::shared_ptr<TensorOperation> BandrejectBiquad::Parse() {
  return std::make_shared<BandrejectBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_);
}

// BassBiquad Transform Operation.
struct BassBiquad::Data {
  Data(int32_t sample_rate, float gain, float central_freq, float Q)
      : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float gain_;
  float central_freq_;
  float Q_;
};

BassBiquad::BassBiquad(int32_t sample_rate, float gain, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, gain, central_freq, Q)) {}

std::shared_ptr<TensorOperation> BassBiquad::Parse() {
  return std::make_shared<BassBiquadOperation>(data_->sample_rate_, data_->gain_, data_->central_freq_, data_->Q_);
}

// Biquad Transform Operation.
struct Biquad::Data {
  Data(float b0, float b1, float b2, float a0, float a1, float a2)
      : b0_(b0), b1_(b1), b2_(b2), a0_(a0), a1_(a1), a2_(a2) {}
  float b0_;
  float b1_;
  float b2_;
  float a0_;
  float a1_;
  float a2_;
};

Biquad::Biquad(float b0, float b1, float b2, float a0, float a1, float a2)
    : data_(std::make_shared<Data>(b0, b1, b2, a0, a1, a2)) {}

std::shared_ptr<TensorOperation> Biquad::Parse() {
  return std::make_shared<BiquadOperation>(data_->b0_, data_->b1_, data_->b2_, data_->a0_, data_->a1_, data_->a1_);
}

// ComplexNorm Transform Operation.
struct ComplexNorm::Data {
  explicit Data(float power) : power_(power) {}
  float power_;
};

ComplexNorm::ComplexNorm(float power) : data_(std::make_shared<Data>(power)) {}

std::shared_ptr<TensorOperation> ComplexNorm::Parse() { return std::make_shared<ComplexNormOperation>(data_->power_); }

// ComputeDeltas Transform Operation.
struct ComputeDeltas::Data {
  Data(int32_t win_length, BorderType pad_mode) : win_length_(win_length), pad_mode_(pad_mode) {}
  int32_t win_length_;
  BorderType pad_mode_;
};

ComputeDeltas::ComputeDeltas(int32_t win_length, BorderType pad_mode)
    : data_(std::make_shared<Data>(win_length, pad_mode)) {}

std::shared_ptr<TensorOperation> ComputeDeltas::Parse() {
  return std::make_shared<ComputeDeltasOperation>(data_->win_length_, data_->pad_mode_);
}

// Contrast Transform Operation.
struct Contrast::Data {
  explicit Data(float enhancement_amount) : enhancement_amount_(enhancement_amount) {}
  float enhancement_amount_;
};

Contrast::Contrast(float enhancement_amount) : data_(std::make_shared<Data>(enhancement_amount)) {}

std::shared_ptr<TensorOperation> Contrast::Parse() {
  return std::make_shared<ContrastOperation>(data_->enhancement_amount_);
}

// DBToAmplitude Transform Operation.
struct DBToAmplitude::Data {
  explicit Data(float ref, float power) : ref_(ref), power_(power) {}
  float ref_;
  float power_;
};

DBToAmplitude::DBToAmplitude(float ref, float power) : data_(std::make_shared<Data>(ref, power)) {}

std::shared_ptr<TensorOperation> DBToAmplitude::Parse() {
  return std::make_shared<DBToAmplitudeOperation>(data_->ref_, data_->power_);
}

// DCShift Transform Operation.
struct DCShift::Data {
  Data(float shift, float limiter_gain) : shift_(shift), limiter_gain_(limiter_gain) {}
  float shift_;
  float limiter_gain_;
};

DCShift::DCShift(float shift) : data_(std::make_shared<Data>(shift, shift)) {}

DCShift::DCShift(float shift, float limiter_gain) : data_(std::make_shared<Data>(shift, limiter_gain)) {}

std::shared_ptr<TensorOperation> DCShift::Parse() {
  return std::make_shared<DCShiftOperation>(data_->shift_, data_->limiter_gain_);
}

Status CreateDct(mindspore::MSTensor *output, int32_t n_mfcc, int32_t n_mels, NormMode norm) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CreateDct", "n_mfcc", n_mfcc));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CreateDct", "n_mels", n_mels));

  std::shared_ptr<dataset::Tensor> dct;
  RETURN_IF_NOT_OK(Dct(&dct, n_mfcc, n_mels, norm));
  CHECK_FAIL_RETURN_UNEXPECTED(dct->HasData(), "CreateDct: get an empty tensor with shape " + dct->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(dct));
  return Status::OK();
}

// DeemphBiquad Transform Operation.
struct DeemphBiquad::Data {
  explicit Data(int32_t sample_rate) : sample_rate_(sample_rate) {}
  int32_t sample_rate_;
};

DeemphBiquad::DeemphBiquad(int32_t sample_rate) : data_(std::make_shared<Data>(sample_rate)) {}

std::shared_ptr<TensorOperation> DeemphBiquad::Parse() {
  return std::make_shared<DeemphBiquadOperation>(data_->sample_rate_);
}

// DetectPitchFrequency Transform Operation.
struct DetectPitchFrequency::Data {
  Data(int32_t sample_rate, float frame_time, int32_t win_length, int32_t freq_low, int32_t freq_high)
      : sample_rate_(sample_rate),
        frame_time_(frame_time),
        win_length_(win_length),
        freq_low_(freq_low),
        freq_high_(freq_high) {}
  int32_t sample_rate_;
  float frame_time_;
  int32_t win_length_;
  int32_t freq_low_;
  int32_t freq_high_;
};

DetectPitchFrequency::DetectPitchFrequency(int32_t sample_rate, float frame_time, int32_t win_length, int32_t freq_low,
                                           int32_t freq_high)
    : data_(std::make_shared<Data>(sample_rate, frame_time, win_length, freq_low, freq_high)) {}

std::shared_ptr<TensorOperation> DetectPitchFrequency::Parse() {
  return std::make_shared<DetectPitchFrequencyOperation>(data_->sample_rate_, data_->frame_time_, data_->win_length_,
                                                         data_->freq_low_, data_->freq_high_);
}

// Dither Transform Operation.
struct Dither::Data {
  Data(DensityFunction density_function, bool noise_shaping)
      : density_function_(density_function), noise_shaping_(noise_shaping) {}
  DensityFunction density_function_;
  bool noise_shaping_;
};

Dither::Dither(DensityFunction density_function, bool noise_shaping)
    : data_(std::make_shared<Data>(density_function, noise_shaping)) {}

std::shared_ptr<TensorOperation> Dither::Parse() {
  return std::make_shared<DitherOperation>(data_->density_function_, data_->noise_shaping_);
}

// EqualizerBiquad Transform Operation.
struct EqualizerBiquad::Data {
  Data(int32_t sample_rate, float center_freq, float gain, float Q)
      : sample_rate_(sample_rate), center_freq_(center_freq), gain_(gain), Q_(Q) {}
  int32_t sample_rate_;
  float center_freq_;
  float gain_;
  float Q_;
};

EqualizerBiquad::EqualizerBiquad(int32_t sample_rate, float center_freq, float gain, float Q)
    : data_(std::make_shared<Data>(sample_rate, center_freq, gain, Q)) {}

std::shared_ptr<TensorOperation> EqualizerBiquad::Parse() {
  return std::make_shared<EqualizerBiquadOperation>(data_->sample_rate_, data_->center_freq_, data_->gain_, data_->Q_);
}

// Fade Transform Operation.
struct Fade::Data {
  Data(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape)
      : fade_in_len_(fade_in_len), fade_out_len_(fade_out_len), fade_shape_(fade_shape) {}
  int32_t fade_in_len_;
  int32_t fade_out_len_;
  FadeShape fade_shape_;
};

Fade::Fade(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape)
    : data_(std::make_shared<Data>(fade_in_len, fade_out_len, fade_shape)) {}

std::shared_ptr<TensorOperation> Fade::Parse() {
  return std::make_shared<FadeOperation>(data_->fade_in_len_, data_->fade_out_len_, data_->fade_shape_);
}

// Filtfilt Transform Operation.
struct Filtfilt::Data {
  Data(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
      : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}
  std::vector<float> a_coeffs_;
  std::vector<float> b_coeffs_;
  bool clamp_;
};

Filtfilt::Filtfilt(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : data_(std::make_shared<Data>(a_coeffs, b_coeffs, clamp)) {}

std::shared_ptr<TensorOperation> Filtfilt::Parse() {
  return std::make_shared<FiltfiltOperation>(data_->a_coeffs_, data_->b_coeffs_, data_->clamp_);
}

// Flanger Transform Operation.
struct Flanger::Data {
  Data(int32_t sample_rate, float delay, float depth, float regen, float width, float speed, float phase,
       Modulation modulation, Interpolation interpolation)
      : sample_rate_(sample_rate),
        delay_(delay),
        depth_(depth),
        regen_(regen),
        width_(width),
        speed_(speed),
        phase_(phase),
        modulation_(modulation),
        interpolation_(interpolation) {}
  int32_t sample_rate_;
  float delay_;
  float depth_;
  float regen_;
  float width_;
  float speed_;
  float phase_;
  Modulation modulation_;
  Interpolation interpolation_;
};

Flanger::Flanger(int32_t sample_rate, float delay, float depth, float regen, float width, float speed, float phase,
                 Modulation modulation, Interpolation interpolation)
    : data_(std::make_shared<Data>(sample_rate, delay, depth, regen, width, speed, phase, modulation, interpolation)) {}

std::shared_ptr<TensorOperation> Flanger::Parse() {
  return std::make_shared<FlangerOperation>(data_->sample_rate_, data_->delay_, data_->depth_, data_->regen_,
                                            data_->width_, data_->speed_, data_->phase_, data_->modulation_,
                                            data_->interpolation_);
}

// FrequencyMasking Transform Operation.
struct FrequencyMasking::Data {
  Data(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, float mask_value)
      : iid_masks_(iid_masks),
        frequency_mask_param_(frequency_mask_param),
        mask_start_(mask_start),
        mask_value_(mask_value) {}
  bool iid_masks_;
  int32_t frequency_mask_param_;
  int32_t mask_start_;
  float mask_value_;
};

FrequencyMasking::FrequencyMasking(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, float mask_value)
    : data_(std::make_shared<Data>(iid_masks, frequency_mask_param, mask_start, mask_value)) {}

std::shared_ptr<TensorOperation> FrequencyMasking::Parse() {
  return std::make_shared<FrequencyMaskingOperation>(data_->iid_masks_, data_->frequency_mask_param_,
                                                     data_->mask_start_, data_->mask_value_);
}

// Gain Transform Operation.
struct Gain::Data {
  explicit Data(float gain_db) : gain_db_(gain_db) {}
  float gain_db_;
};

Gain::Gain(float gain_db) : data_(std::make_shared<Data>(gain_db)) {}

std::shared_ptr<TensorOperation> Gain::Parse() { return std::make_shared<GainOperation>(data_->gain_db_); }

// GriffinLim Transform Operation.
struct GriffinLim::Data {
  Data(int32_t n_fft, int32_t n_iter, int32_t win_length, int32_t hop_length, WindowType window_type, float power,
       float momentum, int32_t length, bool rand_init)
      : n_fft_(n_fft),
        n_iter_(n_iter),
        win_length_(win_length),
        hop_length_(hop_length),
        window_type_(window_type),
        power_(power),
        momentum_(momentum),
        length_(length),
        rand_init_(rand_init) {}
  int32_t n_fft_;
  int32_t n_iter_;
  int32_t win_length_;
  int32_t hop_length_;
  WindowType window_type_;
  float power_;
  float momentum_;
  int32_t length_;
  bool rand_init_;
};

GriffinLim::GriffinLim(int32_t n_fft, int32_t n_iter, int32_t win_length, int32_t hop_length, WindowType window_type,
                       float power, float momentum, int32_t length, bool rand_init)
    : data_(std::make_shared<Data>(n_fft, n_iter, win_length, hop_length, window_type, power, momentum, length,
                                   rand_init)) {}

std::shared_ptr<TensorOperation> GriffinLim::Parse() {
  return std::make_shared<GriffinLimOperation>(data_->n_fft_, data_->n_iter_, data_->win_length_, data_->hop_length_,
                                               data_->window_type_, data_->power_, data_->momentum_, data_->length_,
                                               data_->rand_init_);
}

// HighpassBiquad Transform Operation.
struct HighpassBiquad::Data {
  Data(int32_t sample_rate, float cutoff_freq, float Q) : sample_rate_(sample_rate), cutoff_freq_(cutoff_freq), Q_(Q) {}
  int32_t sample_rate_;
  float cutoff_freq_;
  float Q_;
};

HighpassBiquad::HighpassBiquad(int32_t sample_rate, float cutoff_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, cutoff_freq, Q)) {}

std::shared_ptr<TensorOperation> HighpassBiquad::Parse() {
  return std::make_shared<HighpassBiquadOperation>(data_->sample_rate_, data_->cutoff_freq_, data_->Q_);
}

// InverseMelScale Transform Operation.
struct InverseMelScale::Data {
  Data(int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t max_iter,
       float tolerance_loss, float tolerance_change, const std::map<std::string, float> &sgdargs, NormType norm,
       MelType mel_type)
      : n_stft_(n_stft),
        n_mels_(n_mels),
        sample_rate_(sample_rate),
        f_min_(f_min),
        f_max_(f_max),
        max_iter_(max_iter),
        tolerance_loss_(tolerance_loss),
        tolerance_change_(tolerance_change),
        sgdargs_(sgdargs),
        norm_(norm),
        mel_type_(mel_type) {}
  int32_t n_stft_;
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t max_iter_;
  float tolerance_loss_;
  float tolerance_change_;
  std::map<std::string, float> sgdargs_;
  NormType norm_;
  MelType mel_type_;
};

InverseMelScale::InverseMelScale(int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min, float f_max,
                                 int32_t max_iter, float tolerance_loss, float tolerance_change,
                                 const std::map<std::string, float> &sgdargs, NormType norm, MelType mel_type)
    : data_(std::make_shared<Data>(n_stft, n_mels, sample_rate, f_min, f_max, max_iter, tolerance_loss,
                                   tolerance_change, sgdargs, norm, mel_type)) {}

std::shared_ptr<TensorOperation> InverseMelScale::Parse() {
  return std::make_shared<InverseMelScaleOperation>(
    data_->n_stft_, data_->n_mels_, data_->sample_rate_, data_->f_min_, data_->f_max_, data_->max_iter_,
    data_->tolerance_loss_, data_->tolerance_change_, data_->sgdargs_, data_->norm_, data_->mel_type_);
}

// InverseSpectrogram Transform Operation.
struct InverseSpectrogram::Data {
  Data(int32_t length, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window,
       bool normalized, bool center, BorderType pad_mode, bool onesided)
      : length_(length),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        pad_(pad),
        window_(window),
        normalized_(normalized),
        center_(center),
        pad_mode_(pad_mode),
        onesided_(onesided) {}
  int32_t length_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
};

InverseSpectrogram::InverseSpectrogram(int32_t length, int32_t n_fft, int32_t win_length, int32_t hop_length,
                                       int32_t pad, WindowType window, bool normalized, bool center,
                                       BorderType pad_mode, bool onesided)
    : data_(std::make_shared<Data>(length, n_fft, win_length, hop_length, pad, window, normalized, center, pad_mode,
                                   onesided)) {}

std::shared_ptr<TensorOperation> InverseSpectrogram::Parse() {
  return std::make_shared<InverseSpectrogramOperation>(
    data_->length_, data_->n_fft_, data_->win_length_, data_->hop_length_, data_->pad_, data_->window_,
    data_->normalized_, data_->center_, data_->pad_mode_, data_->onesided_);
}

// LFCC Transform Operation.
struct LFCC::Data {
  Data(int32_t sample_rate, int32_t n_filter, int32_t n_lfcc, float f_min, float f_max, int32_t dct_type, NormMode norm,
       bool log_lf, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window, float power,
       bool normalized, bool center, BorderType pad_mode, bool onesided)
      : sample_rate_(sample_rate),
        n_filter_(n_filter),
        n_lfcc_(n_lfcc),
        f_min_(f_min),
        f_max_(f_max),
        dct_type_(dct_type),
        norm_(norm),
        log_lf_(log_lf),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        pad_(pad),
        window_(window),
        power_(power),
        normalized_(normalized),
        center_(center),
        pad_mode_(pad_mode),
        onesided_(onesided) {}
  int32_t sample_rate_;
  int32_t n_filter_;
  int32_t n_lfcc_;
  float f_min_;
  float f_max_;
  int32_t dct_type_;
  NormMode norm_;
  bool log_lf_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
  float power_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
};

LFCC::LFCC(int32_t sample_rate, int32_t n_filter, int32_t n_lfcc, float f_min, float f_max, int32_t dct_type,
           NormMode norm, bool log_lf, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad,
           WindowType window, float power, bool normalized, bool center, BorderType pad_mode, bool onesided)
    : data_(std::make_shared<Data>(sample_rate, n_filter, n_lfcc, f_min, f_max, dct_type, norm, log_lf, n_fft,
                                   win_length, hop_length, pad, window, power, normalized, center, pad_mode,
                                   onesided)) {}

std::shared_ptr<TensorOperation> LFCC::Parse() {
  return std::make_shared<LFCCOperation>(
    data_->sample_rate_, data_->n_filter_, data_->n_lfcc_, data_->f_min_, data_->f_max_, data_->dct_type_, data_->norm_,
    data_->log_lf_, data_->n_fft_, data_->win_length_, data_->hop_length_, data_->pad_, data_->window_, data_->power_,
    data_->normalized_, data_->center_, data_->pad_mode_, data_->onesided_);
}

// LFilter Transform Operation.
struct LFilter::Data {
  Data(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
      : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}
  std::vector<float> a_coeffs_;
  std::vector<float> b_coeffs_;
  bool clamp_;
};

LFilter::LFilter(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : data_(std::make_shared<Data>(a_coeffs, b_coeffs, clamp)) {}

std::shared_ptr<TensorOperation> LFilter::Parse() {
  return std::make_shared<LFilterOperation>(data_->a_coeffs_, data_->b_coeffs_, data_->clamp_);
}

// LowpassBiquad Transform Operation.
struct LowpassBiquad::Data {
  Data(int32_t sample_rate, float cutoff_freq, float Q) : sample_rate_(sample_rate), cutoff_freq_(cutoff_freq), Q_(Q) {}
  int32_t sample_rate_;
  float cutoff_freq_;
  float Q_;
};

LowpassBiquad::LowpassBiquad(int32_t sample_rate, float cutoff_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, cutoff_freq, Q)) {}

std::shared_ptr<TensorOperation> LowpassBiquad::Parse() {
  return std::make_shared<LowpassBiquadOperation>(data_->sample_rate_, data_->cutoff_freq_, data_->Q_);
}

// Magphase Transform Operation.
struct Magphase::Data {
  explicit Data(float power) : power_(power) {}
  float power_;
};

Magphase::Magphase(float power) : data_(std::make_shared<Data>(power)) {}

std::shared_ptr<TensorOperation> Magphase::Parse() { return std::make_shared<MagphaseOperation>(data_->power_); }

// MaskAlongAxis Transform Operation.
struct MaskAlongAxis::Data {
  Data(int32_t mask_start, int32_t mask_width, float mask_value, int32_t axis)
      : mask_start_(mask_start), mask_width_(mask_width), mask_value_(mask_value), axis_(axis) {}
  int32_t mask_start_;
  int32_t mask_width_;
  float mask_value_;
  int32_t axis_;
};

MaskAlongAxis::MaskAlongAxis(int32_t mask_start, int32_t mask_width, float mask_value, int32_t axis)
    : data_(std::make_shared<Data>(mask_start, mask_width, mask_value, axis)) {}

std::shared_ptr<TensorOperation> MaskAlongAxis::Parse() {
  return std::make_shared<MaskAlongAxisOperation>(data_->mask_start_, data_->mask_width_, data_->mask_value_,
                                                  data_->axis_);
}

// MaskAlongAxisIID Transform Operation.
struct MaskAlongAxisIID::Data {
  Data(int32_t mask_param, float mask_value, int32_t axis)
      : mask_param_(mask_param), mask_value_(mask_value), axis_(axis) {}
  int32_t mask_param_;
  float mask_value_;
  int32_t axis_;
};

MaskAlongAxisIID::MaskAlongAxisIID(int32_t mask_param, float mask_value, int32_t axis)
    : data_(std::make_shared<Data>(mask_param, mask_value, axis)) {}

std::shared_ptr<TensorOperation> MaskAlongAxisIID::Parse() {
  return std::make_shared<MaskAlongAxisIIDOperation>(data_->mask_param_, data_->mask_value_, data_->axis_);
}

// MelScale Transform Operation.
struct MelScale::Data {
  Data(int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t n_stft, NormType norm, MelType mel_type)
      : n_mels_(n_mels),
        sample_rate_(sample_rate),
        f_min_(f_min),
        f_max_(f_max),
        n_stft_(n_stft),
        norm_(norm),
        mel_type_(mel_type) {}
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t n_stft_;
  NormType norm_;
  MelType mel_type_;
};

MelScale::MelScale(int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t n_stft, NormType norm,
                   MelType mel_type)
    : data_(std::make_shared<Data>(n_mels, sample_rate, f_min, f_max, n_stft, norm, mel_type)) {}

std::shared_ptr<TensorOperation> MelScale::Parse() {
  return std::make_shared<MelScaleOperation>(data_->n_mels_, data_->sample_rate_, data_->f_min_, data_->f_max_,
                                             data_->n_stft_, data_->norm_, data_->mel_type_);
}

// MelscaleFbanks Function.
Status MelscaleFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels, int32_t sample_rate,
                      NormType norm, MelType mel_type) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(n_freqs > 0,
                               "MelscaleFbanks: n_freqs must be greater than 0, got: " + std::to_string(n_freqs));

  CHECK_FAIL_RETURN_UNEXPECTED(f_min >= 0, "MelscaleFbanks: f_min must be non negative, got: " + std::to_string(f_min));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > 0,
                               "MelscaleFbanks: f_max must be greater than 0, got: " + std::to_string(f_max));
  CHECK_FAIL_RETURN_UNEXPECTED(n_mels > 0,
                               "MelscaleFbanks: n_mels must be greater than 0, got: " + std::to_string(n_mels));
  CHECK_FAIL_RETURN_UNEXPECTED(
    sample_rate > 0, "MelscaleFbanks: sample_rate must be greater than 0, got: " + std::to_string(sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > f_min, "MelscaleFbanks: f_max must be greater than f_min, got: f_min = " +
                                                std::to_string(f_min) + ", while f_max = " + std::to_string(f_max));
  std::shared_ptr<dataset::Tensor> fb;
  RETURN_IF_NOT_OK(CreateFbanks<float>(&fb, n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_type));
  CHECK_FAIL_RETURN_UNEXPECTED(fb->HasData(),
                               "MelscaleFbanks: get an empty tensor with shape " + fb->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(fb));
  return Status::OK();
}

// MelSpectrogram Transform Operation.
struct MelSpectrogram::Data {
  Data(int32_t sample_rate, int32_t n_fft, int32_t win_length, int32_t hop_length, float f_min, float f_max,
       int32_t pad, int32_t n_mels, WindowType window, float power, bool normalized, bool center, BorderType pad_mode,
       bool onesided, NormType norm, MelType mel_scale)
      : sample_rate_(sample_rate),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        f_min_(f_min),
        f_max_(f_max),
        pad_(pad),
        n_mels_(n_mels),
        window_(window),
        power_(power),
        normalized_(normalized),
        center_(center),
        pad_mode_(pad_mode),
        onesided_(onesided),
        norm_(norm),
        mel_scale_(mel_scale) {}
  int32_t sample_rate_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  float f_min_;
  float f_max_;
  int32_t pad_;
  int32_t n_mels_;
  WindowType window_;
  float power_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
  NormType norm_;
  MelType mel_scale_;
};

MelSpectrogram::MelSpectrogram(int32_t sample_rate, int32_t n_fft, int32_t win_length, int32_t hop_length, float f_min,
                               float f_max, int32_t pad, int32_t n_mels, WindowType window, float power,
                               bool normalized, bool center, BorderType pad_mode, bool onesided, NormType norm,
                               MelType mel_scale)
    : data_(std::make_shared<Data>(sample_rate, n_fft, win_length, hop_length, f_min, f_max, pad, n_mels, window, power,
                                   normalized, center, pad_mode, onesided, norm, mel_scale)) {}

std::shared_ptr<TensorOperation> MelSpectrogram::Parse() {
  return std::make_shared<MelSpectrogramOperation>(
    data_->sample_rate_, data_->n_fft_, data_->win_length_, data_->hop_length_, data_->f_min_, data_->f_max_,
    data_->pad_, data_->n_mels_, data_->window_, data_->power_, data_->normalized_, data_->center_, data_->pad_mode_,
    data_->onesided_, data_->norm_, data_->mel_scale_);
}

// MFCC Transform Operation.
struct MFCC::Data {
  Data(int32_t sample_rate, int32_t n_mfcc, int32_t dct_type, NormMode norm, bool log_mels, int32_t n_fft,
       int32_t win_length, int32_t hop_length, float f_min, float f_max, int32_t pad, int32_t n_mels, WindowType window,
       float power, bool normalized, bool center, BorderType pad_mode, bool onesided, NormType norm_mel,
       MelType mel_scale)
      : sample_rate_(sample_rate),
        n_mfcc_(n_mfcc),
        dct_type_(dct_type),
        norm_(norm),
        log_mels_(log_mels),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        f_min_(f_min),
        f_max_(f_max),
        pad_(pad),
        n_mels_(n_mels),
        window_(window),
        power_(power),
        normalized_(normalized),
        center_(center),
        pad_mode_(pad_mode),
        onesided_(onesided),
        norm_mel_(norm_mel),
        mel_scale_(mel_scale) {}
  int32_t sample_rate_;
  int32_t n_mfcc_;
  int32_t dct_type_;
  NormMode norm_;
  bool log_mels_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  float f_min_;
  float f_max_;
  int32_t pad_;
  int32_t n_mels_;
  WindowType window_;
  float power_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
  NormType norm_mel_;
  MelType mel_scale_;
  std::map<std::string, std::string> melkwargs_;
};

MFCC::MFCC(int32_t sample_rate, int32_t n_mfcc, int32_t dct_type, NormMode norm, bool log_mels, int32_t n_fft,
           int32_t win_length, int32_t hop_length, float f_min, float f_max, int32_t pad, int32_t n_mels,
           WindowType window, float power, bool normalized, bool center, BorderType pad_mode, bool onesided,
           NormType norm_mel, MelType mel_scale)
    : data_(std::make_shared<Data>(sample_rate, n_mfcc, dct_type, norm, log_mels, n_fft, win_length, hop_length, f_min,
                                   f_max, pad, n_mels, window, power, normalized, center, pad_mode, onesided, norm_mel,
                                   mel_scale)) {}

std::shared_ptr<TensorOperation> MFCC::Parse() {
  return std::make_shared<MFCCOperation>(data_->sample_rate_, data_->n_mfcc_, data_->dct_type_, data_->norm_,
                                         data_->log_mels_, data_->n_fft_, data_->win_length_, data_->hop_length_,
                                         data_->f_min_, data_->f_max_, data_->pad_, data_->n_mels_, data_->window_,
                                         data_->power_, data_->normalized_, data_->center_, data_->pad_mode_,
                                         data_->onesided_, data_->norm_mel_, data_->mel_scale_);
}

// MuLawDecoding Transform Operation.
struct MuLawDecoding::Data {
  explicit Data(int32_t quantization_channels) : quantization_channels_(quantization_channels) {}
  int32_t quantization_channels_;
};

MuLawDecoding::MuLawDecoding(int32_t quantization_channels) : data_(std::make_shared<Data>(quantization_channels)) {}

std::shared_ptr<TensorOperation> MuLawDecoding::Parse() {
  return std::make_shared<MuLawDecodingOperation>(data_->quantization_channels_);
}

// MuLawEncoding Transform Operation.
struct MuLawEncoding::Data {
  explicit Data(int32_t quantization_channels) : quantization_channels_(quantization_channels) {}
  int32_t quantization_channels_;
};

MuLawEncoding::MuLawEncoding(int32_t quantization_channels) : data_(std::make_shared<Data>(quantization_channels)) {}

std::shared_ptr<TensorOperation> MuLawEncoding::Parse() {
  return std::make_shared<MuLawEncodingOperation>(data_->quantization_channels_);
}

// LinearFbanks Function.
Status LinearFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_filter,
                    int32_t sample_rate) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(n_freqs > 0,
                               "LinearFbanks: n_freqs must be greater than 0, got: " + std::to_string(n_freqs));

  CHECK_FAIL_RETURN_UNEXPECTED(f_min >= 0, "LinearFbanks: f_min must be non negative, got: " + std::to_string(f_min));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > 0, "LinearFbanks: f_max must be greater than 0, got: " + std::to_string(f_max));
  CHECK_FAIL_RETURN_UNEXPECTED(n_filter > 0,
                               "LinearFbanks: n_filter must be greater than 0, got: " + std::to_string(n_filter));
  CHECK_FAIL_RETURN_UNEXPECTED(sample_rate > 0,
                               "LinearFbanks: sample_rate must be greater than 0, got: " + std::to_string(sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > f_min, "LinearFbanks: f_max must be greater than f_min, got: f_min = " +
                                                std::to_string(f_min) + ", while f_max = " + std::to_string(f_max));
  std::shared_ptr<dataset::Tensor> fb;
  RETURN_IF_NOT_OK(CreateLinearFbanks(&fb, n_freqs, f_min, f_max, n_filter, sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(fb->HasData(), "LinearFbanks: get an empty tensor with shape " + fb->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(fb));
  return Status::OK();
}

// Overdrive Transform Operation.
struct Overdrive::Data {
  Data(float gain, float color) : gain_(gain), color_(color) {}
  float gain_;
  float color_;
};

Overdrive::Overdrive(float gain, float color) : data_(std::make_shared<Data>(gain, color)) {}

std::shared_ptr<TensorOperation> Overdrive::Parse() {
  return std::make_shared<OverdriveOperation>(data_->gain_, data_->color_);
}

// Phaser Transform Operation.
struct Phaser::Data {
  Data(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay, float mod_speed,
       bool sinusoidal)
      : sample_rate_(sample_rate),
        gain_in_(gain_in),
        gain_out_(gain_out),
        delay_ms_(delay_ms),
        decay_(decay),
        mod_speed_(mod_speed),
        sinusoidal_(sinusoidal) {}
  int32_t sample_rate_;
  float gain_in_;
  float gain_out_;
  float delay_ms_;
  float decay_;
  float mod_speed_;
  bool sinusoidal_;
};

Phaser::Phaser(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay, float mod_speed,
               bool sinusoidal)
    : data_(std::make_shared<Data>(sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal)) {}

std::shared_ptr<TensorOperation> Phaser::Parse() {
  return std::make_shared<PhaserOperation>(data_->sample_rate_, data_->gain_in_, data_->gain_out_, data_->delay_ms_,
                                           data_->decay_, data_->mod_speed_, data_->sinusoidal_);
}

// PhaseVocoder Transofrm Operation.
struct PhaseVocoder::Data {
  Data(float rate, const MSTensor &phase_advance) : rate_(rate), phase_advance_(phase_advance) {}
  float rate_;
  MSTensor phase_advance_;
};

PhaseVocoder::PhaseVocoder(float rate, const MSTensor &phase_advance)
    : data_(std::make_shared<Data>(rate, phase_advance)) {}

std::shared_ptr<TensorOperation> PhaseVocoder::Parse() {
  std::shared_ptr<Tensor> phase_advance;
  Status rc = Tensor::CreateFromMSTensor(data_->phase_advance_, &phase_advance);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error creating phase_vocoder constant tensor." << rc;
    return nullptr;
  }
  return std::make_shared<PhaseVocoderOperation>(data_->rate_, phase_advance);
}

// pitchshift
struct PitchShift::Data {
  Data(int32_t sample_rate, int32_t n_steps, int32_t bins_per_octave, int32_t n_fft, int32_t win_length,
       int32_t hop_length, WindowType window)
      : sample_rate_(sample_rate),
        n_steps_(n_steps),
        bins_per_octave_(bins_per_octave),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        window_(window) {}
  int32_t sample_rate_;
  int32_t n_steps_;
  int32_t bins_per_octave_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  WindowType window_;
};

PitchShift::PitchShift(int32_t sample_rate, int32_t n_steps, int32_t bins_per_octave, int32_t n_fft, int32_t win_length,
                       int32_t hop_length, WindowType window)
    : data_(std::make_shared<Data>(sample_rate, n_steps, bins_per_octave, n_fft, win_length, hop_length, window)) {}

std::shared_ptr<TensorOperation> PitchShift::Parse() {
  return std::make_shared<PitchShiftOperation>(data_->sample_rate_, data_->n_steps_, data_->bins_per_octave_,
                                               data_->n_fft_, data_->win_length_, data_->hop_length_, data_->window_);
}

// Resample Transform Operation.
struct Resample::Data {
  Data(float orig_freq, float new_freq, ResampleMethod resample_method, int32_t lowpass_filter_width, float rolloff,
       float beta)
      : orig_freq_(orig_freq),
        new_freq_(new_freq),
        resample_method_(resample_method),
        lowpass_filter_width_(lowpass_filter_width),
        rolloff_(rolloff),
        beta_(beta) {}
  float orig_freq_;
  float new_freq_;
  ResampleMethod resample_method_;
  int32_t lowpass_filter_width_;
  float rolloff_;
  float beta_;
};

Resample::Resample(float orig_freq, float new_freq, ResampleMethod resample_method, int32_t lowpass_filter_width,
                   float rolloff, float beta)
    : data_(std::make_shared<Data>(orig_freq, new_freq, resample_method, lowpass_filter_width, rolloff, beta)) {}

std::shared_ptr<TensorOperation> Resample::Parse() {
  return std::make_shared<ResampleOperation>(data_->orig_freq_, data_->new_freq_, data_->resample_method_,
                                             data_->lowpass_filter_width_, data_->rolloff_, data_->beta_);
}

// RiaaBiquad Transform Operation.
struct RiaaBiquad::Data {
  explicit Data(int32_t sample_rate) : sample_rate_(sample_rate) {}
  int32_t sample_rate_;
};

RiaaBiquad::RiaaBiquad(int32_t sample_rate) : data_(std::make_shared<Data>(sample_rate)) {}

std::shared_ptr<TensorOperation> RiaaBiquad::Parse() {
  return std::make_shared<RiaaBiquadOperation>(data_->sample_rate_);
}

// SlidingWindowCmn Transform Operation.
struct SlidingWindowCmn::Data {
  Data(int32_t cmn_window, int32_t min_cmn_window, bool center, bool norm_vars)
      : cmn_window_(cmn_window), min_cmn_window_(min_cmn_window), center_(center), norm_vars_(norm_vars) {}
  int32_t cmn_window_;
  int32_t min_cmn_window_;
  bool center_;
  bool norm_vars_;
};

SlidingWindowCmn::SlidingWindowCmn(int32_t cmn_window, int32_t min_cmn_window, bool center, bool norm_vars)
    : data_(std::make_shared<Data>(cmn_window, min_cmn_window, center, norm_vars)) {}

std::shared_ptr<TensorOperation> SlidingWindowCmn::Parse() {
  return std::make_shared<SlidingWindowCmnOperation>(data_->cmn_window_, data_->min_cmn_window_, data_->center_,
                                                     data_->norm_vars_);
}

// Spectrogram Transform Operation.
struct Spectrogram::Data {
  Data(int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window, float power,
       bool normalized, bool center, BorderType pad_mode, bool onesided)
      : n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        pad_(pad),
        window_(window),
        power_(power),
        normalized_(normalized),
        center_(center),
        pad_mode_(pad_mode),
        onesided_(onesided) {}
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
  float power_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
};

Spectrogram::Spectrogram(int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window,
                         float power, bool normalized, bool center, BorderType pad_mode, bool onesided)
    : data_(std::make_shared<Data>(n_fft, win_length, hop_length, pad, window, power, normalized, center, pad_mode,
                                   onesided)) {}

std::shared_ptr<TensorOperation> Spectrogram::Parse() {
  return std::make_shared<SpectrogramOperation>(data_->n_fft_, data_->win_length_, data_->hop_length_, data_->pad_,
                                                data_->window_, data_->power_, data_->normalized_, data_->center_,
                                                data_->pad_mode_, data_->onesided_);
}

// SpectralCentroid Transform Operation.
struct SpectralCentroid::Data {
  Data(int32_t sample_rate, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window)
      : sample_rate_(sample_rate),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        pad_(pad),
        window_(window) {}
  int32_t sample_rate_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
};

SpectralCentroid::SpectralCentroid(int32_t sample_rate, int32_t n_fft, int32_t win_length, int32_t hop_length,
                                   int32_t pad, WindowType window)
    : data_(std::make_shared<Data>(sample_rate, n_fft, win_length, hop_length, pad, window)) {}

std::shared_ptr<TensorOperation> SpectralCentroid::Parse() {
  return std::make_shared<SpectralCentroidOperation>(data_->sample_rate_, data_->n_fft_, data_->win_length_,
                                                     data_->hop_length_, data_->pad_, data_->window_);
}

// TimeMasking Transform Operation.
struct TimeMasking::Data {
  Data(bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value)
      : iid_masks_(iid_masks), time_mask_param_(time_mask_param), mask_start_(mask_start), mask_value_(mask_value) {}
  bool iid_masks_;
  int32_t time_mask_param_;
  int32_t mask_start_;
  float mask_value_;
};

TimeMasking::TimeMasking(bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value)
    : data_(std::make_shared<Data>(iid_masks, time_mask_param, mask_start, mask_value)) {}

std::shared_ptr<TensorOperation> TimeMasking::Parse() {
  return std::make_shared<TimeMaskingOperation>(data_->iid_masks_, data_->time_mask_param_, data_->mask_start_,
                                                data_->mask_value_);
}

// TimeStretch Transform Operation.
struct TimeStretch::Data {
  explicit Data(float hop_length, int32_t n_freq, float fixed_rate)
      : hop_length_(hop_length), n_freq_(n_freq), fixed_rate_(fixed_rate) {}
  float hop_length_;
  int32_t n_freq_;
  float fixed_rate_;
};

TimeStretch::TimeStretch(float hop_length, int32_t n_freq, float fixed_rate)
    : data_(std::make_shared<Data>(hop_length, n_freq, fixed_rate)) {}

std::shared_ptr<TensorOperation> TimeStretch::Parse() {
  return std::make_shared<TimeStretchOperation>(data_->hop_length_, data_->n_freq_, data_->fixed_rate_);
}

// TrebleBiquad Transform Operation.
struct TrebleBiquad::Data {
  Data(int32_t sample_rate, float gain, float central_freq, float Q)
      : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float gain_;
  float central_freq_;
  float Q_;
};

TrebleBiquad::TrebleBiquad(int32_t sample_rate, float gain, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, gain, central_freq, Q)) {}

std::shared_ptr<TensorOperation> TrebleBiquad::Parse() {
  return std::make_shared<TrebleBiquadOperation>(data_->sample_rate_, data_->gain_, data_->central_freq_, data_->Q_);
}

// Vad Transform Operation.
struct Vad::Data {
  Data(int32_t sample_rate, float trigger_level, float trigger_time, float search_time, float allowed_gap,
       float pre_trigger_time, float boot_time, float noise_up_time, float noise_down_time,
       float noise_reduction_amount, float measure_freq, float measure_duration, float measure_smooth_time,
       float hp_filter_freq, float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq)
      : sample_rate_(sample_rate),
        trigger_level_(trigger_level),
        trigger_time_(trigger_time),
        search_time_(search_time),
        allowed_gap_(allowed_gap),
        pre_trigger_time_(pre_trigger_time),
        boot_time_(boot_time),
        noise_up_time_(noise_up_time),
        noise_down_time_(noise_down_time),
        noise_reduction_amount_(noise_reduction_amount),
        measure_freq_(measure_freq),
        measure_duration_(measure_duration),
        measure_smooth_time_(measure_smooth_time),
        hp_filter_freq_(hp_filter_freq),
        lp_filter_freq_(lp_filter_freq),
        hp_lifter_freq_(hp_lifter_freq),
        lp_lifter_freq_(lp_lifter_freq) {}
  int32_t sample_rate_;
  float trigger_level_;
  float trigger_time_;
  float search_time_;
  float allowed_gap_;
  float pre_trigger_time_;
  float boot_time_;
  float noise_up_time_;
  float noise_down_time_;
  float noise_reduction_amount_;
  float measure_freq_;
  float measure_duration_;
  float measure_smooth_time_;
  float hp_filter_freq_;
  float lp_filter_freq_;
  float hp_lifter_freq_;
  float lp_lifter_freq_;
};

Vad::Vad(int32_t sample_rate, float trigger_level, float trigger_time, float search_time, float allowed_gap,
         float pre_trigger_time, float boot_time, float noise_up_time, float noise_down_time,
         float noise_reduction_amount, float measure_freq, float measure_duration, float measure_smooth_time,
         float hp_filter_freq, float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq)
    : data_(std::make_shared<Data>(sample_rate, trigger_level, trigger_time, search_time, allowed_gap, pre_trigger_time,
                                   boot_time, noise_up_time, noise_down_time, noise_reduction_amount, measure_freq,
                                   measure_duration, measure_smooth_time, hp_filter_freq, lp_filter_freq,
                                   hp_lifter_freq, lp_lifter_freq)) {}

std::shared_ptr<TensorOperation> Vad::Parse() {
  return std::make_shared<VadOperation>(
    data_->sample_rate_, data_->trigger_level_, data_->trigger_time_, data_->search_time_, data_->allowed_gap_,
    data_->pre_trigger_time_, data_->boot_time_, data_->noise_up_time_, data_->noise_down_time_,
    data_->noise_reduction_amount_, data_->measure_freq_, data_->measure_duration_, data_->measure_smooth_time_,
    data_->hp_filter_freq_, data_->lp_filter_freq_, data_->hp_lifter_freq_, data_->lp_lifter_freq_);
}

// Vol Transform Operation.
struct Vol::Data {
  Data(float gain, GainType gain_type) : gain_(gain), gain_type_(gain_type) {}
  float gain_;
  GainType gain_type_;
};

Vol::Vol(float gain, GainType gain_type) : data_(std::make_shared<Data>(gain, gain_type)) {}

std::shared_ptr<TensorOperation> Vol::Parse() {
  return std::make_shared<VolOperation>(data_->gain_, data_->gain_type_);
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
