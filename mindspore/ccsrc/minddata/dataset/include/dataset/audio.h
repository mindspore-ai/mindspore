/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/transforms.h"

namespace mindspore {
namespace dataset {

class TensorOperation;

// Transform operations for performing computer audio.
namespace audio {

/// \brief Compute the angle of complex tensor input.
class Angle final : public TensorTransform {
 public:
  /// \brief Constructor.
  Angle();
  /// \brief Destructor.
  ~Angle() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Design two-pole band filter.
class BandBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  /// \param[in] noise Choose alternate mode for un-pitched audio or mode oriented to pitched audio(Default: False).
  explicit BandBiquad(int32_t sample_rate, float central_freq, float Q = 0.707, bool noise = false);

  /// \brief Destructor.
  ~BandBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design two-pole allpass filter. Similar to SoX implementation.
class AllpassBiquad final : public TensorTransform {
 public:
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  explicit AllpassBiquad(int32_t sample_rate, float central_freq, float Q = 0.707);

  /// \brief Destructor.
  ~AllpassBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief AmplitudeToDB TensorTransform.
/// \notes Turn a tensor from the power/amplitude scale to the decibel scale.
class AmplitudeToDB final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] stype Scale of input tensor, must be one of [ScaleType::kPower, ScaleType::kMagnitude] (Default:
  ///    ScaleType::kPower).
  /// \param[in] ref_value Calculate db_multiplier (Default: 1.0).
  /// \param[in] amin Minimum threshold for input tensor and ref_value (Default: 1e-10). It must be greater than zero.
  /// \param[in] top_db Decibels cut-off value (Default: 80.0). It must be greater than or equal to zero.
  explicit AmplitudeToDB(ScaleType stype = ScaleType::kPower, float ref_value = 1.0, float amin = 1e-10,
                         float top_db = 80.0);

  /// \brief Destructor.
  ~AmplitudeToDB() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design two-pole band-pass filter.
class BandpassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  /// \param[in] const_skirt_gain, If True, uses a constant skirt gain (peak gain = Q). If False, uses a
  ///     constant 0dB peak gain (Default: False).
  explicit BandpassBiquad(int32_t sample_rate, float central_freq, float Q = 0.707, bool const_skirt_gain = false);

  /// \brief Destructor.
  ~BandpassBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design two-pole band-reject filter. Similar to SoX implementation.
class BandrejectBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  explicit BandrejectBiquad(int32_t sample_rate, float central_freq, float Q = 0.707);

  /// \brief Destructor.
  ~BandrejectBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design a bass tone-control effect.
class BassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] gain Desired gain at the boost (or attenuation) in dB.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  explicit BassBiquad(int32_t sample_rate, float gain, float central_freq = 100, float Q = 0.707);

  /// \brief Destructor.
  ~BassBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Perform a biquad filter of input tensor.
class Biquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] b0 Numerator coefficient of current input, x[n].
  /// \param[in] b1 Numerator coefficient of input one time step ago x[n-1].
  /// \param[in] b2 Numerator coefficient of input two time steps ago x[n-2].
  /// \param[in] a0 Denominator coefficient of current output y[n], the value can't be zero, typically 1.
  /// \param[in] a1 Denominator coefficient of current output y[n-1].
  /// \param[in] a2 Denominator coefficient of current output y[n-2].
  explicit Biquad(float b0, float b1, float b2, float a0, float a1, float a2);

  /// \brief Destructor.
  ~Biquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief ComplexNorm TensorTransform.
/// \notes Compute the norm of complex tensor input.
class ComplexNorm final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] power Power of the norm, which must be non-negative (Default: 1.0).
  explicit ComplexNorm(float power = 1.0);

  /// \brief Destructor.
  ~ComplexNorm() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply contrast effect.
class Contrast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] enhancement_amount Controls the amount of the enhancement (Default: 75.0).
  explicit Contrast(float enhancement_amount = 75.0);

  /// \brief Destructor.
  ~Contrast() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply a DC shift to the audio.
class DCShift : public TensorTransform {
 public:
  /// \brief Constructor
  /// \param[in] shift Indicates the amount to shift the audio, the value must be in the range [-2.0, 2.0].
  /// \param[in] limiter_gain Used only on peaks to prevent clipping.
  DCShift(float shift, float limiter_gain);

  /// \brief Constructor
  /// \param[in] shift Indicates the amount to shift the audio.
  /// \note This constructor will use `shift` as `limiter_gain`.
  explicit DCShift(float shift);

  /// \brief Destructor.
  ~DCShift() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design two-pole deemph filter. Similar to SoX implementation.
class DeemphBiquad final : public TensorTransform {
 public:
  /// \param[in] sample_rate Sampling rate of the waveform, the value can only be 44100 (Hz) or 48000(hz).
  explicit DeemphBiquad(int32_t sample_rate);

  /// \brief Destructor.
  ~DeemphBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief EqualizerBiquad TensorTransform. Apply highpass biquad filter on audio.
class EqualizerBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] center_freq Filter's central frequency (in Hz).
  /// \param[in] gain Desired gain at the boost (or attenuation) in dB.
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  EqualizerBiquad(int32_t sample_rate, float center_freq, float gain, float Q = 0.707);

  /// \brief Destructor.
  ~EqualizerBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Add fade in or/and fade out on the input audio.
class Fade final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] fade_in_len Length of fade-in (time frames), which must be non-negative
  ///     and no more than the length of waveform (Default: 0).
  /// \param[in] fade_out_len Length of fade-out (time frames), which must be non-negative
  ///     and no more than the length of waveform (Default: 0).
  /// \param[in] fade_shape An enum for the fade shape (Default: FadeShape::kLinear).
  explicit Fade(int32_t fade_in_len = 0, int32_t fade_out_len = 0, FadeShape fade_shape = FadeShape::kLinear);

  /// \brief Destructor.
  ~Fade() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief FrequencyMasking TensorTransform.
/// \notes Apply masking to a spectrogram in the frequency domain.
class FrequencyMasking final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] iid_masks Whether to apply different masks to each example.
  /// \param[in] frequency_mask_param Maximum possible length of the mask, range: [0, freq_length] (Default: 0).
  ///     Indices uniformly sampled from [0, frequency_mask_param].
  ///     Mask width when iid_masks=true.
  /// \param[in] mask_start Mask start when iid_masks=true, range: [0, freq_length-frequency_mask_param] (Default: 0).
  /// \param[in] mask_value Mask value.
  explicit FrequencyMasking(bool iid_masks = false, int32_t frequency_mask_param = 0, int32_t mask_start = 0,
                            float mask_value = 0.0);

  /// \brief Destructor.
  ~FrequencyMasking() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief HighpassBiquad TensorTransform. Apply highpass biquad filter on audio.
class HighpassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] cutoff_freq Filter cutoff frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  HighpassBiquad(int32_t sample_rate, float cutoff_freq, float Q = 0.707);

  /// \brief Destructor.
  ~HighpassBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design filter. Similar to SoX implementation.
class LFilter final : public TensorTransform {
 public:
  /// \param[in] a_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [a0, a1, a2, ...].
  ///     Must be same size as b_coeffs (pad with 0's as necessary).
  /// \param[in] b_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [b0, b1, b2, ...].
  ///     Must be same size as a_coeffs (pad with 0's as necessary).
  /// \param[in] clamp If True, clamp the output signal to be in the range [-1, 1] (Default: True).
  explicit LFilter(std::vector<float> a_coeffs, std::vector<float> b_coeffs, bool clamp = true);

  /// \brief Destructor.
  ~LFilter() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design biquad lowpass filter and perform filtering. Similar to SoX implementation.
class LowpassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] cutoff_freq Filter cutoff frequency.
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  LowpassBiquad(int32_t sample_rate, float cutoff_freq, float Q = 0.707);

  /// \brief Destructor.
  ~LowpassBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Separate a complex-valued spectrogram with shape (..., 2) into its magnitude and phase.
class Magphase final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] power Power of the norm, which must be non-negative (Default: 1.0).
  explicit Magphase(float power);

  /// \brief Destructor.
  ~Magphase() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief MuLawDecoding TensorTransform.
/// \note Decode mu-law encoded signal.
class MuLawDecoding final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] quantization_channels Number of channels, which must be positive (Default: 256).
  explicit MuLawDecoding(int quantization_channels = 256);

  /// \brief Destructor.
  ~MuLawDecoding() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief TimeMasking TensorTransform.
/// \notes Apply masking to a spectrogram in the time domain.
class TimeMasking final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] iid_masks Whether to apply different masks to each example.
  /// \param[in] time_mask_param Maximum possible length of the mask, range: [0, time_length] (Default: 0).
  ///     Indices uniformly sampled from [0, time_mask_param].
  ///     Mask width when iid_masks=true.
  /// \param[in] mask_start Mask start when iid_masks=true, range: [0, time_length-time_mask_param] (Default: 0).
  /// \param[in] mask_value Mask value.
  explicit TimeMasking(bool iid_masks = false, int32_t time_mask_param = 0, int32_t mask_start = 0,
                       float mask_value = 0.0);

  /// \brief Destructor.
  ~TimeMasking() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief TimeStretch TensorTransform
/// \notes Stretch STFT in time at a given rate, without changing the pitch.
class TimeStretch final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] hop_length Length of hop between STFT windows. Default: None.
  /// \param[in] n_freq Number of filter banks form STFT. Default: 201.
  /// \param[in] fixed_rate Rate to speed up or slow down the input in time. Default: None.
  explicit TimeStretch(float hop_length = std::numeric_limits<float>::quiet_NaN(), int n_freq = 201,
                       float fixed_rate = std::numeric_limits<float>::quiet_NaN());

  /// \brief Destructor.
  ~TimeStretch() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Vol TensorTransform.
/// \notes Add a volume to an waveform.
class Vol final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gain Gain value, varies according to the value of gain_type. If gain_type is GainType::kAmplitude,
  ///    gain must be greater than or equal to zero. If gain_type is GainType::kPower, gain must be greater than zero.
  ///    If gain_type is GainType::kDb, there is no limit for gain.
  /// \param[in] gain_type Type of gain, should be one of [GainType::kAmplitude, GainType::kDb, GainType::kPower].
  explicit Vol(float gain, GainType gain_type = GainType::kAmplitude);

  /// \brief Destructor.
  ~Vol() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
