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
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
