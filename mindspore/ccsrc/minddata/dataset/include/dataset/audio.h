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
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor (Default: 0.707).
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
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q https://en.wikipedia.org/wiki/Q_factor (Default: 0.707).
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
  /// \param[in] stype ['kPower', 'kMagnitude'].
  /// \param[in] ref_value Calculate db_multiplier.
  /// \param[in] amin Clamp the input waveform.
  /// \param[in] top_db Decibels cut-off value.
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
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor (Default: 0.707).
  /// \param[in] const_skirt_gain, If ``True``, uses a constant skirt gain (peak gain = Q). If ``False``, uses a
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
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor (Default: 0.707).
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
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] gain Desired gain at the boost (or attenuation) in dB.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q https://en.wikipedia.org/wiki/Q_factor (Default: 0.707).
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

/// \brief FrequencyMasking TensorTransform.
/// \notes Apply masking to a spectrogram in the frequency domain.
class FrequencyMasking final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] iid_masks Whether to apply different masks to each example.
  /// \param[in] frequency_mask_param Maximum possible length of the mask.
  ///     Indices uniformly sampled from [0, frequency_mask_param].
  ///     Mask width when iid_masks=true.
  /// \param[in] mask_start Mask start when iid_masks=true.
  /// \param[in] mask_value Mask value.
  explicit FrequencyMasking(bool iid_masks = false, int32_t frequency_mask_param = 0, int32_t mask_start = 0,
                            double mask_value = 0.0);

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

/// \brief TimeMasking TensorTransform.
/// \notes Apply masking to a spectrogram in the time domain.
class TimeMasking final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] iid_masks Whether to apply different masks to each example.
  /// \param[in] time_mask_param Maximum possible length of the mask.
  ///     Indices uniformly sampled from [0, time_mask_param].
  ///     Mask width when iid_masks=true.
  /// \param[in] mask_start Mask start when iid_masks=true.
  /// \param[in] mask_value Mask value.
  explicit TimeMasking(bool iid_masks = false, int64_t time_mask_param = 0, int64_t mask_start = 0,
                       double mask_value = 0.0);

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
