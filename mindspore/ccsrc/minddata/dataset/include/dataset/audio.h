/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "utils/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/ms_tensor.h"
#include "include/dataset/transforms.h"
#include "minddata/dataset/util/visible.h"

namespace mindspore::dataset {
class TensorOperation;

// Transform operations for performing computer audio.
namespace audio {

/// \brief Design two-pole allpass filter. Similar to SoX implementation.
class DATASET_API AllpassBiquad final : public TensorTransform {
 public:
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  explicit AllpassBiquad(int32_t sample_rate, float central_freq, float Q = 0.707);

  /// \brief Destructor.
  ~AllpassBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design two-pole band filter.
class DATASET_API BandBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  /// \param[in] noise Choose alternate mode for un-pitched audio or mode oriented to pitched audio. Default: False.
  explicit BandBiquad(int32_t sample_rate, float central_freq, float Q = 0.707, bool noise = false);

  /// \brief Destructor.
  ~BandBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design two-pole band-pass filter.
class DATASET_API BandpassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  /// \param[in] const_skirt_gain, If True, uses a constant skirt gain (peak gain = Q). If False, uses a
  ///     constant 0dB peak gain. Default: False.
  explicit BandpassBiquad(int32_t sample_rate, float central_freq, float Q = 0.707, bool const_skirt_gain = false);

  /// \brief Destructor.
  ~BandpassBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design two-pole band-reject filter. Similar to SoX implementation.
class DATASET_API BandrejectBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  explicit BandrejectBiquad(int32_t sample_rate, float central_freq, float Q = 0.707);

  /// \brief Destructor.
  ~BandrejectBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design a bass tone-control effect.
class DATASET_API BassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] gain Desired gain at the boost (or attenuation) in dB.
  /// \param[in] central_freq Central frequency (in Hz). Default: 100.0.
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  explicit BassBiquad(int32_t sample_rate, float gain, float central_freq = 100.0, float Q = 0.707);

  /// \brief Destructor.
  ~BassBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Perform a biquad filter of input tensor.
class DATASET_API Biquad final : public TensorTransform {
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
  ~Biquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \param[in] n_mfcc Number of mfc coefficients to retain, the value must be greater than 0.
/// \param[in] n_mels Number of mel filterbanks, the value must be greater than 0.
/// \param[in] norm Norm to use, can be NormMode::kNone or NormMode::kOrtho.
/// \return Status error code, returns OK if no error encountered.
Status CreateDct(mindspore::MSTensor *output, int32_t n_mfcc, int32_t n_mels, NormMode norm = NormMode::kNone);

/// \brief Design two-pole deemph filter. Similar to SoX implementation.
class DATASET_API DeemphBiquad final : public TensorTransform {
 public:
  /// \param[in] sample_rate Sampling rate of the waveform, the value can only be 44100 (Hz) or 48000(hz).
  explicit DeemphBiquad(int32_t sample_rate);

  /// \brief Destructor.
  ~DeemphBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief EqualizerBiquad TensorTransform. Apply highpass biquad filter on audio.
class DATASET_API EqualizerBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] center_freq Filter's central frequency (in Hz).
  /// \param[in] gain Desired gain at the boost (or attenuation) in dB.
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  EqualizerBiquad(int32_t sample_rate, float center_freq, float gain, float Q = 0.707);

  /// \brief Destructor.
  ~EqualizerBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design IIR forward and backward filter.
class DATASET_API Filtfilt final : public TensorTransform {
 public:
  /// \param[in] a_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [a0, a1, a2, ...].
  ///     Must be same size as b_coeffs (pad with 0's as necessary).
  /// \param[in] b_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [b0, b1, b2, ...].
  ///     Must be same size as a_coeffs (pad with 0's as necessary).
  /// \param[in] clamp If True, clamp the output signal to be in the range [-1, 1]. Default: True.
  Filtfilt(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp = true);

  /// \brief Destructor.
  ~Filtfilt() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief HighpassBiquad TensorTransform. Apply highpass biquad filter on audio.
class DATASET_API HighpassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] cutoff_freq Filter cutoff frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  HighpassBiquad(int32_t sample_rate, float cutoff_freq, float Q = 0.707);

  /// \brief Destructor.
  ~HighpassBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design filter. Similar to SoX implementation.
class DATASET_API LFilter final : public TensorTransform {
 public:
  /// \param[in] a_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [a0, a1, a2, ...].
  ///     Must be same size as b_coeffs (pad with 0's as necessary).
  /// \param[in] b_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [b0, b1, b2, ...].
  ///     Must be same size as a_coeffs (pad with 0's as necessary).
  /// \param[in] clamp If True, clamp the output signal to be in the range [-1, 1]. Default: True.
  explicit LFilter(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp = true);

  /// \brief Destructor.
  ~LFilter() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Creates a linear triangular filterbank.
/// \param output Tensor of a linear triangular filterbank.
/// \param n_freqs: Number of frequency.
/// \param f_min: Minimum of frequency in Hz.
/// \param f_max: Maximum of frequency in Hz.
/// \param n_filter: Number of (linear) triangular filter.
/// \param sample_rate: Sample rate.
/// \return Status code.
Status DATASET_API LinearFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_filter,
                                int32_t sample_rate);

/// \brief Design biquad lowpass filter and perform filtering. Similar to SoX implementation.
class DATASET_API LowpassBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] cutoff_freq Filter cutoff frequency.
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  LowpassBiquad(int32_t sample_rate, float cutoff_freq, float Q = 0.707);

  /// \brief Destructor.
  ~LowpassBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Create a frequency transformation matrix with shape (n_freqs, n_mels).
/// \param[in] output Tensor of the frequency transformation matrix.
/// \param[in] n_freqs Number of frequencies to highlight/apply.
/// \param[in] f_min Minimum frequency (Hz).
/// \param[in] f_max Maximum frequency (Hz).
/// \param[in] n_mels Number of mel filterbanks.
/// \param[in] sample_rate Sample rate of the audio waveform.
/// \param[in] norm Norm to use, can be NormType::kNone or NormType::kSlaney. Default: NormType::kNone.
/// \param[in] mel_type Scale to use, can be MelType::kHtk or MelType::kSlaney. Default: MelType::kHtz.
/// \return Status code.
Status DATASET_API MelscaleFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels,
                                  int32_t sample_rate, NormType norm = NormType::kNone,
                                  MelType mel_type = MelType::kHtk);

// \brief Shift the pitch of a waveform by 'n_steps' steps.

/// \brief Apply RIAA vinyl playback equalization.
class DATASET_API RiaaBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz),
  ///     can only be one of 44100, 48000, 88200, 96000.
  explicit RiaaBiquad(int32_t sample_rate);

  /// \brief Destructor.
  ~RiaaBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Design a treble tone-control effect.
class DATASET_API TrebleBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] gain Desired gain at the boost (or attenuation) in dB.
  /// \param[in] central_freq Central frequency (in Hz). Default: 3000.0.
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1]. Default: 0.707.
  TrebleBiquad(int32_t sample_rate, float gain, float central_freq = 3000.0, float Q = 0.707);

  /// \brief Destructor.
  ~TrebleBiquad() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

}  // namespace audio
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
