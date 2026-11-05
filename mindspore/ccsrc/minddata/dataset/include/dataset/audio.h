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

/// \brief Apply contrast effect.
class DATASET_API Contrast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] enhancement_amount Controls the amount of the enhancement. Default: 75.0.
  explicit Contrast(float enhancement_amount = 75.0);

  /// \brief Destructor.
  ~Contrast() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply a DC shift to the audio.
class DATASET_API DCShift : public TensorTransform {
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
  ~DCShift() override = default;

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

/// \brief Dither increases the perceived dynamic range of audio stored at a
///     particular bit-depth by eliminating nonlinear truncation distortion.
class DATASET_API Dither final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] density_function The density function of a continuous random variable.
  ///     Can be one of DensityFunction::kTPDF (Triangular Probability Density Function),
  ///     DensityFunction::kRPDF (Rectangular Probability Density Function) or
  ///     DensityFunction::kGPDF (Gaussian Probability Density Function). Default: DensityFunction::kTPDF.
  /// \param[in] noise_shaping A filtering process that shapes the spectral energy of
  ///     quantisation error. Default: false.
  explicit Dither(DensityFunction density_function = DensityFunction::kTPDF, bool noise_shaping = false);

  /// \brief Destructor.
  ~Dither() override = default;

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

/// \brief Add fade in or/and fade out on the input audio.
class DATASET_API Fade final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] fade_in_len Length of fade-in (time frames), which must be non-negative
  ///     and no more than the length of waveform. Default: 0.
  /// \param[in] fade_out_len Length of fade-out (time frames), which must be non-negative
  ///     and no more than the length of waveform. Default: 0.
  /// \param[in] fade_shape An enum for the fade shape. Default: FadeShape::kLinear.
  explicit Fade(int32_t fade_in_len = 0, int32_t fade_out_len = 0, FadeShape fade_shape = FadeShape::kLinear);

  /// \brief Destructor.
  ~Fade() override = default;

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

/// \brief Apply a flanger effect to the audio.
class DATASET_API Flanger final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] delay Desired delay in milliseconds (ms), range: [0, 30]. Default: 0.0.
  /// \param[in] depth Desired delay depth in milliseconds (ms), range: [0, 10]. Default: 2.0.
  /// \param[in] regen Desired regen (feedback gain) in dB., range: [-95, 95]. Default: 0.0.
  /// \param[in] width Desired width (delay gain) in dB, range: [0, 100]. Default: 71.0.
  /// \param[in] speed Modulation speed in Hz, range: [0.1, 10]. Default: 0.5.
  /// \param[in] phase Percentage phase-shift for multi-channel, range: [0, 100]. Default: 25.0.
  /// \param[in] modulation Modulation of input tensor, must be one of [Modulation::kSinusoidal,
  ///     Modulation::kTriangular]. Default:Modulation::kSinusoidal.
  /// \param[in] interpolation Interpolation of input tensor, must be one of [Interpolation::kLinear,
  ///     Interpolation::kQuadratic]. Default:Interpolation::kLinear.
  explicit Flanger(int32_t sample_rate, float delay = 0.0, float depth = 2.0, float regen = 0.0, float width = 71.0,
                   float speed = 0.5, float phase = 25.0, Modulation modulation = Modulation::kSinusoidal,
                   Interpolation interpolation = Interpolation::kLinear);

  /// \brief Destructor.
  ~Flanger() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply amplification or attenuation to the whole waveform.
class DATASET_API Gain final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gain_db Gain adjustment in decibels (dB). Default: 1.0.
  explicit Gain(float gain_db = 1.0);

  /// \brief Destructor.
  ~Gain() override = default;

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

/// \brief MuLawDecoding TensorTransform.
/// \note Decode mu-law encoded signal.
class DATASET_API MuLawDecoding final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] quantization_channels Number of channels, which must be positive. Default: 256.
  explicit MuLawDecoding(int32_t quantization_channels = 256);

  /// \brief Destructor.
  ~MuLawDecoding() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief MuLawEncoding TensorTransform.
/// \note Encode signal based on mu-law companding.
class DATASET_API MuLawEncoding final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] quantization_channels Number of channels, which must be positive. Default: 256.
  explicit MuLawEncoding(int32_t quantization_channels = 256);

  /// \brief Destructor.
  ~MuLawEncoding() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Overdrive TensorTransform.
class DATASET_API Overdrive final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gain Coefficient of overload in dB, in range of [0, 100]. Default: 20.0.
  /// \param[in] color Coefficient of translation, in range of [0, 100]. Default: 20.0.
  explicit Overdrive(float gain = 20.0, float color = 20.0);

  /// \brief Destructor.
  ~Overdrive() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Phaser TensorTransform.
class DATASET_API Phaser final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] gain_in Desired input gain at the boost (or attenuation) in dB.
  ///     Allowed range of values is [0, 1]. Default: 0.4.
  /// \param[in] gain_out Desired output gain at the boost (or attenuation) in dB.
  ///     Allowed range of values is [0, 1e9]. Default: 0.74.
  /// \param[in] delay_ms Desired delay in milli seconds. Allowed range of values is [0, 5]. Default: 3.0.
  /// \param[in] decay Desired decay relative to gain-in. Allowed range of values is [0, 0.99]. Default: 0.4.
  /// \param[in] mod_speed Modulation speed in Hz. Allowed range of values is [0.1, 2]. Default: 0.5.
  /// \param[in] sinusoidal If true, use sinusoidal modulation (preferable for multiple instruments).
  ///     If false, use triangular modulation (gives single instruments a sharper phasing effect). Default: true.
  explicit Phaser(int32_t sample_rate, float gain_in = 0.4, float gain_out = 0.74, float delay_ms = 3.0,
                  float decay = 0.4, float mod_speed = 0.5, bool sinusoidal = true);

  /// \brief Destructor.
  ~Phaser() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

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

/// \brief Vad TensorTransform.
/// \notes Attempt to trim silent background sounds from the end of the voice recording.
class DATASET_API Vad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sample rate of audio signal.
  /// \param[in] trigger_level The measurement level used to trigger activity detection. Default: 7.0.
  /// \param[in] trigger_time The time constant (in seconds) used to help ignore short sounds. Default: 0.25.
  /// \param[in] search_time The amount of audio (in seconds) to search for quieter/shorter sounds to include prior to
  ///     the detected trigger point. Default: 1.0.
  /// \param[in] allowed_gap The allowed gap (in seconds) between quiteter/shorter sounds to include prior to the
  ///     detected trigger point. Default: 0.25.
  /// \param[in] pre_trigger_time The amount of audio (in seconds) to preserve before the trigger point and any found
  ///     quieter/shorter bursts. Default: 0.0.
  /// \param[in] boot_time The time for the initial noise estimate. Default: 0.35.
  /// \param[in] noise_up_time Time constant used by the adaptive noise estimator, when the noise level is increasing.
  ///     Default: 0.1.
  /// \param[in] noise_down_time Time constant used by the adaptive noise estimator, when the noise level is decreasing.
  ///     Default: 0.01.
  /// \param[in] noise_reduction_amount The amount of noise reduction used in the detection algorithm. Default: 1.35.
  /// \param[in] measure_freq The frequency of the algorithm’s processing. Default: 20.0.
  /// \param[in] measure_duration The duration of measurement. Default: 0, use twice the measurement period.
  /// \param[in] measure_smooth_time The time constant used to smooth spectral measurements. Default: 0.4.
  /// \param[in] hp_filter_freq The "Brick-wall" frequency of high-pass filter applied at the input to the detector
  ///     algorithm. Default: 50.0.
  /// \param[in] lp_filter_freq The "Brick-wall" frequency of low-pass filter applied at the input to the detector
  ///     algorithm. Default: 6000.0.
  /// \param[in] hp_lifter_freq The "Brick-wall" frequency of high-pass lifter applied at the input to the detector
  ///     algorithm. Default: 150.0.
  /// \param[in] lp_lifter_freq The "Brick-wall" frequency of low-pass lifter applied at the input to the detector
  ///     algorithm. Default: 2000.0.
  explicit Vad(int32_t sample_rate, float trigger_level = 7.0, float trigger_time = 0.25, float search_time = 1.0,
               float allowed_gap = 0.25, float pre_trigger_time = 0.0, float boot_time = 0.35,
               float noise_up_time = 0.1, float noise_down_time = 0.01, float noise_reduction_amount = 1.35,
               float measure_freq = 20.0, float measure_duration = 0.0, float measure_smooth_time = 0.4,
               float hp_filter_freq = 50.0, float lp_filter_freq = 6000.0, float hp_lifter_freq = 150.0,
               float lp_lifter_freq = 2000.0);

  /// \brief Destructor.
  ~Vad() override = default;

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
class DATASET_API Vol final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gain Gain value, varies according to the value of gain_type. If gain_type is GainType::kAmplitude,
  ///    gain must be greater than or equal to zero. If gain_type is GainType::kPower, gain must be greater than zero.
  ///    If gain_type is GainType::kDb, there is no limit for gain.
  /// \param[in] gain_type Type of gain, should be one of [GainType::kAmplitude, GainType::kDb, GainType::kPower].
  explicit Vol(float gain, GainType gain_type = GainType::kAmplitude);

  /// \brief Destructor.
  ~Vol() override = default;

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
