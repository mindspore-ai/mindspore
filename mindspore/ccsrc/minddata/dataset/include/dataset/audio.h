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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"

namespace mindspore {
namespace dataset {
class TensorOperation;

// Transform operations for performing computer audio.
namespace audio {
/// \brief Compute the angle of complex tensor input.
class MS_API Angle final : public TensorTransform {
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

/// \brief Design two-pole allpass filter. Similar to SoX implementation.
class MS_API AllpassBiquad final : public TensorTransform {
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
class MS_API AmplitudeToDB final : public TensorTransform {
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

/// \brief Design two-pole band filter.
class MS_API BandBiquad final : public TensorTransform {
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

/// \brief Design two-pole band-pass filter.
class MS_API BandpassBiquad final : public TensorTransform {
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
class MS_API BandrejectBiquad final : public TensorTransform {
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
class MS_API BassBiquad final : public TensorTransform {
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
class MS_API Biquad final : public TensorTransform {
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
class MS_API ComplexNorm final : public TensorTransform {
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

/// \brief ComputeDeltas Transform.
/// \note Compute delta coefficients of a spectrogram.
class MS_API ComputeDeltas final : public TensorTransform {
 public:
  /// \brief Construct a new Compute Deltas object.
  /// \param[in] win_length The window length used for computing delta, must be no less than 3 (Default: 5).
  /// \param[in] pad_mode Mode parameter passed to padding (Default: BorderType::kEdge).
  explicit ComputeDeltas(int32_t win_length = 5, BorderType pad_mode = BorderType::kEdge);

  /// \brief Destructor.
  ~ComputeDeltas() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply contrast effect.
class MS_API Contrast final : public TensorTransform {
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

/// \brief Turn a waveform from the decibel scale to the power/amplitude scale.
class MS_API DBToAmplitude final : public TensorTransform {
 public:
  /// \brief Constructor
  /// \param[in] ref Reference which the output will be scaled by.
  /// \param[in] power If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.
  explicit DBToAmplitude(float ref, float power);

  /// \brief Destructor.
  ~DBToAmplitude() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply a DC shift to the audio.
class MS_API DCShift : public TensorTransform {
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

/// \param[in] n_mfcc Number of mfc coefficients to retain, the value must be greater than 0.
/// \param[in] n_mels Number of mel filterbanks, the value must be greater than 0.
/// \param[in] norm Norm to use, can be NormMode::kNone or NormMode::kOrtho.
/// \return Status error code, returns OK if no error encountered.
Status CreateDct(mindspore::MSTensor *output, int32_t n_mfcc, int32_t n_mels, NormMode norm = NormMode::kNone);

/// \brief Design two-pole deemph filter. Similar to SoX implementation.
class MS_API DeemphBiquad final : public TensorTransform {
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

/// \brief Detect pitch frequency.
class MS_API DetectPitchFrequency final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] frame_time Duration of a frame, the value must be greater than zero (default=0.02).
  /// \param[in] win_length The window length for median smoothing (in number of frames), the value must
  ///     be greater than zero (default=30).
  /// \param[in] freq_low Lowest frequency that can be detected (Hz), the value must be greater than zero (default=85).
  /// \param[in] freq_high Highest frequency that can be detected (Hz), the value must be greater than
  ///     zero (default=3400).
  explicit DetectPitchFrequency(int32_t sample_rate, float frame_time = 0.01, int32_t win_length = 30,
                                int32_t freq_low = 85, int32_t freq_high = 3400);

  /// \brief Destructor.
  ~DetectPitchFrequency() = default;

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
class MS_API Dither final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] density_function The density function of a continuous random variable.
  ///     Can be one of DensityFunction::kTPDF (Triangular Probability Density Function),
  ///     DensityFunction::kRPDF (Rectangular Probability Density Function) or
  ///     DensityFunction::kGPDF (Gaussian Probability Density Function) (Default: DensityFunction::kTPDF).
  /// \param[in] noise_shaping A filtering process that shapes the spectral energy of
  ///     quantisation error (Default: false).
  explicit Dither(DensityFunction density_function = DensityFunction::kTPDF, bool noise_shaping = false);

  /// \brief Destructor.
  ~Dither() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief EqualizerBiquad TensorTransform. Apply highpass biquad filter on audio.
class MS_API EqualizerBiquad final : public TensorTransform {
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
class MS_API Fade final : public TensorTransform {
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

/// \brief Apply a flanger effect to the audio.
class MS_API Flanger final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] delay Desired delay in milliseconds (ms), range: [0, 30] (Default: 0.0).
  /// \param[in] depth Desired delay depth in milliseconds (ms), range: [0, 10] (Default: 2.0).
  /// \param[in] regen Desired regen (feedback gain) in dB., range: [-95, 95] (Default: 0.0).
  /// \param[in] width Desired width (delay gain) in dB, range: [0, 100] (Default: 71.0).
  /// \param[in] speed Modulation speed in Hz, range: [0.1, 10] (Default: 0.5).
  /// \param[in] phase Percentage phase-shift for multi-channel, range: [0, 100] (Default: 25.0).
  /// \param[in] modulation Modulation of input tensor, must be one of [Modulation::kSinusoidal,
  ///     Modulation::kTriangular] (Default:Modulation::kSinusoidal).
  /// \param[in] interpolation Interpolation of input tensor, must be one of [Interpolation::kLinear,
  ///     Interpolation::kQuadratic] (Default:Interpolation::kLinear).
  explicit Flanger(int32_t sample_rate, float delay = 0.0, float depth = 2.0, float regen = 0.0, float width = 71.0,
                   float speed = 0.5, float phase = 25.0, Modulation modulation = Modulation::kSinusoidal,
                   Interpolation interpolation = Interpolation::kLinear);

  /// \brief Destructor.
  ~Flanger() = default;

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
class MS_API FrequencyMasking final : public TensorTransform {
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

/// \brief Apply amplification or attenuation to the whole waveform.
class MS_API Gain final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gain_db Gain adjustment in decibels (dB) (Default: 1.0).
  explicit Gain(float gain_db = 1.0);

  /// \brief Destructor.
  ~Gain() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief HighpassBiquad TensorTransform. Apply highpass biquad filter on audio.
class MS_API HighpassBiquad final : public TensorTransform {
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
class MS_API LFilter final : public TensorTransform {
 public:
  /// \param[in] a_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [a0, a1, a2, ...].
  ///     Must be same size as b_coeffs (pad with 0's as necessary).
  /// \param[in] b_coeffs Numerator coefficients of difference equation of dimension of (n_order + 1).
  ///     Lower delays coefficients are first, e.g. [b0, b1, b2, ...].
  ///     Must be same size as a_coeffs (pad with 0's as necessary).
  /// \param[in] clamp If True, clamp the output signal to be in the range [-1, 1] (Default: True).
  explicit LFilter(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp = true);

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
class MS_API LowpassBiquad final : public TensorTransform {
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
class MS_API Magphase final : public TensorTransform {
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
class MS_API MuLawDecoding final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] quantization_channels Number of channels, which must be positive (Default: 256).
  explicit MuLawDecoding(int32_t quantization_channels = 256);

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

/// \brief MuLawEncoding TensorTransform.
/// \note Encode signal based on mu-law companding.
class MS_API MuLawEncoding final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] quantization_channels Number of channels, which must be positive (Default: 256).
  explicit MuLawEncoding(int32_t quantization_channels = 256);

  /// \brief Destructor.
  ~MuLawEncoding() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Overdrive TensorTransform.
class MS_API Overdrive final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gain Coefficient of overload in dB, in range of [0, 100] (Default: 20.0).
  /// \param[in] color Coefficient of translation, in range of [0, 100] (Default: 20.0).
  explicit Overdrive(float gain = 20.0f, float color = 20.0f);

  /// \brief Destructor.
  ~Overdrive() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Phaser TensorTransform.
class MS_API Phaser final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] gain_in Desired input gain at the boost (or attenuation) in dB.
  ///     Allowed range of values is [0, 1] (Default=0.4).
  /// \param[in] gain_out Desired output gain at the boost (or attenuation) in dB.
  ///     Allowed range of values is [0, 1e9] (Default=0.74).
  /// \param[in] delay_ms Desired delay in milli seconds. Allowed range of values is [0, 5] (Default=3.0).
  /// \param[in] decay Desired decay relative to gain-in. Allowed range of values is [0, 0.99] (Default=0.4).
  /// \param[in] mod_speed Modulation speed in Hz. Allowed range of values is [0.1, 2] (Default=0.5).
  /// \param[in] sinusoidal If true, use sinusoidal modulation (preferable for multiple instruments).
  ///     If false, use triangular modulation (gives single instruments a sharper phasing effect) (Default=true).
  explicit Phaser(int32_t sample_rate, float gain_in = 0.4f, float gain_out = 0.74f, float delay_ms = 3.0f,
                  float decay = 0.4f, float mod_speed = 0.5f, bool sinusoidal = true);

  /// \brief Destructor.
  ~Phaser() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply RIAA vinyl playback equalization.
class MS_API RiaaBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz),
  ///     can only be one of 44100, 48000, 88200, 96000.
  explicit RiaaBiquad(int32_t sample_rate);

  /// \brief Destructor.
  ~RiaaBiquad() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.
class MS_API SlidingWindowCmn final : public TensorTransform {
 public:
  /// \brief Constructor of SlidingWindowCmnOp.
  /// \param[in] cmn_window The window in frames for running average CMN computation (Default: 600).
  /// \param[in] min_cmn_window The minimum CMN window. Only applicable if center is false, ignored if center
  ///      is true (Default: 100).
  /// \param[in] center If true, use a window centered on the current frame. If false, window is to the left
  ///     (Default: false).
  /// \param[in] norm_vars If true, normalize variance to one (Default: false).
  explicit SlidingWindowCmn(int32_t cmn_window = 600, int32_t min_cmn_window = 100, bool center = false,
                            bool norm_vars = false);

  /// \brief Destructor.
  ~SlidingWindowCmn() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Create a spectral centroid from an audio signal.
class MS_API SpectralCentroid : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz).
  /// \param[in] n_fft Size of FFT, creates n_fft / 2 + 1 bins (Default: 400).
  /// \param[in] win_length Window size (Default: 0, will use n_fft).
  /// \param[in] hop_length Length of hop between STFT windows (Default: 0, will use win_length / 2).
  /// \param[in] pad Two sided padding of signal (Default: 0).
  /// \param[in] window Window function that is applied/multiplied to each frame/window,
  ///     which can be WindowType::kBartlett, WindowType::kBlackman, WindowType::kHamming,
  ///     WindowType::kHann or WindowType::kKaiser (Default: WindowType::kHann).
  explicit SpectralCentroid(int32_t sample_rate, int32_t n_fft = 400, int32_t win_length = 0, int32_t hop_length = 0,
                            int32_t pad = 0, WindowType window = WindowType::kHann);

  ~SpectralCentroid() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  int32_t sample_rate_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Create a spectrogram from an audio signal.
class MS_API Spectrogram : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] n_fft Size of FFT, creates n_fft / 2 + 1 bins (Default: 400).
  /// \param[in] win_length Window size (Default: 0, will use n_fft).
  /// \param[in] hop_length Length of hop between STFT windows (Default: 0, will use win_length / 2).
  /// \param[in] pad Two sided padding of signal (Default: 0).
  /// \param[in] window Window function that is applied/multiplied to each frame/window,
  ///     which can be WindowType::kBartlett, WindowType::kBlackman, WindowType::kHamming,
  ///     WindowType::kHann or WindowType::kKaiser (Default: WindowType::kHann).
  /// \param[in] power Exponent for the magnitude spectrogram, which must be greater than or equal to 0 (Default: 2.0).
  /// \param[in] normalized Whether to normalize by magnitude after stft (Default: false).
  /// \param[in] center Whether to pad waveform on both sides (Default: true).
  /// \param[in] pad_mode Controls the padding method used when center is true (Default: BorderType::kReflect).
  /// \param[in] onesided Controls whether to return half of results to avoid redundancy (Default: true).
  explicit Spectrogram(int32_t n_fft = 400, int32_t win_length = 0, int32_t hop_length = 0, int32_t pad = 0,
                       WindowType window = WindowType::kHann, float power = 2.0, bool normalized = false,
                       bool center = true, BorderType pad_mode = BorderType::kReflect, bool onesided = true);

  /// \brief Destructor.
  ~Spectrogram() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
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
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief TimeMasking TensorTransform.
/// \notes Apply masking to a spectrogram in the time domain.
class MS_API TimeMasking final : public TensorTransform {
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
class MS_API TimeStretch final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] hop_length Length of hop between STFT windows (Default: None, will use ((n_freq - 1) * 2) // 2).
  /// \param[in] n_freq Number of filter banks form STFT (Default: 201).
  /// \param[in] fixed_rate Rate to speed up or slow down the input in time
  ///     (Default: std::numeric_limits<float>::quiet_NaN(), will keep the original rate).
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

/// \brief Design a treble tone-control effect.
class MS_API TrebleBiquad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] gain Desired gain at the boost (or attenuation) in dB.
  /// \param[in] central_freq Central frequency (in Hz) (Default: 3000).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  TrebleBiquad(int32_t sample_rate, float gain, float central_freq = 3000, float Q = 0.707);

  /// \brief Destructor.
  ~TrebleBiquad() = default;

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
class MS_API Vol final : public TensorTransform {
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
