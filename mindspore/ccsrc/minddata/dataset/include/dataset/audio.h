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
#include <map>
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
class DATASET_API Angle final : public TensorTransform {
 public:
  /// \brief Constructor.
  Angle();

  /// \brief Destructor.
  ~Angle() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Design two-pole allpass filter. Similar to SoX implementation.
class DATASET_API AllpassBiquad final : public TensorTransform {
 public:
  /// \param[in] sample_rate Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
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

/// \brief AmplitudeToDB TensorTransform.
/// \notes Turn a tensor from the power/amplitude scale to the decibel scale.
class DATASET_API AmplitudeToDB final : public TensorTransform {
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
  ~AmplitudeToDB() override = default;

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
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  /// \param[in] noise Choose alternate mode for un-pitched audio or mode oriented to pitched audio(Default: False).
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
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  /// \param[in] const_skirt_gain, If True, uses a constant skirt gain (peak gain = Q). If False, uses a
  ///     constant 0dB peak gain (Default: False).
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
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
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
  /// \param[in] central_freq Central frequency (in Hz).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  explicit BassBiquad(int32_t sample_rate, float gain, float central_freq = 100, float Q = 0.707);

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

/// \brief ComplexNorm TensorTransform.
/// \notes Compute the norm of complex tensor input.
class DATASET_API ComplexNorm final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] power Power of the norm, which must be non-negative (Default: 1.0).
  explicit ComplexNorm(float power = 1.0);

  /// \brief Destructor.
  ~ComplexNorm() override = default;

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
class DATASET_API ComputeDeltas final : public TensorTransform {
 public:
  /// \brief Construct a new Compute Deltas object.
  /// \f[
  /// d_{t}=\frac{{\textstyle\sum_{n=1}^{N}}n(c_{t+n}-c_{t-n})}{2{\textstyle\sum_{n=1}^{N}}n^{2}}
  /// \f]
  /// \param[in] win_length The window length used for computing delta, must be no less than 3 (Default: 5).
  /// \param[in] pad_mode Padding mode. Can be one of BorderType::kConstant, BorderType::kEdge,
  ///     BorderType::kReflect or BorderType::kSymmetric (Default: BorderType::kEdge).
  explicit ComputeDeltas(int32_t win_length = 5, BorderType pad_mode = BorderType::kEdge);

  /// \brief Destructor.
  ~ComputeDeltas() override = default;

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
  /// \param[in] enhancement_amount Controls the amount of the enhancement (Default: 75.0).
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

/// \brief Turn a waveform from the decibel scale to the power/amplitude scale.
class DATASET_API DBToAmplitude final : public TensorTransform {
 public:
  /// \brief Constructor
  /// \param[in] ref Reference which the output will be scaled by.
  /// \param[in] power If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.
  explicit DBToAmplitude(float ref, float power);

  /// \brief Destructor.
  ~DBToAmplitude() override = default;

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

/// \brief Detect pitch frequency.
class DATASET_API DetectPitchFrequency final : public TensorTransform {
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
  ~DetectPitchFrequency() override = default;

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
  ///     DensityFunction::kGPDF (Gaussian Probability Density Function) (Default: DensityFunction::kTPDF).
  /// \param[in] noise_shaping A filtering process that shapes the spectral energy of
  ///     quantisation error (Default: false).
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
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
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
  ///     and no more than the length of waveform (Default: 0).
  /// \param[in] fade_out_len Length of fade-out (time frames), which must be non-negative
  ///     and no more than the length of waveform (Default: 0).
  /// \param[in] fade_shape An enum for the fade shape (Default: FadeShape::kLinear).
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
  ~Flanger() override = default;

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
class DATASET_API FrequencyMasking final : public TensorTransform {
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
  ~FrequencyMasking() override = default;

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
  /// \param[in] gain_db Gain adjustment in decibels (dB) (Default: 1.0).
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

/// \brief Waveform calculation from linear scalar amplitude spectrogram using GriffinLim transform.
class DATASET_API GriffinLim final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \notes Calculated by formula:
  ///     x(n)=\frac{\sum_{m=-\infty}^{\infty} w(m S-n) y_{w}(m S, n)}{\sum_{m=-\infty}^{\infty} w^{2}(m S-n)}
  ///     where w represents the window function, y represents the reconstructed signal of each frame and x represents
  ///     the whole signal.
  /// \param[in] n_fft Size of FFT (Default: 400).
  /// \param[in] n_iter Number of iteration for phase recovery (Default: 32).
  /// \param[in] win_length Window size for GriffinLim (Default: 0, will be set to n_fft).
  /// \param[in] hop_length Length of hop between STFT windows (Default: 0, will be set to win_length / 2).
  /// \param[in] window_type Window type for GriffinLim (Default: WindowType::kHann).
  /// \param[in] power Exponent for the magnitude spectrogram (Default: 2.0).
  /// \param[in] momentum The momentum for fast Griffin-Lim (Default: 0.99).
  /// \param[in] length Length of the expected output waveform (Default: 0.0, will be set to the value of last
  ///     dimension of the stft matrix).
  /// \param[in] rand_init Flag for random phase initialization or all-zero phase initialization (Default: true).
  explicit GriffinLim(int32_t n_fft = 400, int32_t n_iter = 32, int32_t win_length = 0, int32_t hop_length = 0,
                      WindowType window_type = WindowType::kHann, float power = 2.0, float momentum = 0.99,
                      int32_t length = 0, bool rand_init = true);

  /// \brief Destructor.
  ~GriffinLim() = default;

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
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
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

/// \brief InverseMelScale TensorTransform
/// \notes Solve for a normal STFT from a mel frequency STFT, using a conversion matrix.
class DATASET_API InverseMelScale final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] n_stft Number of bins in STFT, must be positive.
  /// \param[in] n_mels Number of mel filter, must be positive (Default: 128).
  /// \param[in] sample_rate Sample rate of the signal, the value can't be zero (Default: 16000).
  /// \param[in] f_min Minimum frequency, must be non-negative (Default: 0.0).
  /// \param[in] f_max Maximum frequency, must be non-negative (Default: 0.0, will be set to sample_rate / 2).
  /// \param[in] max_iter Maximum number of optimization iterations, must be positive (Default: 100000).
  /// \param[in] tolerance_loss Value of loss to stop optimization at, must be non-negative (Default: 1e-5).
  /// \param[in] tolerance_change Difference in losses to stop optimization at, must be non-negative (Default: 1e-8).
  /// \param[in] sgdargs Parameters of SGD optimizer, including lr, momentum
  ///     (Default: {{"sgd_lr", 0.1}, {"sgd_momentum", 0.0}}).
  /// \param[in] norm Type of norm, value should be NormType::kSlaney or NormType::kNone. If norm is NormType::kSlaney,
  ///     divide the triangle mel weight by the width of the mel band (Default: NormType::kNone).
  /// \param[in] mel_type Type of mel, value should be MelType::kHtk or MelType::kSlaney (Default: MelType::kHtk).
  explicit InverseMelScale(int32_t n_stft, int32_t n_mels = 128, int32_t sample_rate = 16000, float f_min = 0.0,
                           float f_max = 0.0, int32_t max_iter = 100000, float tolerance_loss = 1e-5,
                           float tolerance_change = 1e-8,
                           const std::map<std::string, float> &sgdargs = {{"sgd_lr", 0.1}, {"sgd_momentum", 0.0}},
                           NormType norm = NormType::kNone, MelType mel_type = MelType::kHtk);

  /// \brief Destructor.
  ~InverseMelScale() = default;

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
  /// \param[in] clamp If True, clamp the output signal to be in the range [-1, 1] (Default: True).
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
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
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

/// \brief Separate a complex-valued spectrogram with shape (..., 2) into its magnitude and phase.
class DATASET_API Magphase final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] power Power of the norm, which must be non-negative (Default: 1.0).
  explicit Magphase(float power);

  /// \brief Destructor.
  ~Magphase() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief MaskAlongAxis TensorTransform.
/// \note Tensor operation to mask the input tensor along axis.
class MaskAlongAxis final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mask_start Starting position of the mask, which must be non negative.
  /// \param[in] mask_width The width of the mask, which must be positive.
  /// \param[in] mask_value Value to assign to the masked columns.
  /// \param[in] axis Axis to apply masking on (1 for frequency and 2 for time).
  MaskAlongAxis(int32_t mask_start, int32_t mask_width, float mask_value, int32_t axis);

  /// \brief Destructor.
  ~MaskAlongAxis() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief MaskAlongAxisIID TensorTransform.
/// \note Apply a mask along axis.
class MaskAlongAxisIID final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mask_param Number of columns to be masked, will be uniformly sampled from [0, mask_param],
  ///     must be non negative.
  /// \param[in] mask_value Value to assign to the masked columns.
  /// \param[in] axis Axis to apply masking on (1 for frequency and 2 for time).
  MaskAlongAxisIID(int32_t mask_param, float mask_value, int32_t axis);

  /// \brief Destructor.
  ~MaskAlongAxisIID() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief MelScale TensorTransform.
/// \notes Convert normal STFT to STFT at the Mel scale.
class DATASET_API MelScale final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] n_mels Number of mel filter, which must be positive (Default: 128).
  /// \param[in] sample_rate Sample rate of the signal, the value can't be zero (Default: 16000).
  /// \param[in] f_min Minimum frequency, which must be non negative (Default: 0).
  /// \param[in] f_max Maximum frequency, which must be positive (Default: 0, will be set to sample_rate / 2).
  /// \param[in] n_stft Number of bins in STFT, which must be positive (Default: 201).
  /// \param[in] norm Type of norm, value should be NormType::kSlaney or NormType::kNone. If norm is NormType::kSlaney,
  ///     divide the triangle mel weight by the width of the mel band (Default: NormType::kNone).
  /// \param[in] mel_type Type of mel, value should be MelType::kHtk or MelType::kSlaney (Default: MelType::kHtk).
  explicit MelScale(int32_t n_mels = 128, int32_t sample_rate = 16000, float f_min = 0, float f_max = 0.0,
                    int32_t n_stft = 201, NormType norm = NormType::kNone, MelType mel_type = MelType::kHtk);

  /// \brief Destructor.
  ~MelScale() = default;

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
/// \param[in] norm Norm to use, can be NormType::kNone or NormType::kSlaney (Default: NormType::kNone).
/// \param[in] mel_type Scale to use, can be MelType::kHtk or MelType::kSlaney (Default: MelType::kHtz).
/// \return Status code.
Status DATASET_API MelscaleFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels,
                                  int32_t sample_rate, NormType norm = NormType::kNone,
                                  MelType mel_type = MelType::kHtk);

/// \brief MuLawDecoding TensorTransform.
/// \note Decode mu-law encoded signal.
class DATASET_API MuLawDecoding final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] quantization_channels Number of channels, which must be positive (Default: 256).
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
  /// \param[in] quantization_channels Number of channels, which must be positive (Default: 256).
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
  /// \param[in] gain Coefficient of overload in dB, in range of [0, 100] (Default: 20.0).
  /// \param[in] color Coefficient of translation, in range of [0, 100] (Default: 20.0).
  explicit Overdrive(float gain = 20.0f, float color = 20.0f);

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
  ~Phaser() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief PhaseVocoder TensorTransform
/// \notes Given a STFT tensor, speed up in time without modifying pitch by factor of rate.
class DATASET_API PhaseVocoder final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rate Speed-up factor.
  /// \param[in] phase_advance Expected phase advance in each bin in shape of (freq, 1).
  PhaseVocoder(float rate, const MSTensor &phase_advance);

  /// \brief Destructor.
  ~PhaseVocoder() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resample TensorTransform.
/// \notes Resample a signal from one frequency to another. A sampling method can be given.
class DATASET_API Resample : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] orig_freq The original frequency of the signal, which must be positive (default=16000).
  /// \param[in] new_freq The desired frequency, which must be positive (default=16000).
  /// \param[in] resample_method The resampling method, which can be ResampleMethod::kSincInterpolation
  ///     and ResampleMethod::kKaiserWindow (default=ResampleMethod::kSincInterpolation).
  /// \param[in] lowpass_filter_width Controls the sharpness of the filter, more means sharper but less efficient,
  ///     which must be positive (default=6).
  /// \param[in] rolloff The roll-off frequency of the filter, as a fraction of the Nyquist. Lower values
  ///     reduce anti-aliasing, but also reduce some of the highest frequencies, range: (0, 1] (default=0.99).
  /// \param[in] beta The shape parameter used for kaiser window (default=14.769656459379492).
  explicit Resample(float orig_freq = 16000, float new_freq = 16000,
                    ResampleMethod resample_method = ResampleMethod::kSincInterpolation,
                    int32_t lowpass_filter_width = 6, float rolloff = 0.99, float beta = 14.769656459379492);

  /// \brief Destructor.
  ~Resample() = default;

  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

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

/// \brief Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.
class DATASET_API SlidingWindowCmn final : public TensorTransform {
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
  ~SlidingWindowCmn() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Create a spectral centroid from an audio signal.
class DATASET_API SpectralCentroid : public TensorTransform {
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

  ~SpectralCentroid() override = default;

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
class DATASET_API Spectrogram : public TensorTransform {
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
  /// \param[in] pad_mode Controls the padding method used when center is true,
  ///     which can be BorderType::kReflect, BorderType::kConstant, BorderType::kEdge,
  ///     BorderType::kSymmetric (Default: BorderType::kReflect).
  /// \param[in] onesided Controls whether to return half of results to avoid redundancy (Default: true).
  explicit Spectrogram(int32_t n_fft = 400, int32_t win_length = 0, int32_t hop_length = 0, int32_t pad = 0,
                       WindowType window = WindowType::kHann, float power = 2.0, bool normalized = false,
                       bool center = true, BorderType pad_mode = BorderType::kReflect, bool onesided = true);

  /// \brief Destructor.
  ~Spectrogram() override = default;

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
class DATASET_API TimeMasking final : public TensorTransform {
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
  ~TimeMasking() override = default;

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
class DATASET_API TimeStretch final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] hop_length Length of hop between STFT windows (Default: None, will use ((n_freq - 1) * 2) // 2).
  /// \param[in] n_freq Number of filter banks form STFT (Default: 201).
  /// \param[in] fixed_rate Rate to speed up or slow down the input in time
  ///     (Default: std::numeric_limits<float>::quiet_NaN(), will keep the original rate).
  explicit TimeStretch(float hop_length = std::numeric_limits<float>::quiet_NaN(), int n_freq = 201,
                       float fixed_rate = std::numeric_limits<float>::quiet_NaN());

  /// \brief Destructor.
  ~TimeStretch() override = default;

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
  /// \param[in] central_freq Central frequency (in Hz) (Default: 3000).
  /// \param[in] Q Quality factor, https://en.wikipedia.org/wiki/Q_factor, range: (0, 1] (Default: 0.707).
  TrebleBiquad(int32_t sample_rate, float gain, float central_freq = 3000, float Q = 0.707);

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
  /// \param[in] trigger_level The measurement level used to trigger activity detection (Default: 7.0).
  /// \param[in] trigger_time The time constant (in seconds) used to help ignore short sounds (Default: 0.25).
  /// \param[in] search_time The amount of audio (in seconds) to search for quieter/shorter sounds to include prior to
  ///     the detected trigger point (Default: 1.0).
  /// \param[in] allowed_gap The allowed gap (in seconds) between quiteter/shorter sounds to include prior to the
  ///     detected trigger point (Default: 0.25).
  /// \param[in] pre_trigger_time The amount of audio (in seconds) to preserve before the trigger point and any found
  ///     quieter/shorter bursts (Default: 0.0).
  /// \param[in] boot_time The time for the initial noise estimate (Default: 0.35).
  /// \param[in] noise_up_time Time constant used by the adaptive noise estimator, when the noise level is increasing
  ///     (Default: 0.1).
  /// \param[in] noise_down_time Time constant used by the adaptive noise estimator, when the noise level is decreasing
  ///     (Default: 0.01).
  /// \param[in] noise_reduction_amount The amount of noise reduction used in the detection algorithm (Default: 1.35).
  /// \param[in] measure_freq The frequency of the algorithm’s processing (Default: 20.0).
  /// \param[in] measure_duration The duration of measurement (Default: 0, use twice the measurement period).
  /// \param[in] measure_smooth_time The time constant used to smooth spectral measurements (Default: 0.4).
  /// \param[in] hp_filter_freq The "Brick-wall" frequency of high-pass filter applied at the input to the detector
  ///     algorithm (Default: 50.0).
  /// \param[in] lp_filter_freq The "Brick-wall" frequency of low-pass filter applied at the input to the detector
  ///     algorithm (Default: 6000.0).
  /// \param[in] hp_lifter_freq The "Brick-wall" frequency of high-pass lifter applied at the input to the detector
  ///     algorithm (Default: 150.0).
  /// \param[in] lp_lifter_freq The "Brick-wall" frequency of low-pass lifter applied at the input to the detector
  ///     algorithm (Default: 2000.0).
  explicit Vad(int32_t sample_rate, float trigger_level = 7.0, float trigger_time = 0.25, float search_time = 1.0,
               float allowed_gap = 0.25, float pre_trigger_time = 0.0, float boot_time = 0.35,
               float noise_up_time = 0.1, float noise_down_time = 0.01, float noise_reduction_amount = 1.35,
               float measure_freq = 20.0, float measure_duration = 0.0, float measure_smooth_time = 0.4,
               float hp_filter_freq = 50.0, float lp_filter_freq = 6000.0, float hp_lifter_freq = 150.0,
               float lp_lifter_freq = 2000.0);

  /// \brief Destructor.
  ~Vad() = default;

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
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_AUDIO_H_
