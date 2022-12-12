/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_MEL_SPECTROGRAM_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_MEL_SPECTROGRAM_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "include/dataset/constants.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class MelSpectrogramOp : public TensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sample rate of audio signal.
  /// \param[in] n_fft Size of FFT, creates n_fft // 2 + 1 bins.
  /// \param[in] win_length Window size.
  /// \param[in] hop_length Length of hop between STFT windows.
  /// \param[in] f_min Minimum frequency.
  /// \param[in] f_max Maximum frequency.
  /// \param[in] pad Two sided padding of signal.
  /// \param[in] n_mels Number of mel filterbanks.
  /// \param[in] window A function to create a window tensor that is applied/multiplied to each rame/window.
  /// \param[in] power Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for energy, 2 for power, etc.
  /// \param[in] normalized Whether to normalize by magnitude after stft.
  /// \param[in] center Whether to pad waveform on both sides.
  /// \param[in] pad_mode Controls the padding method used when center is True.
  /// \param[in] onesided Controls whether to return half of results to avoid redundancy.
  /// \param[in] norm If 'slaney', divide the triangular mel weights by the width of the mel band (area normalization).
  /// \param[in] mel_scale Scale to use: htk or slaney.
  MelSpectrogramOp(int32_t sample_rate, int32_t n_fft, int32_t win_length, int32_t hop_length, float f_min, float f_max,
                   int32_t pad, int32_t n_mels, WindowType window, float power, bool normalized, bool center,
                   BorderType pad_mode, bool onesided, NormType norm, MelType mel_scale)
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

  ~MelSpectrogramOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kMelSpectrogramOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
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
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_MEL_SPECTROGRAM_OP_H_
