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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_LFCC_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_LFCC_IR_H_

#include <map>
#include <memory>
#include <string>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace audio {
constexpr char kLFCCOperation[] = "LFCC";

class LFCCOperation : public TensorOperation {
 public:
  /// \brief Constructor.
  /// \param[in] sample_rate Sample rate of audio signal.
  /// \param[in] n_filter Number of linear filters to apply.
  /// \param[in] n_lfcc Number of lfc coefficients to retain.
  /// \param[in] f_min Minimum frequency.
  /// \param[in] f_max Maximum frequency.
  /// \param[in] dct_type Type of DCT (discrete cosine transform) to use.
  /// \param[in] norm Norm to use.
  /// \param[in] log_lf Whether to use log-lf spectrograms instead of db-scaled.
  /// \param[in] n_fft Size of FFT, creates n_fft // 2 + 1 bins.
  /// \param[in] win_length Window size.
  /// \param[in] hop_length Length of hop between STFT windows.
  /// \param[in] pad Two sided padding of signal.
  /// \param[in] window A function to create a window tensor that is applied/multiplied to each frame/window.
  /// \param[in] power Exponent for the magnitude spectrogram.
  /// \param[in] normalized Whether to normalize by magnitude after stft.
  /// \param[in] center Whether to pad waveform on both sides so that the tt-th frame is centered at
  ///     time t t*hop_length.
  /// \param[in] pad_mode Controls the padding method used when center is True.
  /// \param[in] onesided Controls whether to return half of results to avoid redundancy.
  LFCCOperation(int32_t sample_rate, int32_t n_filter, int32_t n_lfcc, float f_min, float f_max, int32_t dct_type,
                NormMode norm, bool log_lf, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad,
                WindowType window, float power, bool normalized, bool center, BorderType pad_mode, bool onesided);

  ~LFCCOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  int32_t n_filter_;
  int32_t n_lfcc_;
  int32_t dct_type_;
  NormMode norm_;
  bool log_lf_;
  std::map<std::string, std::string> speckwargs_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  float f_min_;
  float f_max_;
  int32_t pad_;
  WindowType window_;
  float power_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_LFCC_IR_H_
