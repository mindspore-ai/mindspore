/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/ir/kernels/mel_spectrogram_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/audio/kernels/mel_spectrogram_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace audio {
// MelSpectrogram
MelSpectrogramOperation::MelSpectrogramOperation(int32_t sample_rate, int32_t n_fft, int32_t win_length,
                                                 int32_t hop_length, float f_min, float f_max, int32_t pad,
                                                 int32_t n_mels, WindowType window, float power, bool normalized,
                                                 bool center, BorderType pad_mode, bool onesided, NormType norm,
                                                 MelType mel_scale)
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

MelSpectrogramOperation::~MelSpectrogramOperation() = default;

Status MelSpectrogramOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("MelSpectrogram", "n_fft", n_fft_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("MelSpectrogram", "win_length", win_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("MelSpectrogram", "hop_length", hop_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MelSpectrogram", "pad", pad_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("MelSpectrogram", "power", power_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MelSpectrogram", "n_mels", n_mels_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MelSpectrogram", "sample_rate", sample_rate_));
  CHECK_FAIL_RETURN_UNEXPECTED(f_min_ <= f_max_,
                               "MelSpectrogram: f_max must be greater than or equal to f_min, but got f_max: " +
                                 std::to_string(f_max_) + ", f_min: " + std::to_string(f_min_));
  if (f_max_ != 0) {
    RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("MelSpectrogram", "f_max", f_max_));
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(f_min_ < (sample_rate_ * HALF),
                                 "MelSpectrogram: f_min must be less than half of sample_rate when f_max is 0, but "
                                 "got f_min: " +
                                   std::to_string(f_min_));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(win_length_ <= n_fft_,
                               "MelSpectrogram: win_length must be less than or equal to n_fft, but got win_length: " +
                                 std::to_string(win_length_) + ", n_fft: " + std::to_string(n_fft_));
  return Status::OK();
}

std::shared_ptr<TensorOp> MelSpectrogramOperation::Build() {
  win_length_ = win_length_ == 0 ? n_fft_ : win_length_;
  hop_length_ = win_length_ == 0 ? win_length_ / TWO : hop_length_;
  f_max_ = f_max_ == 0 ? sample_rate_ / TWO : f_max_;
  std::shared_ptr<MelSpectrogramOp> tensor_op =
    std::make_shared<MelSpectrogramOp>(sample_rate_, n_fft_, win_length_, hop_length_, f_min_, f_max_, pad_, n_mels_,
                                       window_, power_, normalized_, center_, pad_mode_, onesided_, norm_, mel_scale_);
  return tensor_op;
}

std::string MelSpectrogramOperation::Name() const { return kMelSpectrogramOperation; }

Status MelSpectrogramOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["n_fft"] = n_fft_;
  args["win_length"] = win_length_;
  args["hop_length"] = hop_length_;
  args["f_min"] = f_min_;
  args["f_max"] = f_max_;
  args["pad"] = pad_;
  args["n_mels"] = n_mels_;
  args["window"] = window_;
  args["power"] = power_;
  args["normalized"] = normalized_;
  args["center"] = center_;
  args["pad_mode"] = pad_mode_;
  args["onesided"] = onesided_;
  args["norm"] = norm_;
  args["mel_scale"] = mel_scale_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
