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

#include "minddata/dataset/audio/ir/kernels/inverse_spectrogram_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/audio/kernels/inverse_spectrogram_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace audio {
// InverseSpectrogram
InverseSpectrogramOperation::InverseSpectrogramOperation(int32_t length, int32_t n_fft, int32_t win_length,
                                                         int32_t hop_length, int32_t pad, WindowType window,
                                                         bool normalized, bool center, BorderType pad_mode,
                                                         bool onesided)
    : length_(length),
      n_fft_(n_fft),
      win_length_(win_length),
      hop_length_(hop_length),
      pad_(pad),
      window_(window),
      normalized_(normalized),
      center_(center),
      pad_mode_(pad_mode),
      onesided_(onesided) {}

InverseSpectrogramOperation::~InverseSpectrogramOperation() = default;

Status InverseSpectrogramOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("InverseSpectrogram", "length", length_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("InverseSpectrogram", "n_fft", n_fft_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("InverseSpectrogram", "win_length", win_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("InverseSpectrogram", "hop_length", hop_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("InverseSpectrogram", "pad", pad_));
  CHECK_FAIL_RETURN_UNEXPECTED(
    win_length_ <= n_fft_, "InverseSpectrogram: win_length must be less than or equal to n_fft, but got win_length: " +
                             std::to_string(win_length_) + ", n_fft: " + std::to_string(n_fft_));
  return Status::OK();
}

std::shared_ptr<TensorOp> InverseSpectrogramOperation::Build() {
  win_length_ = win_length_ == 0 ? n_fft_ : win_length_;
  hop_length_ = win_length_ == 0 ? win_length_ / TWO : hop_length_;
  std::shared_ptr<InverseSpectrogramOp> tensor_op = std::make_shared<InverseSpectrogramOp>(
    length_, n_fft_, win_length_, hop_length_, pad_, window_, normalized_, center_, pad_mode_, onesided_);

  return tensor_op;
}

std::string InverseSpectrogramOperation::Name() const { return kInverseSpectrogramOperation; }

Status InverseSpectrogramOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["length"] = length_;
  args["n_fft"] = n_fft_;
  args["win_length"] = win_length_;
  args["hop_length"] = hop_length_;
  args["pad"] = pad_;
  args["window"] = window_;
  args["normalized"] = normalized_;
  args["center"] = center_;
  args["pad_mode"] = pad_mode_;
  args["onesided"] = onesided_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
