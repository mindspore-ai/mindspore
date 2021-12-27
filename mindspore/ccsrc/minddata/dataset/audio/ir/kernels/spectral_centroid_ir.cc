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
#include "minddata/dataset/audio/ir/kernels/spectral_centroid_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/spectral_centroid_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// SpectralCentroidOperation
SpectralCentroidOperation::SpectralCentroidOperation(int32_t sample_rate, int32_t n_fft, int32_t win_length,
                                                     int32_t hop_length, int32_t pad, WindowType window)
    : sample_rate_(sample_rate),
      n_fft_(n_fft),
      win_length_(win_length),
      hop_length_(hop_length),
      pad_(pad),
      window_(window) {}

Status SpectralCentroidOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("SpectralCentroid", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("SpectralCentroid", "n_fft", n_fft_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("SpectralCentroid", "win_length", win_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("SpectralCentroid", "hop_length", hop_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("SpectralCentroid", "pad", pad_));
  CHECK_FAIL_RETURN_UNEXPECTED(
    win_length_ <= n_fft_, "SpectralCentroid: win_length must be less than or equal to n_fft, but got win_length: " +
                             std::to_string(win_length_) + ", n_fft: " + std::to_string(n_fft_));
  return Status::OK();
}

std::shared_ptr<TensorOp> SpectralCentroidOperation::Build() {
  int32_t win_length = (win_length_ == 0) ? n_fft_ : win_length_;
  int32_t hop_length = (hop_length_ == 0) ? win_length / 2 : hop_length_;
  std::shared_ptr<SpectralCentroidOp> tensor_op =
    std::make_shared<SpectralCentroidOp>(sample_rate_, n_fft_, win_length, hop_length, pad_, window_);
  return tensor_op;
}

Status SpectralCentroidOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["n_fft"] = n_fft_;
  args["win_length"] = win_length_;
  args["hop_length"] = hop_length_;
  args["pad"] = pad_;
  args["window"] = window_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
