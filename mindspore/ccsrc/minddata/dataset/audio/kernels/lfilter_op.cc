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
#include "minddata/dataset/audio/kernels/lfilter_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status LFilterOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("LFilter", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorFloat("LFilter", input));
  if (input->type() == DataType(DataType::DE_FLOAT32)) {
    return LFilter(input, output, a_coeffs_, b_coeffs_, clamp_);
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    std::vector<double> a_coeffs_double;
    std::vector<double> b_coeffs_double;
    for (auto i = 0; i < a_coeffs_.size(); i++) {
      a_coeffs_double.push_back(static_cast<double>(a_coeffs_[i]));
    }
    for (auto i = 0; i < b_coeffs_.size(); i++) {
      b_coeffs_double.push_back(static_cast<double>(b_coeffs_[i]));
    }
    return LFilter(input, output, a_coeffs_double, b_coeffs_double, clamp_);
  } else {
    std::vector<float16> a_coeffs_float16;
    std::vector<float16> b_coeffs_float16;
    for (auto i = 0; i < a_coeffs_.size(); i++) {
      a_coeffs_float16.push_back(static_cast<float16>(a_coeffs_[i]));
    }
    for (auto i = 0; i < b_coeffs_.size(); i++) {
      b_coeffs_float16.push_back(static_cast<float16>(b_coeffs_[i]));
    }
    return LFilter(input, output, a_coeffs_float16, b_coeffs_float16, clamp_);
  }
}
}  // namespace dataset
}  // namespace mindspore
