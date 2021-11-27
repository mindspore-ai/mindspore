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
#include "minddata/dataset/audio/kernels/fade_op.h"

#include <cmath>

#include "minddata/dataset/audio/kernels/audio_utils.h"

namespace mindspore {
namespace dataset {
constexpr int32_t FadeOp::kFadeInLen = 0;
constexpr int32_t FadeOp::kFadeOutLen = 0;
constexpr FadeShape FadeOp::kFadeShape = FadeShape::kLinear;

Status FadeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateLowRank("Fade", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Fade", input));
  if (fade_in_len_ == 0 && fade_out_len_ == 0) {
    *output = input;
  } else {
    RETURN_IF_NOT_OK(Fade(input, output, fade_in_len_, fade_out_len_, fade_shape_));
  }
  return Status::OK();
}

Status FadeOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  RETURN_IF_NOT_OK(ValidateTensorType("Fade", inputs[0].IsNumeric(), "[int, float, double]", inputs[0].ToString()));
  if (inputs[0] == DataType(DataType::DE_FLOAT64)) {
    outputs[0] = DataType(DataType::DE_FLOAT64);
  } else {
    outputs[0] = DataType(DataType::DE_FLOAT32);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
