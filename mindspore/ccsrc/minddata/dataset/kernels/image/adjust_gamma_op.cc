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

#include "minddata/dataset/kernels/image/adjust_gamma_op.h"

#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
constexpr float AdjustGammaOp::kGain = 1.0;

Status AdjustGammaOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  // typecast
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() != DataType::DE_STRING,
                               "AdjustGamma: input tensor type should be int, float or double, but got: string.");

  if (input->type().IsFloat()) {
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return AdjustGamma(input_tensor, output, gamma_, gain_);
  } else {
    return AdjustGamma(input, output, gamma_, gain_);
  }
}
}  // namespace dataset
}  // namespace mindspore
