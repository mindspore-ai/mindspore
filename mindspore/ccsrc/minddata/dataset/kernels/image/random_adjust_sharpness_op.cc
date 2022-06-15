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

#include "minddata/dataset/kernels/image/random_adjust_sharpness_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
const float RandomAdjustSharpnessOp::kDefProbability = 0.5;

Status RandomAdjustSharpnessOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  // Check input
  if (input->Rank() != kMinImageRank && input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("RandomAdjustSharpness: image shape is not <H,W,C> or <H,W>, got rank: " +
                             std::to_string(input->Rank()));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(input->type().AsCVType() != kCVInvalidType,
                               "RandomAdjustSharpness: Cannot convert from OpenCV type, unknown CV type. Currently "
                               "supported data type: [int8, uint8, int16, uint16, int32, float16, float32, float64].");

  if (distribution_(rnd_)) {
    return SharpnessOp::Compute(input, output);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
