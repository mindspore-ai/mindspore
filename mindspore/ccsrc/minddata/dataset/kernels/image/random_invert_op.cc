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

#include "minddata/dataset/kernels/image/random_invert_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
const float RandomInvertOp::kDefProbability = 0.5;

Status RandomInvertOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input
  if (input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("RandomInvert: image shape is not <H,W,C>, got rank: " + std::to_string(input->Rank()));
  }
  if (input->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
    RETURN_STATUS_UNEXPECTED(
      "RandomInvert: image shape is incorrect, expected num of channels is 3, "
      "but got:" +
      std::to_string(input->shape()[kChannelIndexHWC]));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(input->type().AsCVType() != kCVInvalidType,
                               "RandomInvert: Cannot convert from OpenCV type, unknown CV type. Currently "
                               "supported data type: [int8, uint8, int16, uint16, int32, float16, float32, float64].");
  if (distribution_(rnd_)) {
    return InvertOp::Compute(input, output);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
