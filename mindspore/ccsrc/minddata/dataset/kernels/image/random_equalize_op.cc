/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/random_equalize_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const float RandomEqualizeOp::kDefProbability = 0.5;

Status RandomEqualizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Check input
  RETURN_IF_NOT_OK(ValidateImageRank("RandomEqualize", input->Rank()));
  if (input->Rank() == kDefaultImageRank) {
    int num_channels = static_cast<int>(input->shape()[kChannelIndexHWC]);
    if (num_channels != kMinImageChannel && num_channels != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("RandomEqualize: input image is not in channel of 1 or 3, but got: " +
                               std::to_string(input->shape()[kChannelIndexHWC]));
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type() == DataType(DataType::DE_UINT8),
    "RandomEqualize: input image is not in type of uint8, but got: " + input->type().ToString());
  if (distribution_(rnd_)) {
    return Equalize(input, output);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
