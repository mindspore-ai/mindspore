/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/invert_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// only supports RGB images
Status InvertOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->Rank() == kDefaultImageRank,
    "Invert: input tensor is not in shape of <H,W,C>, but got rank: " + std::to_string(input->Rank()));
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape()[kChannelIndexHWC] == kDefaultImageChannel,
                               "Invert: the number of channels of input tensor is not 3, but got: " +
                                 std::to_string(input->shape()[kChannelIndexHWC]));
  return Invert(input, output);
}
}  // namespace dataset
}  // namespace mindspore
