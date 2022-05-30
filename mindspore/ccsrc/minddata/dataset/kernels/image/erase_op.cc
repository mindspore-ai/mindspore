/**
 * Copyright 2022 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/kernels/image/erase_op.h"

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// constructor
EraseOp::EraseOp(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<uint8_t> &value,
                 bool inplace)
    : top_(top), left_(left), height_(height), width_(width), value_(value), inplace_(inplace) {}

Status EraseOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImageDtype("Erase", input->type()));
  if (input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("Erase: input tensor is not in shape of <H,W,C>, but got rank: " +
                             std::to_string(input->Rank()));
  }
  int num_channels = input->shape()[2];
  if (num_channels != kDefaultImageChannel) {
    RETURN_STATUS_UNEXPECTED("Erase: channel of input image should be 3, but got: " + std::to_string(num_channels));
  }
  return Erase(input, output, top_, left_, height_, width_, value_, inplace_);
}
}  // namespace dataset
}  // namespace mindspore
