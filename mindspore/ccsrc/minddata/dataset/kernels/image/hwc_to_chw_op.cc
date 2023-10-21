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
#include "minddata/dataset/kernels/image/hwc_to_chw_op.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status HwcToChwOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return HwcToChw(input, output);
}

Status HwcToChwOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "HWC2CHW: inputs cannot be empty.");
  TensorShape image_shape = inputs[0];
  constexpr auto kDefaultImageRank = 3;
  if (image_shape.Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(TensorShape{image_shape[2], image_shape[0], image_shape[1]});
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    !outputs.empty(),
    "HWC2CHW: invalid input shape, expected 3D input, but got input dimension is:" + std::to_string(inputs[0].Rank()));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
