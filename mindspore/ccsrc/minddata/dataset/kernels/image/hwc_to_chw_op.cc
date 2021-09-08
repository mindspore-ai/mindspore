/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status HwcToChwOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // input.shape == HWC
  // output.shape == CHW
  return HwcToChw(input, output);
}
Status HwcToChwOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  CHECK_FAIL_RETURN_UNEXPECTED(inputs.size() > 0, "HwcToChwOp::OutputShape inputs size should > 0");
  TensorShape in = inputs[0];
  TensorShape out = TensorShape{in[2], in[0], in[1]};
  if (inputs[0].Rank() == 3) {
    (void)outputs.emplace_back(out);
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(
    StatusCode::kMDUnexpectedError,
    "HWC2CHW: invalid input shape, expected 3D input, but got input dimension is:" + std::to_string(inputs[0].Rank()));
}
}  // namespace dataset
}  // namespace mindspore
