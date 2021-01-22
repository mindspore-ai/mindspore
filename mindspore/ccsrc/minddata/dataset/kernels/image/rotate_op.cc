/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/rotate_op.h"
#ifdef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif

namespace mindspore {
namespace dataset {

RotateOp::RotateOp(int angle_id) : angle_id_(angle_id) {}

Status RotateOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->shape().Size() >= 2,
    "Rotate: image shape " + std::to_string(input->shape().Size()) + " is not <H,W,C> or <H,W>.");
#ifdef ENABLE_ANDROID
  Rotate(input, output, angle_id_);
#endif
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
