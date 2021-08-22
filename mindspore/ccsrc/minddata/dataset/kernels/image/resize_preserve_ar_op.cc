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
#include "minddata/dataset/kernels/image/resize_preserve_ar_op.h"

#ifdef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t ResizePreserveAROp::kDefImgorientation = 0;

ResizePreserveAROp::ResizePreserveAROp(int32_t height, int32_t width, int32_t img_orientation)
    : height_(height), width_(width), img_orientation_(img_orientation) {}

Status ResizePreserveAROp::Compute(const TensorRow &inputs, TensorRow *outputs) {
  IO_CHECK_VECTOR(inputs, outputs);
#ifdef ENABLE_ANDROID
  return ResizePreserve(inputs, height_, width_, img_orientation_, outputs);
#endif
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
