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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_PRESERVE_AR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_PRESERVE_AR_OP_H_

#include <memory>
#include <vector>
#include <string>

#include "minddata/dataset/core/tensor.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class ResizePreserveAROp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefImgorientation;

  ResizePreserveAROp(int32_t height, int32_t width, int32_t img_orientation = kDefImgorientation);

  ~ResizePreserveAROp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kResizePreserveAROp; }

 protected:
  int32_t height_;
  int32_t width_;
  int32_t img_orientation_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_PRESERVE_AR_OP_H_
