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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_WITH_BBOX_OP_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_WITH_BBOX_OP_H

#include <string>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/kernels/image/resize_op.h"

namespace mindspore {
namespace dataset {
class ResizeWithBBoxOp : public ResizeOp {
 public:
  //  Constructor for ResizeWithBBoxOp, with default value and passing to base class constructor
  explicit ResizeWithBBoxOp(int32_t size_1, int32_t size_2 = kDefWidth,
                            InterpolationMode mInterpolation = kDefInterpolation)
      : ResizeOp(size_1, size_2, mInterpolation) {}

  ~ResizeWithBBoxOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ": " << size1_ << " " << size2_; }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kResizeWithBBoxOp; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RESIZE_WITH_BBOX_OP_H
