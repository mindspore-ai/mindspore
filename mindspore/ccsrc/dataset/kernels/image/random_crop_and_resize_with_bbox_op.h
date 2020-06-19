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
#ifndef DATASET_KERNELS_IMAGE_RANDOM_CROP_AND_RESIZE_WITH_BBOX_OP_H_
#define DATASET_KERNELS_IMAGE_RANDOM_CROP_AND_RESIZE_WITH_BBOX_OP_H_

#include "dataset/kernels/image/random_crop_and_resize_op.h"

namespace mindspore {
namespace dataset {

class RandomCropAndResizeWithBBoxOp : public RandomCropAndResizeOp {
 public:
  //  Constructor for RandomCropAndResizeWithBBoxOp, with default value and passing to base class constructor
  RandomCropAndResizeWithBBoxOp(int32_t target_height, int32_t target_width, float scale_lb = kDefScaleLb,
                                float scale_ub = kDefScaleUb, float aspect_lb = kDefAspectLb,
                                float aspect_ub = kDefAspectUb, InterpolationMode interpolation = kDefInterpolation,
                                int32_t max_iter = kDefMaxIter)
      : RandomCropAndResizeOp(target_height, target_width, scale_lb, scale_ub, aspect_lb, aspect_ub, interpolation,
                              max_iter) {}

  ~RandomCropAndResizeWithBBoxOp() override = default;

  void Print(std::ostream &out) const override {
    out << "RandomCropAndResizeWithBBox: " << RandomCropAndResizeOp::target_height_ << " "
        << RandomCropAndResizeOp::target_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_RANDOM_CROP_AND_RESIZE_WITH_BBOX_OP_H_
