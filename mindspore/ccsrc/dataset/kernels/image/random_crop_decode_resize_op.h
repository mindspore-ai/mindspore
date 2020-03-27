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
#ifndef DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_
#define DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>
#include "dataset/core/tensor.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/kernels/image/random_crop_and_resize_op.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomCropDecodeResizeOp : public RandomCropAndResizeOp {
 public:
  RandomCropDecodeResizeOp(int32_t target_height, int32_t target_width, float scale_lb = kDefScaleLb,
                           float scale_ub = kDefScaleUb, float aspect_lb = kDefAspectLb, float aspect_ub = kDefAspectUb,
                           InterpolationMode interpolation = kDefInterpolation, int32_t max_iter = kDefMaxIter);

  ~RandomCropDecodeResizeOp() override = default;

  void Print(std::ostream &out) const override {
    out << "RandomCropDecodeResize: " << RandomCropAndResizeOp::target_height_ << " "
        << RandomCropAndResizeOp::target_width_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_
