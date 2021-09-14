/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomCropDecodeResizeOp : public RandomCropAndResizeOp {
 public:
  RandomCropDecodeResizeOp(int32_t target_height, int32_t target_width, float scale_lb = kDefScaleLb,
                           float scale_ub = kDefScaleUb, float aspect_lb = kDefAspectLb, float aspect_ub = kDefAspectUb,
                           InterpolationMode interpolation = kDefInterpolation, int32_t max_attempts = kDefMaxIter);

  explicit RandomCropDecodeResizeOp(const RandomCropAndResizeOp &rhs) : RandomCropAndResizeOp(rhs) {}

  ~RandomCropDecodeResizeOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": " << RandomCropAndResizeOp::target_height_ << " " << RandomCropAndResizeOp::target_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomCropDecodeResizeOp; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_
