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
#ifndef DATASET_KERNELS_IMAGE_DISTORT_BOUNDING_BOX_CROP_OP_H_
#define DATASET_KERNELS_IMAGE_DISTORT_BOUNDING_BOX_CROP_OP_H_

#include <memory>
#include <random>
#include <vector>
#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class DistortBoundingBoxCropOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefMaxAttempts;
  static const int32_t kDefBoxGenAttempts;

  // Constructor for DistortBoundingBoxCropOp
  // @param max_attempts tries before the crop happens
  // @param box_gen_attempts crop box generation attempts
  // @param aspect_ratio aspect ratio of the generated crop box
  // @param intersect_ratio area overlap ratio, condition for crop only if area over lap between the generated
  //                       crop box has sufficient overlap with any 1 bounding box
  // @param crop_ratio_lb the crop ratio lower bound
  // @param crop_ratio_ub the crop ratio upper bound
  // @param seed
  DistortBoundingBoxCropOp(float aspect_ratio, float intersect_ratio, float crop_ratio_lb, float crop_ratio_ub,
                           int32_t max_attempts = kDefMaxAttempts, int32_t box_gen_attempts = kDefBoxGenAttempts);

  ~DistortBoundingBoxCropOp() override = default;

  void Print(std::ostream& out) const override {
    out << "DistortBoundingBoxCropOp: " << max_attempts_ << " " << intersect_ratio_;
  }

  Status Compute(const std::vector<std::shared_ptr<Tensor>>& input,
                 std::vector<std::shared_ptr<Tensor>>* output) override;

  uint32_t NumInput() override { return 5; }
  Status OutputShape(const std::vector<TensorShape>& inputs, std::vector<TensorShape>& outputs) override;
  Status OutputType(const std::vector<DataType>& inputs, std::vector<DataType>& outputs) override;

 private:
  int32_t max_attempts_;
  int32_t box_gen_attempts_;
  float aspect_ratio_;
  float intersect_ratio_;
  float crop_ratio_lb_;
  float crop_ratio_ub_;
  std::mt19937 rnd_;
  uint32_t seed_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_DISTORT_BOUNDING_BOX_CROP_OP_H_
