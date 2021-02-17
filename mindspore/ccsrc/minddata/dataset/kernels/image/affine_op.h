/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_AFFINE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_AFFINE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class AffineOp : public TensorOp {
 public:
  /// Default values
  static const float_t kDegrees;
  static const std::vector<float_t> kTranslation;
  static const float_t kScale;
  static const std::vector<float_t> kShear;
  static const InterpolationMode kDefInterpolation;
  static const std::vector<uint8_t> kFillValue;

  /// Constructor
  explicit AffineOp(float_t degrees, const std::vector<float_t> &translation = kTranslation, float_t scale = kScale,
                    const std::vector<float_t> &shear = kShear, InterpolationMode interpolation = kDefInterpolation,
                    const std::vector<uint8_t> &fill_value = kFillValue);

  ~AffineOp() override = default;

  std::string Name() const override { return kAffineOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 protected:
  float_t degrees_;
  std::vector<float_t> translation_;  // translation_x and translation_y
  float_t scale_;
  std::vector<float_t> shear_;  // shear_x and shear_y
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_AFFINE_OP_H_
