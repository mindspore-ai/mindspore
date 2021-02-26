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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_AFFINE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_AFFINE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/affine_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomAffineOp : public AffineOp {
 public:
  /// Default values, also used by python_bindings.cc
  static const std::vector<float_t> kDegreesRange;
  static const std::vector<float_t> kTranslationPercentages;
  static const std::vector<float_t> kScaleRange;
  static const std::vector<float_t> kShearRanges;
  static const InterpolationMode kDefInterpolation;
  static const std::vector<uint8_t> kFillValue;

  explicit RandomAffineOp(std::vector<float_t> degrees, std::vector<float_t> translate_range = kTranslationPercentages,
                          std::vector<float_t> scale_range = kScaleRange,
                          std::vector<float_t> shear_ranges = kShearRanges,
                          InterpolationMode interpolation = kDefInterpolation,
                          std::vector<uint8_t> fill_value = kFillValue);

  ~RandomAffineOp() override = default;

  std::string Name() const override { return kRandomAffineOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  std::vector<float_t> degrees_range_;    // min_degree, max_degree
  std::vector<float_t> translate_range_;  // maximum x translation percentage, maximum y translation percentage
  std::vector<float_t> scale_range_;      // min_scale, max_scale
  std::vector<float_t> shear_ranges_;     // min_x_shear, max_x_shear, min_y_shear, max_y_shear
  std::mt19937 rnd_;                      // random device
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_AFFINE_OP_H_
