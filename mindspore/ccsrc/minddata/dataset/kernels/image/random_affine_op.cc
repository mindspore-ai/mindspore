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
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "minddata/dataset/kernels/image/random_affine_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/kernels/image/math_utils.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {

const std::vector<float_t> RandomAffineOp::kDegreesRange = {0.0, 0.0};
const std::vector<float_t> RandomAffineOp::kTranslationPercentages = {0.0, 0.0, 0.0, 0.0};
const std::vector<float_t> RandomAffineOp::kScaleRange = {1.0, 1.0};
const std::vector<float_t> RandomAffineOp::kShearRanges = {0.0, 0.0, 0.0, 0.0};
const InterpolationMode RandomAffineOp::kDefInterpolation = InterpolationMode::kNearestNeighbour;
const std::vector<uint8_t> RandomAffineOp::kFillValue = {0, 0, 0};

RandomAffineOp::RandomAffineOp(std::vector<float_t> degrees, std::vector<float_t> translate_range,
                               std::vector<float_t> scale_range, std::vector<float_t> shear_ranges,
                               InterpolationMode interpolation, std::vector<uint8_t> fill_value)
    : AffineOp(0.0),
      degrees_range_(degrees),
      translate_range_(translate_range),
      scale_range_(scale_range),
      shear_ranges_(shear_ranges) {
  interpolation_ = interpolation;
  fill_value_ = fill_value;
  rnd_.seed(GetSeed());
  is_deterministic_ = false;
}

Status RandomAffineOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  dsize_t height = input->shape()[0];
  dsize_t width = input->shape()[1];
  float_t min_dx = translate_range_[0] * width;
  float_t max_dx = translate_range_[1] * width;
  float_t min_dy = translate_range_[2] * height;
  float_t max_dy = translate_range_[3] * height;
  float_t degrees = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(degrees_range_[0], degrees_range_[1], &rnd_, &degrees));
  float_t translation_x = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(min_dx, max_dx, &rnd_, &translation_x));
  float_t translation_y = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(min_dy, max_dy, &rnd_, &translation_y));
  float_t scale = 1.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(scale_range_[0], scale_range_[1], &rnd_, &scale));
  float_t shear_x = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(shear_ranges_[0], shear_ranges_[1], &rnd_, &shear_x));
  float_t shear_y = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(shear_ranges_[2], shear_ranges_[3], &rnd_, &shear_y));
  // assign to base class variables
  degrees_ = fmod(degrees, 360.0);
  scale_ = scale;
  translation_[0] = translation_x;
  translation_[1] = translation_y;
  shear_[0] = shear_x;
  shear_[1] = shear_y;
  return AffineOp::Compute(input, output);
}
}  // namespace dataset
}  // namespace mindspore
