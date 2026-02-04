/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/vision/kernels/random_affine_op.h"

#include <cmath>
#include <limits>
#include <utility>

#include "minddata/dataset/vision/kernels/image_utils.h"
#include "minddata/dataset/vision/kernels/math_utils.h"
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
    : degrees_range_(std::move(degrees)),
      translate_range_(std::move(translate_range)),
      scale_range_(std::move(scale_range)),
      shear_ranges_(std::move(shear_ranges)),
      interpolation_(interpolation),
      fill_value_(std::move(fill_value)) {}

Status RandomAffineOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  CHECK_FAIL_RETURN_UNEXPECTED(translate_range_.size() == 4, "RandomAffine: the translate range size is not 4.");
  CHECK_FAIL_RETURN_UNEXPECTED(degrees_range_.size() == 2, "RandomAffine: the degrees range size is not 2.");
  CHECK_FAIL_RETURN_UNEXPECTED(scale_range_.size() == 2, "RandomAffine: the scale range size is not 2.");
  CHECK_FAIL_RETURN_UNEXPECTED(shear_ranges_.size() == 4, "RandomAffine: the shear ranges size is not 4.");

  RETURN_IF_NOT_OK(ValidateImageRank("Affine", input->shape().Rank()));

  dsize_t height = input->shape()[0];
  dsize_t width = input->shape()[1];
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / std::abs(translate_range_[0])) > width,
                               "RandomAffineOp: multiplication out of bounds.");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / std::abs(translate_range_[1])) > width,
                               "RandomAffineOp: multiplication out of bounds.");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / std::abs(translate_range_[2])) > height,
                               "RandomAffineOp: multiplication out of bounds.");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / std::abs(translate_range_[3])) > height,
                               "RandomAffineOp: multiplication out of bounds.");
  float_t min_dx = translate_range_[0] * static_cast<float_t>(width);
  float_t max_dx = translate_range_[1] * static_cast<float_t>(width);
  float_t min_dy = translate_range_[2] * static_cast<float_t>(height);
  float_t max_dy = translate_range_[3] * static_cast<float_t>(height);
  float_t degrees = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(degrees_range_[0], degrees_range_[1], &random_generator_, &degrees));
  float_t translation_x = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(min_dx, max_dx, &random_generator_, &translation_x));
  float_t translation_y = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(min_dy, max_dy, &random_generator_, &translation_y));
  float_t scale = 1.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(scale_range_[0], scale_range_[1], &random_generator_, &scale));
  float_t shear_x = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(shear_ranges_[0], shear_ranges_[1], &random_generator_, &shear_x));
  float_t shear_y = 0.0;
  RETURN_IF_NOT_OK(GenerateRealNumber(shear_ranges_[2], shear_ranges_[3], &random_generator_, &shear_y));
  // assign to base class variables
  degrees = fmod(degrees, 360.0F);
  std::vector<float_t> translation = {translation_x, translation_y};
  std::vector<float_t> shear = {shear_x, shear_y};
  return Affine(input, output, degrees, translation, scale, shear, interpolation_, fill_value_);
}
}  // namespace dataset
}  // namespace mindspore
