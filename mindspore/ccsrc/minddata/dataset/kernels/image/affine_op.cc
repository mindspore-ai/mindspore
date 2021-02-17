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
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "minddata/dataset/kernels/image/affine_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/kernels/image/math_utils.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {

const InterpolationMode AffineOp::kDefInterpolation = InterpolationMode::kNearestNeighbour;
const float_t AffineOp::kDegrees = 0.0;
const std::vector<float_t> AffineOp::kTranslation = {0.0, 0.0};
const float_t AffineOp::kScale = 1.0;
const std::vector<float_t> AffineOp::kShear = {0.0, 0.0};
const std::vector<uint8_t> AffineOp::kFillValue = {0, 0, 0};

AffineOp::AffineOp(float_t degrees, const std::vector<float_t> &translation, float_t scale,
                   const std::vector<float_t> &shear, InterpolationMode interpolation,
                   const std::vector<uint8_t> &fill_value)
    : degrees_(degrees),
      translation_(translation),
      scale_(scale),
      shear_(shear),
      interpolation_(interpolation),
      fill_value_(fill_value) {}

Status AffineOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  float_t translation_x = translation_[0];
  float_t translation_y = translation_[1];
  float_t degrees = 0.0;
  DegreesToRadians(degrees_, &degrees);
  float_t shear_x = shear_[0];
  float_t shear_y = shear_[1];
  DegreesToRadians(shear_x, &shear_x);
  DegreesToRadians(-1 * shear_y, &shear_y);

  // Apply Affine Transformation
  //       T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
  //       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
  //       RSS is rotation with scale and shear matrix
  //       RSS(a, s, (sx, sy)) =
  //       = R(a) * S(s) * SHy(sy) * SHx(sx)
  //       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
  //         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
  //         [ 0                    , 0                                      , 1 ]
  //
  // where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
  // SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
  //          [0, 1      ]              [-tan(s), 1]
  //
  // Thus, the affine matrix is M = T * C * RSS * C^-1

  // image is hwc, rows = shape()[0]
  float_t cx = ((input->shape()[1] - 1) / 2.0);
  float_t cy = ((input->shape()[0] - 1) / 2.0);
  // Calculate RSS
  std::vector<float_t> matrix{
    static_cast<float>(scale_ * cos(degrees + shear_y) / cos(shear_y)),
    static_cast<float>(scale_ * (-1 * cos(degrees + shear_y) * tan(shear_x) / cos(shear_y) - sin(degrees))),
    0,
    static_cast<float>(scale_ * sin(degrees + shear_y) / cos(shear_y)),
    static_cast<float>(scale_ * (-1 * sin(degrees + shear_y) * tan(shear_x) / cos(shear_y) + cos(degrees))),
    0};
  // Compute T * C * RSS * C^-1
  matrix[2] = (1 - matrix[0]) * cx - matrix[1] * cy + translation_x;
  matrix[5] = (1 - matrix[4]) * cy - matrix[3] * cx + translation_y;
  RETURN_IF_NOT_OK(Affine(input, output, matrix, interpolation_, fill_value_[0], fill_value_[1], fill_value_[2]));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
