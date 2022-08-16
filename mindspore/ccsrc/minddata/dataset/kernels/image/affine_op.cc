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
#include <random>
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
  dsize_t height = input->shape()[0];
  dsize_t width = input->shape()[1];
  float_t translation_x = translation_[0] * width;
  float_t translation_y = translation_[1] * height;
  std::vector<float_t> new_translation{translation_x, translation_y};
  return Affine(input, output, degrees_, new_translation, scale_, shear_, interpolation_, fill_value_);
}
}  // namespace dataset
}  // namespace mindspore
