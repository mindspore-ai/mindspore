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

#include "minddata/dataset/kernels/ir/vision/random_color_adjust_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_color_adjust_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
constexpr size_t dimension_zero = 0;
constexpr size_t dimension_one = 1;
constexpr size_t size_two = 2;

#ifndef ENABLE_ANDROID
// RandomColorAdjustOperation.
RandomColorAdjustOperation::RandomColorAdjustOperation(const std::vector<float> &brightness,
                                                       const std::vector<float> &contrast,
                                                       const std::vector<float> &saturation,
                                                       const std::vector<float> &hue)
    : brightness_(brightness), contrast_(contrast), saturation_(saturation), hue_(hue) {
  random_op_ = true;
}

RandomColorAdjustOperation::~RandomColorAdjustOperation() = default;

std::string RandomColorAdjustOperation::Name() const { return kRandomColorAdjustOperation; }

Status RandomColorAdjustOperation::ValidateParams() {
  // brightness
  RETURN_IF_NOT_OK(ValidateVectorColorAttribute("RandomColorAdjust", "brightness", brightness_, {0}));
  // contrast
  RETURN_IF_NOT_OK(ValidateVectorColorAttribute("RandomColorAdjust", "contrast", contrast_, {0}));
  // saturation
  RETURN_IF_NOT_OK(ValidateVectorColorAttribute("RandomColorAdjust", "saturation", saturation_, {0}));
  // hue
  RETURN_IF_NOT_OK(ValidateVectorColorAttribute("RandomColorAdjust", "hue", hue_, {-0.5, 0.5}));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomColorAdjustOperation::Build() {
  float brightness_lb, brightness_ub, contrast_lb, contrast_ub, saturation_lb, saturation_ub, hue_lb, hue_ub;

  brightness_lb = brightness_[dimension_zero];
  brightness_ub = brightness_[dimension_zero];

  if (brightness_.size() == size_two) brightness_ub = brightness_[dimension_one];

  contrast_lb = contrast_[dimension_zero];
  contrast_ub = contrast_[dimension_zero];

  if (contrast_.size() == size_two) contrast_ub = contrast_[dimension_one];

  saturation_lb = saturation_[dimension_zero];
  saturation_ub = saturation_[dimension_zero];

  if (saturation_.size() == size_two) saturation_ub = saturation_[dimension_one];

  hue_lb = hue_[dimension_zero];
  hue_ub = hue_[dimension_zero];

  if (hue_.size() == size_two) hue_ub = hue_[dimension_one];

  std::shared_ptr<RandomColorAdjustOp> tensor_op = std::make_shared<RandomColorAdjustOp>(
    brightness_lb, brightness_ub, contrast_lb, contrast_ub, saturation_lb, saturation_ub, hue_lb, hue_ub);
  return tensor_op;
}

Status RandomColorAdjustOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["brightness"] = brightness_;
  args["contrast"] = contrast_;
  args["saturation"] = saturation_;
  args["hue"] = hue_;
  *out_json = args;
  return Status::OK();
}

Status RandomColorAdjustOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("brightness") != op_params.end(), "Failed to find brightness");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("contrast") != op_params.end(), "Failed to find contrast");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("saturation") != op_params.end(), "Failed to find saturation");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("hue") != op_params.end(), "Failed to find hue");
  std::vector<float> brightness = op_params["brightness"];
  std::vector<float> contrast = op_params["contrast"];
  std::vector<float> saturation = op_params["saturation"];
  std::vector<float> hue = op_params["hue"];
  *operation = std::make_shared<vision::RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
