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

#ifndef ENABLE_ANDROID

// RandomColorAdjustOperation.
RandomColorAdjustOperation::RandomColorAdjustOperation(std::vector<float> brightness, std::vector<float> contrast,
                                                       std::vector<float> saturation, std::vector<float> hue)
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

  brightness_lb = brightness_[0];
  brightness_ub = brightness_[0];

  if (brightness_.size() == 2) brightness_ub = brightness_[1];

  contrast_lb = contrast_[0];
  contrast_ub = contrast_[0];

  if (contrast_.size() == 2) contrast_ub = contrast_[1];

  saturation_lb = saturation_[0];
  saturation_ub = saturation_[0];

  if (saturation_.size() == 2) saturation_ub = saturation_[1];

  hue_lb = hue_[0];
  hue_ub = hue_[0];

  if (hue_.size() == 2) hue_ub = hue_[1];

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
#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
