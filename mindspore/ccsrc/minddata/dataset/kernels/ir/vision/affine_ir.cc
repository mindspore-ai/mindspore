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
#include "minddata/dataset/kernels/ir/vision/affine_ir.h"

#include "minddata/dataset/kernels/image/affine_op.h"

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
AffineOperation::AffineOperation(float_t degrees, const std::vector<float> &translation, float scale,
                                 const std::vector<float> &shear, InterpolationMode interpolation,
                                 const std::vector<uint8_t> &fill_value)
    : degrees_(degrees),
      translation_(translation),
      scale_(scale),
      shear_(shear),
      interpolation_(interpolation),
      fill_value_(fill_value) {}

AffineOperation::~AffineOperation() = default;

std::string AffineOperation::Name() const { return kAffineOperation; }

Status AffineOperation::ValidateParams() {
  // Translate
  constexpr size_t kExpectedTranslationSize = 2;
  if (translation_.size() != kExpectedTranslationSize) {
    std::string err_msg =
      "Affine: translate expecting size 2, got: translation.size() = " + std::to_string(translation_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateScalar("Affine", "translate", translation_[0], {-1, 1}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Affine", "translate", translation_[1], {-1, 1}, false, false));

  // Shear
  constexpr size_t kExpectedShearSize = 2;
  if (shear_.size() != kExpectedShearSize) {
    std::string err_msg = "Affine: shear_ranges expecting size 2, got: shear.size() = " + std::to_string(shear_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // Fill Value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("Affine", fill_value_));
  // interpolation
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea) {
    std::string err_msg = "Affine: Invalid InterpolationMode, check input value of enum.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> AffineOperation::Build() {
  std::shared_ptr<AffineOp> tensor_op =
    std::make_shared<AffineOp>(degrees_, translation_, scale_, shear_, interpolation_, fill_value_);
  return tensor_op;
}

Status AffineOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["degrees"] = degrees_;
  args["translate"] = translation_;
  args["scale"] = scale_;
  args["shear"] = shear_;
  args["resample"] = interpolation_;
  args["fill_value"] = fill_value_;
  *out_json = args;
  return Status::OK();
}

Status AffineOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("degrees") != op_params.end(), "Failed to find degrees");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("translate") != op_params.end(), "Failed to find translate");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("scale") != op_params.end(), "Failed to find scale");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("shear") != op_params.end(), "Failed to find shear");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("resample") != op_params.end(), "Failed to find resample");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("fill_value") != op_params.end(), "Failed to find fill_value");
  float_t degrees = op_params["degrees"];
  std::vector<float> translation = op_params["translate"];
  float scale = op_params["scale"];
  std::vector<float> shear = op_params["shear"];
  InterpolationMode interpolation = static_cast<InterpolationMode>(op_params["resample"]);
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  *operation = std::make_shared<vision::AffineOperation>(degrees, translation, scale, shear, interpolation, fill_value);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
