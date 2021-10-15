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

#include "minddata/dataset/kernels/ir/vision/random_rotation_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_rotation_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
constexpr size_t dimension_zero = 0;
constexpr size_t dimension_one = 1;
constexpr size_t dimension_two = 2;
constexpr size_t size_one = 1;
constexpr size_t size_two = 2;
constexpr size_t size_three = 3;

// Function to create RandomRotationOperation.
RandomRotationOperation::RandomRotationOperation(const std::vector<float> &degrees, InterpolationMode resample,
                                                 bool expand, const std::vector<float> &center,
                                                 const std::vector<uint8_t> &fill_value)
    : TensorOperation(true),
      degrees_(degrees),
      interpolation_mode_(resample),
      expand_(expand),
      center_(center),
      fill_value_(fill_value) {}

RandomRotationOperation::~RandomRotationOperation() = default;

std::string RandomRotationOperation::Name() const { return kRandomRotationOperation; }

Status RandomRotationOperation::ValidateParams() {
  // degrees
  if (degrees_.size() != size_two && degrees_.size() != size_one) {
    std::string err_msg =
      "RandomRotation: degrees must be a vector of one or two values, got: " + std::to_string(degrees_.size());
    MS_LOG(ERROR) << "RandomRotation: degrees must be a vector of one or two values, got: " << degrees_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if ((degrees_.size() == size_two) && (degrees_[dimension_one] < degrees_[dimension_zero])) {
    std::string err_msg = "RandomRotation: degrees must be in the format of (min, max), got: (" +
                          std::to_string(degrees_[dimension_zero]) + ", " + std::to_string(degrees_[dimension_one]) +
                          ")";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  } else if ((degrees_.size() == size_one) && (degrees_[dimension_zero] < 0)) {
    std::string err_msg =
      "RandomRotation: if degrees only has one value, it must be greater than or equal to 0, got: " +
      std::to_string(degrees_[dimension_zero]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // center
  if (center_.size() != 0 && center_.size() != size_two) {
    std::string err_msg =
      "RandomRotation: center must be a vector of two values or empty, got: " + std::to_string(center_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("RandomRotation", fill_value_));
  // interpolation
  if (interpolation_mode_ != InterpolationMode::kLinear &&
      interpolation_mode_ != InterpolationMode::kNearestNeighbour && interpolation_mode_ != InterpolationMode::kCubic &&
      interpolation_mode_ != InterpolationMode::kArea) {
    std::string err_msg = "RandomRotation: Invalid InterpolationMode, check input value of enum.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomRotationOperation::Build() {
  float start_degree, end_degree;
  if (degrees_.size() == size_one) {
    start_degree = -degrees_[dimension_zero];
    end_degree = degrees_[dimension_zero];
  } else if (degrees_.size() == size_two) {
    start_degree = degrees_[dimension_zero];
    end_degree = degrees_[dimension_one];
  }

  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[dimension_zero];
  fill_g = fill_value_[dimension_zero];
  fill_b = fill_value_[dimension_zero];

  if (fill_value_.size() == size_three) {
    fill_r = fill_value_[dimension_zero];
    fill_g = fill_value_[dimension_one];
    fill_b = fill_value_[dimension_two];
  }

  std::shared_ptr<RandomRotationOp> tensor_op = std::make_shared<RandomRotationOp>(
    start_degree, end_degree, interpolation_mode_, expand_, center_, fill_r, fill_g, fill_b);
  return tensor_op;
}

Status RandomRotationOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["degrees"] = degrees_;
  args["resample"] = interpolation_mode_;
  args["expand"] = expand_;
  args["center"] = center_;
  args["fill_value"] = fill_value_;
  *out_json = args;
  return Status::OK();
}

Status RandomRotationOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("degrees") != op_params.end(), "Failed to find degrees");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("resample") != op_params.end(), "Failed to find resample");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("expand") != op_params.end(), "Failed to find expand");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("center") != op_params.end(), "Failed to find center");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("fill_value") != op_params.end(), "Failed to find fill_value");
  std::vector<float> degrees = op_params["degrees"];
  InterpolationMode resample = static_cast<InterpolationMode>(op_params["resample"]);
  bool expand = op_params["expand"];
  std::vector<float> center = op_params["center"];
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  *operation = std::make_shared<vision::RandomRotationOperation>(degrees, resample, expand, center, fill_value);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
