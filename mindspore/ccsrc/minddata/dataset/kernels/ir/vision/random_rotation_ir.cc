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

// Function to create RandomRotationOperation.
RandomRotationOperation::RandomRotationOperation(std::vector<float> degrees, InterpolationMode resample, bool expand,
                                                 std::vector<float> center, std::vector<uint8_t> fill_value)
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
  if (degrees_.size() != 2 && degrees_.size() != 1) {
    std::string err_msg =
      "RandomRotation: degrees must be a vector of one or two values, got: " + std::to_string(degrees_.size());
    MS_LOG(ERROR) << "RandomRotation: degrees must be a vector of one or two values, got: " << degrees_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if ((degrees_.size() == 2) && (degrees_[1] < degrees_[0])) {
    std::string err_msg = "RandomRotation: degrees must be in the format of (min, max), got: (" +
                          std::to_string(degrees_[0]) + ", " + std::to_string(degrees_[1]) + ")";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  } else if ((degrees_.size() == 1) && (degrees_[0] < 0)) {
    std::string err_msg =
      "RandomRotation: if degrees only has one value, it must be greater than or equal to 0, got: " +
      std::to_string(degrees_[0]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // center
  if (center_.size() != 0 && center_.size() != 2) {
    std::string err_msg =
      "RandomRotation: center must be a vector of two values or empty, got: " + std::to_string(center_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("RandomRotation", fill_value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomRotationOperation::Build() {
  float start_degree, end_degree;
  if (degrees_.size() == 1) {
    start_degree = -degrees_[0];
    end_degree = degrees_[0];
  } else if (degrees_.size() == 2) {
    start_degree = degrees_[0];
    end_degree = degrees_[1];
  }

  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[0];
  fill_g = fill_value_[0];
  fill_b = fill_value_[0];

  if (fill_value_.size() == 3) {
    fill_r = fill_value_[0];
    fill_g = fill_value_[1];
    fill_b = fill_value_[2];
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

#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
