/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/ir/vision/solarize_ir.h"

#include "minddata/dataset/kernels/image/solarize_op.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// SolarizeOperation
SolarizeOperation::SolarizeOperation(const std::vector<float> &threshold) : threshold_(threshold) {}

SolarizeOperation::~SolarizeOperation() = default;

Status SolarizeOperation::ValidateParams() {
  constexpr size_t kThresholdSize = 2;
  constexpr float kThresholdMax = 255.0;

  if (threshold_.size() != kThresholdSize) {
    std::string err_msg =
      "Solarize: threshold must be a vector of two values, got: " + std::to_string(threshold_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (float threshold_value : threshold_) {
    if (threshold_value < 0 || threshold_value > kThresholdMax) {
      std::string err_msg = "Solarize: threshold has to be between 0 and 255, got:" + std::to_string(threshold_value);
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (threshold_[0] > threshold_[1]) {
    std::string err_msg = "Solarize: threshold must be passed in a (min, max) format";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SolarizeOperation::Build() {
  std::shared_ptr<SolarizeOp> tensor_op = std::make_shared<SolarizeOp>(threshold_);
  return tensor_op;
}

Status SolarizeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["threshold"] = threshold_;
  return Status::OK();
}

Status SolarizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "threshold", kSolarizeOperation));
  std::vector<float> threshold = op_params["threshold"];
  *operation = std::make_shared<vision::SolarizeOperation>(threshold);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
