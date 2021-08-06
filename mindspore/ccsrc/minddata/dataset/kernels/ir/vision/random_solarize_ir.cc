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

#include "minddata/dataset/kernels/ir/vision/random_solarize_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_solarize_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomSolarizeOperation.
RandomSolarizeOperation::RandomSolarizeOperation(const std::vector<uint8_t> &threshold)
    : TensorOperation(true), threshold_(threshold) {}

RandomSolarizeOperation::~RandomSolarizeOperation() = default;

std::string RandomSolarizeOperation::Name() const { return kRandomSolarizeOperation; }

Status RandomSolarizeOperation::ValidateParams() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;
  constexpr uint8_t kThresholdMax = 255;

  if (threshold_.size() != size_two) {
    std::string err_msg =
      "RandomSolarize: threshold must be a vector of two values, got: " + std::to_string(threshold_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (size_t i = 0; i < threshold_.size(); ++i) {
    if (threshold_[i] < 0 || threshold_[i] > kThresholdMax) {
      std::string err_msg =
        "RandomSolarize: threshold has to be between 0 and 255, got:" + std::to_string(threshold_[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (threshold_[dimension_zero] > threshold_[dimension_one]) {
    std::string err_msg = "RandomSolarize: threshold must be passed in a (min, max) format";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSolarizeOperation::Build() {
  std::shared_ptr<RandomSolarizeOp> tensor_op = std::make_shared<RandomSolarizeOp>(threshold_);
  return tensor_op;
}

Status RandomSolarizeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["threshold"] = threshold_;
  return Status::OK();
}

Status RandomSolarizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("threshold") != op_params.end(), "Failed to find threshold");
  std::vector<uint8_t> threshold = op_params["threshold"];
  *operation = std::make_shared<vision::RandomSolarizeOperation>(threshold);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
