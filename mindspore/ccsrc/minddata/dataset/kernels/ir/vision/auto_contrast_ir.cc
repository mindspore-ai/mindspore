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

#include "minddata/dataset/kernels/ir/vision/auto_contrast_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/auto_contrast_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AutoContrastOperation
AutoContrastOperation::AutoContrastOperation(float cutoff, const std::vector<uint32_t> &ignore)
    : cutoff_(cutoff), ignore_(ignore) {}

AutoContrastOperation::~AutoContrastOperation() = default;

std::string AutoContrastOperation::Name() const { return kAutoContrastOperation; }

Status AutoContrastOperation::ValidateParams() {
  constexpr int64_t max_cutoff = 100;
  if (cutoff_ < 0 || cutoff_ > max_cutoff) {
    std::string err_msg = "AutoContrast: 'cutoff' has to be between 0 and 100, got: " + std::to_string(cutoff_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  constexpr uint32_t kMaxIgnoreSize = 255;
  for (uint32_t single_ignore : ignore_) {
    if (single_ignore > kMaxIgnoreSize) {
      std::string err_msg =
        "AutoContrast: invalid size, 'ignore' has to be between 0 and 255, got: " + std::to_string(single_ignore);
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> AutoContrastOperation::Build() {
  std::shared_ptr<AutoContrastOp> tensor_op = std::make_shared<AutoContrastOp>(cutoff_, ignore_);
  return tensor_op;
}

Status AutoContrastOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["cutoff"] = cutoff_;
  args["ignore"] = ignore_;
  *out_json = args;
  return Status::OK();
}

Status AutoContrastOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "cutoff", kAutoContrastOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "ignore", kAutoContrastOperation));
  float cutoff = op_params["cutoff"];
  std::vector<uint32_t> ignore = op_params["ignore"];
  *operation = std::make_shared<vision::AutoContrastOperation>(cutoff, ignore);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
