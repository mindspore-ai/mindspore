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

#include "minddata/dataset/kernels/ir/vision/convert_color_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/convert_color_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// ConvertColorOperation
ConvertColorOperation::ConvertColorOperation(ConvertMode convert_mode) : convert_mode_(convert_mode) {}

ConvertColorOperation::~ConvertColorOperation() = default;

std::string ConvertColorOperation::Name() const { return kConvertColorOperation; }

Status ConvertColorOperation::ValidateParams() {
  if (convert_mode_ < ConvertMode::COLOR_BGR2BGRA || convert_mode_ > ConvertMode::COLOR_RGBA2GRAY) {
    std::string err_msg = "ConvertColorOperation: convert_mode must be in ConvertMode.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ConvertColorOperation::Build() {
  std::shared_ptr<ConvertColorOp> tensor_op = std::make_shared<ConvertColorOp>(convert_mode_);
  return tensor_op;
}

Status ConvertColorOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["convert_mode"] = convert_mode_;
  *out_json = args;
  return Status::OK();
}

Status ConvertColorOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("convert_mode") != op_params.end(), "Failed to find convert_mode");
  ConvertMode convert_mode = static_cast<ConvertMode>(op_params["convert_mode"]);
  *operation = std::make_shared<vision::ConvertColorOperation>(convert_mode);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
