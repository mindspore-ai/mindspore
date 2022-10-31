/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/rescale_ir.h"

#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
#include "minddata/dataset/kernels/image/rescale_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
// RescaleOperation
RescaleOperation::RescaleOperation(float rescale, float shift) : rescale_(rescale), shift_(shift) {}

RescaleOperation::~RescaleOperation() = default;

std::string RescaleOperation::Name() const { return kRescaleOperation; }

Status RescaleOperation::ValidateParams() {
  if (rescale_ < 0) {
    std::string err_msg = "Rescale: rescale must be greater than or equal to 0, got: " + std::to_string(rescale_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RescaleOperation::Build() {
  std::shared_ptr<RescaleOp> tensor_op = std::make_shared<RescaleOp>(rescale_, shift_);
  return tensor_op;
}

Status RescaleOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["rescale"] = rescale_;
  args["shift"] = shift_;
  *out_json = args;
  return Status::OK();
}

Status RescaleOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "rescale", kRescaleOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "shift", kRescaleOperation));
  float rescale = op_params["rescale"];
  float shift = op_params["shift"];
  *operation = std::make_shared<vision::RescaleOperation>(rescale, shift);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
