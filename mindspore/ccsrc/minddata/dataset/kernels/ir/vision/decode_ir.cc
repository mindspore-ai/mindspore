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
#include "minddata/dataset/kernels/ir/vision/decode_ir.h"

#include "minddata/dataset/kernels/image/decode_op.h"

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
// DecodeOperation
DecodeOperation::DecodeOperation(bool rgb) : rgb_(rgb) {}

DecodeOperation::~DecodeOperation() = default;

std::string DecodeOperation::Name() const { return kDecodeOperation; }

Status DecodeOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DecodeOperation::Build() { return std::make_shared<DecodeOp>(rgb_); }

Status DecodeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["rgb"] = rgb_;
  return Status::OK();
}
Status DecodeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("rgb") != op_params.end(), "Failed to find rgb");
  bool rgb = op_params["rgb"];
  *operation = std::make_shared<vision::DecodeOperation>(rgb);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
