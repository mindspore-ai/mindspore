/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/to_tensor_ir.h"

#include "minddata/dataset/kernels/image/to_tensor_op.h"

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
ToTensorOperation::ToTensorOperation(const std::string &data_type) {
  DataType temp_data_type(data_type);
  data_type_ = temp_data_type;
}

ToTensorOperation::~ToTensorOperation() = default;

std::string ToTensorOperation::Name() const { return kToTensorOperation; }

Status ToTensorOperation::ValidateParams() {
  if (data_type_ == DataType::DE_UNKNOWN) {
    std::string err_msg = "ToTensor: Invalid data type";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ToTensorOperation::Build() { return std::make_shared<ToTensorOp>(data_type_); }

Status ToTensorOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["data_type"] = data_type_.ToString();
  *out_json = args;
  return Status::OK();
}

Status ToTensorOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "data_type", kToTensorOperation));
  std::string data_type = op_params["data_type"];
  *operation = std::make_shared<vision::ToTensorOperation>(data_type);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
