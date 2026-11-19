/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/general/transform/transforms_ir.h"

#include <algorithm>
#include <utility>

#include "minddata/dataset/engine/serdes.h"

// Kernel data headers (in alphabetical order)
#include "minddata/dataset/general/kernels/compose_op.h"
#include "minddata/dataset/general/kernels/concatenate_op.h"
#include "minddata/dataset/general/kernels/parse_example_op.h"
#include "minddata/dataset/general/kernels/slice_op.h"
#include "minddata/dataset/general/kernels/type_cast_op.h"

#include "minddata/dataset/kernels/validators.h"
#include "minddata/dataset/kernels/plugin_op.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/kernels/py_func_op.h"
#endif
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
// Transform operations for data.
namespace transforms {
/* ####################################### Derived TensorOperation classes ################################# */
// (In alphabetical order)

// ComposeOperation
ComposeOperation::ComposeOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms)
    : transforms_(transforms) {}

Status ComposeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorTransforms("Compose", transforms_));
  return Status::OK();
}

std::shared_ptr<TensorOp> ComposeOperation::Build() {
  std::vector<std::shared_ptr<TensorOp>> tensor_ops;
  (void)std::transform(transforms_.begin(), transforms_.end(), std::back_inserter(tensor_ops),
                       [](const auto &op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  return std::make_shared<ComposeOp>(tensor_ops);
}

Status ComposeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  auto transforms = nlohmann::json::array();
  for (auto &tensor_operation : transforms_) {
    nlohmann::json tensor_op, args;
    RETURN_IF_NOT_OK(tensor_operation->to_json(&args));
    tensor_op["tensor_op_params"] = args;
    tensor_op["tensor_op_name"] = tensor_operation->Name();
    transforms.push_back(tensor_op);
  }
  (*out_json)["transforms"] = transforms;
  return Status::OK();
}

Status ComposeOperation::from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "transforms", kComposeOperation));
  nlohmann::json transforms = op_params["transforms"];
  std::vector<std::shared_ptr<TensorOperation>> operations;
  RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(transforms, &operations));
  *operation = std::make_shared<transforms::ComposeOperation>(operations);
  return Status::OK();
}

// DuplicateOperation

#if !defined(_WIN32) && !defined(_WIN64)
// ParseExampleOperation
ParseExampleOperation::ParseExampleOperation(DataSchema schema, std::vector<std::string> column_list,
                                             bool parallel_parse)
    : schema_(std::move(schema)), column_list_(std::move(column_list)), parallel_parse_(parallel_parse) {}

std::shared_ptr<TensorOp> ParseExampleOperation::Build() {
  return std::make_shared<ParseExampleOp>(schema_, column_list_, parallel_parse_);
}
#endif

// PreBuiltOperation
PreBuiltOperation::PreBuiltOperation(std::shared_ptr<TensorOp> tensor_op) : op_(std::move(tensor_op)) {
#ifdef ENABLE_PYTHON
  auto pyfunc_tensor_op = std::dynamic_pointer_cast<PyFuncOp>(op_);
  if (pyfunc_tensor_op && pyfunc_tensor_op->IsRandom()) {
    random_op_ = true;
  }
#endif
}

Status PreBuiltOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> PreBuiltOperation::Build() {
  if (op_->Name() == "PyFuncOp") {
    // create new tensor_op so that each worker has its own tensor_op
    return std::make_shared<PyFuncOp>(std::dynamic_pointer_cast<PyFuncOp>(op_));
  }
  return op_;
}

std::string PreBuiltOperation::Name() const { return op_ ? op_->Name() : kPreBuiltOperation; }

Status PreBuiltOperation::to_json(nlohmann::json *out_json) {
  RETURN_IF_NOT_OK(op_->to_json(out_json));
  return Status::OK();
}

// TypeCastOperation
// DataType data_type - required for C++ API
TypeCastOperation::TypeCastOperation(const DataType &data_type) : data_type_(data_type) {}

// std::string data_type - required for Pybind
TypeCastOperation::TypeCastOperation(const std::string &data_type) {
  // Convert from string to DEType
  DataType temp_data_type(data_type);
  data_type_ = temp_data_type;
}

Status TypeCastOperation::ValidateParams() {
  if (data_type_ == DataType::DE_UNKNOWN) {
    std::string err_msg = "TypeCast: Invalid data type";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> TypeCastOperation::Build() { return std::make_shared<TypeCastOp>(data_type_); }

Status TypeCastOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["data_type"] = data_type_.ToString();
  *out_json = args;
  return Status::OK();
}

Status TypeCastOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "data_type", kTypeCastOperation));
  std::string data_type = op_params["data_type"];
  *operation = std::make_shared<transforms::TypeCastOperation>(data_type);
  return Status::OK();
}

// UniqueOperation

Status PluginOperation::ValidateParams() {
  std::string err_msg;
  err_msg += lib_path_.empty() ? "lib_path is empty, please specify a path to .so file. " : "";
  err_msg += func_name_.empty() ? "func_name_ is empty, please specify function name to load." : "";
  if (!err_msg.empty()) {
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}
std::shared_ptr<TensorOp> PluginOperation::Build() {
  return std::make_shared<PluginOp>(lib_path_, func_name_, user_args_);
}

Status PluginOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["lib_path"] = lib_path_;
  args["func_name"] = func_name_;
  args["user_args"] = user_args_;
  *out_json = args;
  return Status::OK();
}

Status PluginOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "lib_path", kPluginOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "func_name", kPluginOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "user_args", kPluginOperation));
  std::string lib_path = op_params["lib_path"];
  std::string func_name = op_params["func_name"];
  std::string user_args = op_params["user_args"];
  *operation = std::make_shared<transforms::PluginOperation>(lib_path, func_name, user_args);
  return Status::OK();
}
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
