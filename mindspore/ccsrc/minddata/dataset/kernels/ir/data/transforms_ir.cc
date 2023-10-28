/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

#include <algorithm>
#include <utility>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif

// Kernel data headers (in alphabetical order)
#include "minddata/dataset/kernels/data/compose_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/data/concatenate_op.h"
#endif
#include "minddata/dataset/kernels/data/duplicate_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/data/fill_op.h"
#include "minddata/dataset/kernels/data/mask_op.h"
#endif
#include "minddata/dataset/kernels/data/one_hot_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/data/pad_end_op.h"
#endif
#include "minddata/dataset/kernels/data/random_apply_op.h"
#include "minddata/dataset/kernels/data/random_choice_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/data/slice_op.h"
#endif
#include "minddata/dataset/kernels/data/type_cast_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/data/unique_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/plugin_op.h"
#endif
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

#ifndef ENABLE_ANDROID
Status ComposeOperation::from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "transforms", kComposeOperation));
  nlohmann::json transforms = op_params["transforms"];
  std::vector<std::shared_ptr<TensorOperation>> operations;
  RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(transforms, &operations));
  *operation = std::make_shared<transforms::ComposeOperation>(operations);
  return Status::OK();
}

// ConcatenateOperation
ConcatenateOperation::ConcatenateOperation(int8_t axis, const std::shared_ptr<Tensor> &prepend,
                                           const std::shared_ptr<Tensor> &append)
    : axis_(axis), prepend_(prepend), append_(append) {}

Status ConcatenateOperation::ValidateParams() {
  if (axis_ != 0 && axis_ != -1) {
    std::string err_msg =
      "Concatenate: Only 1D concatenation supported, input 'axis' should be 0 or -1, but got:" + std::to_string(axis_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (prepend_) {
    if (prepend_->shape().Size() != 1) {
      std::string err_msg = "Concatenate: Can only prepend 1D arrays, rank of input 'prepend' should be 1, but got:" +
                            std::to_string(prepend_->shape().Size());
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (append_) {
    if (append_->shape().Size() != 1) {
      std::string err_msg = "Concatenate: Can only append 1D arrays, rank of input 'append' should be 1, but got:" +
                            std::to_string(append_->shape().Size());
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ConcatenateOperation::Build() {
  return std::make_shared<ConcatenateOp>(axis_, prepend_, append_);
}

Status ConcatenateOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["axis"] = axis_;
  nlohmann::json prepend;
  nlohmann::json append;
  RETURN_IF_NOT_OK(prepend_->to_json(&prepend));
  RETURN_IF_NOT_OK(append_->to_json(&append));
  args["prepend"] = prepend;
  args["append"] = append;
  *out_json = args;
  return Status::OK();
}

Status ConcatenateOperation::from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "axis", kConcatenateOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prepend", kConcatenateOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "append", kConcatenateOperation));
  int8_t axis = op_params["axis"];
  std::shared_ptr<Tensor> prepend;
  std::shared_ptr<Tensor> append;
  RETURN_IF_NOT_OK(Tensor::from_json(op_params["prepend"], &prepend));
  RETURN_IF_NOT_OK(Tensor::from_json(op_params["append"], &append));
  *operation = std::make_shared<transforms::ConcatenateOperation>(axis, prepend, append);
  return Status::OK();
}
#endif

// DuplicateOperation
Status DuplicateOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DuplicateOperation::Build() { return std::make_shared<DuplicateOp>(); }

Status DuplicateOperation::from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<transforms::DuplicateOperation>();
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// FillOperation
FillOperation::FillOperation(const std::shared_ptr<Tensor> &fill_value) : fill_value_(fill_value) {}

Status FillOperation::ValidateParams() {
  if (fill_value_->shape() != TensorShape::CreateScalar()) {
    std::string err_msg = "Fill: fill_value is not a scalar tensor, got shape:" + fill_value_->shape().ToString();
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> FillOperation::Build() { return std::make_shared<FillOp>(fill_value_); }

Status FillOperation::to_json(nlohmann::json *out_json) {
  RETURN_IF_NOT_OK(fill_value_->to_json(out_json));
  return Status::OK();
}

Status FillOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  std::shared_ptr<Tensor> fill_value;
  RETURN_IF_NOT_OK(Tensor::from_json(op_params, &fill_value));
  *operation = std::make_shared<transforms::FillOperation>(fill_value);
  return Status::OK();
}

// MaskOperation
MaskOperation::MaskOperation(RelationalOp op, const std::shared_ptr<Tensor> &constant, const DataType &dtype)
    : op_(op), constant_(constant), dtype_(dtype) {}

Status MaskOperation::ValidateParams() {
  if (!dtype_.IsBool() && !dtype_.IsFloat() && !dtype_.IsInt()) {
    std::string err_msg =
      "Mask: Only supports bool or numeric datatype for generated mask type, but got:" + dtype_.ToString();
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> MaskOperation::Build() { return std::make_shared<MaskOp>(op_, constant_, dtype_); }

Status MaskOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["op"] = op_;
  nlohmann::json constant;
  RETURN_IF_NOT_OK(constant_->to_json(&constant));
  args["constant"] = constant;
  args["dtype"] = dtype_.value();
  *out_json = args;
  return Status::OK();
}

Status MaskOperation::from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "op", kMaskOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "constant", kMaskOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "dtype", kMaskOperation));
  RelationalOp op = op_params["op"];
  std::shared_ptr<Tensor> constant;
  RETURN_IF_NOT_OK(Tensor::from_json(op_params["constant"], &constant));
  auto dtype = DataType(static_cast<DataType::Type>(op_params["dtype"]));
  *operation = std::make_shared<transforms::MaskOperation>(op, constant, dtype);
  return Status::OK();
}
#endif

// OneHotOperation
OneHotOperation::OneHotOperation(int32_t num_classes, double smoothing_rate)
    : num_classes_(num_classes), smoothing_rate_(smoothing_rate) {}

Status OneHotOperation::ValidateParams() {
  if (num_classes_ <= 0) {
    std::string err_msg = "OneHot: Number of classes must be greater than 0, but got: " + std::to_string(num_classes_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (smoothing_rate_ < 0.0 || smoothing_rate_ > 1.0) {
    std::string err_msg = "OneHot: Smoothing rate must be between 0 and 1, but got: " + std::to_string(smoothing_rate_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> OneHotOperation::Build() { return std::make_shared<OneHotOp>(num_classes_, smoothing_rate_); }

Status OneHotOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["num_classes"] = num_classes_;
  args["smoothing_rate"] = smoothing_rate_;

  *out_json = args;
  return Status::OK();
}

Status OneHotOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "num_classes", kOneHotOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "smoothing_rate", kOneHotOperation));
  int32_t num_classes = op_params["num_classes"];
  double smoothing_rate = op_params["smoothing_rate"];
  *operation = std::make_shared<transforms::OneHotOperation>(num_classes, smoothing_rate);
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// PadEndOperation
PadEndOperation::PadEndOperation(const TensorShape &pad_shape, const std::shared_ptr<Tensor> &pad_value)
    : pad_shape_(pad_shape), pad_value_(pad_value) {}

Status PadEndOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> PadEndOperation::Build() { return std::make_shared<PadEndOp>(pad_shape_, pad_value_); }

Status PadEndOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["pad_shape"] = pad_shape_.AsVector();
  nlohmann::json pad_value;
  RETURN_IF_NOT_OK(pad_value_->to_json(&pad_value));
  args["pad_value"] = pad_value;
  *out_json = args;
  return Status::OK();
}

Status PadEndOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "pad_shape", kPadEndOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "pad_value", kPadEndOperation));
  std::vector<dsize_t> shape_vector = op_params["pad_shape"];
  TensorShape pad_shape = TensorShape(shape_vector);
  std::shared_ptr<Tensor> pad_value;
  RETURN_IF_NOT_OK(Tensor::from_json(op_params["pad_value"], &pad_value));
  *operation = std::make_shared<transforms::PadEndOperation>(pad_shape, pad_value);
  return Status::OK();
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

std::shared_ptr<TensorOp> PreBuiltOperation::Build() { return op_; }

std::string PreBuiltOperation::Name() const { return op_ ? op_->Name() : kPreBuiltOperation; }

Status PreBuiltOperation::to_json(nlohmann::json *out_json) {
  RETURN_IF_NOT_OK(op_->to_json(out_json));
  return Status::OK();
}

// RandomApplyOperation
RandomApplyOperation::RandomApplyOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms, double prob)
    : TensorOperation(true), transforms_(transforms), prob_(prob) {}

Status RandomApplyOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorTransforms("RandomApply", transforms_));
  RETURN_IF_NOT_OK(ValidateProbability("RandomApply", prob_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomApplyOperation::Build() {
  std::vector<std::shared_ptr<TensorOp>> tensor_ops;
  (void)std::transform(transforms_.begin(), transforms_.end(), std::back_inserter(tensor_ops),
                       [](std::shared_ptr<TensorOperation> op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  return std::make_shared<RandomApplyOp>(tensor_ops, prob_);
}

Status RandomApplyOperation::to_json(nlohmann::json *out_json) {
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
  (*out_json)["prob"] = prob_;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status RandomApplyOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "transforms", kRandomApplyOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomApplyOperation));
  nlohmann::json transforms = op_params["transforms"];
  std::vector<std::shared_ptr<TensorOperation>> operations;
  RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(transforms, &operations));
  double prob = op_params["prob"];
  *operation = std::make_shared<transforms::RandomApplyOperation>(operations, prob);
  return Status::OK();
}
#endif

// RandomChoiceOperation
RandomChoiceOperation::RandomChoiceOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms)
    : TensorOperation(true), transforms_(transforms) {}

Status RandomChoiceOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorTransforms("RandomChoice", transforms_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomChoiceOperation::Build() {
  std::vector<std::shared_ptr<TensorOp>> tensor_ops;
  (void)std::transform(transforms_.begin(), transforms_.end(), std::back_inserter(tensor_ops),
                       [](const auto &op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  return std::make_shared<RandomChoiceOp>(tensor_ops);
}

Status RandomChoiceOperation::to_json(nlohmann::json *out_json) {
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

#ifndef ENABLE_ANDROID
Status RandomChoiceOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "transforms", kRandomChoiceOperation));
  nlohmann::json transforms = op_params["transforms"];
  std::vector<std::shared_ptr<TensorOperation>> operations;
  RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(transforms, &operations));
  *operation = std::make_shared<transforms::RandomChoiceOperation>(operations);
  return Status::OK();
}

// SliceOperation
SliceOperation::SliceOperation(const std::vector<SliceOption> &slice_input) : slice_input_(slice_input) {}

Status SliceOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> SliceOperation::Build() { return std::make_shared<SliceOp>(slice_input_); }
#endif

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

#ifndef ENABLE_ANDROID
// UniqueOperation
Status UniqueOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> UniqueOperation::Build() { return std::make_shared<UniqueOp>(); }

Status UniqueOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<transforms::UniqueOperation>();
  return Status::OK();
}

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
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
