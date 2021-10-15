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
#include <utility>

#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

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
#include "minddata/dataset/kernels/plugin_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/kernels/py_func_op.h"
#endif

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

#ifndef ENABLE_ANDROID
// ConcatenateOperation
ConcatenateOperation::ConcatenateOperation(int8_t axis, const std::shared_ptr<Tensor> &prepend,
                                           const std::shared_ptr<Tensor> &append)
    : axis_(axis), prepend_(prepend), append_(append) {}

Status ConcatenateOperation::ValidateParams() {
  if (axis_ != 0 && axis_ != -1) {
    std::string err_msg = "Concatenate: Only 1D concatenation supported.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (prepend_) {
    if (prepend_->shape().Size() != 1) {
      std::string err_msg = "Concatenate: Can only prepend 1D arrays.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (append_) {
    if (append_->shape().Size() != 1) {
      std::string err_msg = "Concatenate: Can only append 1D arrays.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ConcatenateOperation::Build() {
  return std::make_shared<ConcatenateOp>(axis_, prepend_, append_);
}
#endif

// DuplicateOperation
Status DuplicateOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DuplicateOperation::Build() { return std::make_shared<DuplicateOp>(); }

#ifndef ENABLE_ANDROID

// FillOperation
FillOperation::FillOperation(const std::shared_ptr<Tensor> &fill_value) : fill_value_(fill_value) {}

Status FillOperation::ValidateParams() {
  if (fill_value_->shape() != TensorShape::CreateScalar()) {
    std::string err_msg = "Fill: fill_value is not a scalar tensor.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> FillOperation::Build() { return std::make_shared<FillOp>(fill_value_); }

Status FillOperation::to_json(nlohmann::json *out_json) {
  RETURN_IF_NOT_OK(fill_value_->to_json(out_json));
  return Status::OK();
}

Status FillOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
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
    std::string err_msg = "Mask: Only supports bool or numeric datatype for generated mask type.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> MaskOperation::Build() { return std::make_shared<MaskOp>(op_, constant_, dtype_); }
#endif

// OneHotOperation
OneHotOperation::OneHotOperation(int32_t num_classes) : num_classes_(num_classes) {}

Status OneHotOperation::ValidateParams() {
  if (num_classes_ <= 0) {
    std::string err_msg = "OneHot: Number of classes must be greater than 0, but got: " + std::to_string(num_classes_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> OneHotOperation::Build() { return std::make_shared<OneHotOp>(num_classes_); }

Status OneHotOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_classes"] = num_classes_;
  *out_json = args;
  return Status::OK();
}

Status OneHotOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("num_classes") != op_params.end(), "Failed tofind num_classes");
  int32_t num_classes = op_params["num_classes"];
  *operation = std::make_shared<transforms::OneHotOperation>(num_classes);
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// PadEndOperation
PadEndOperation::PadEndOperation(const TensorShape &pad_shape, const std::shared_ptr<Tensor> &pad_value)
    : pad_shape_(pad_shape), pad_value_(pad_value) {}

Status PadEndOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> PadEndOperation::Build() { return std::make_shared<PadEndOp>(pad_shape_, pad_value_); }
#endif

// PreBuiltOperation
PreBuiltOperation::PreBuiltOperation(std::shared_ptr<TensorOp> tensor_op) : op_(std::move(tensor_op)) {
#ifdef ENABLE_PYTHON
  auto pyfunc_tensor_op = std::dynamic_pointer_cast<PyFuncOp>(tensor_op);
  if (pyfunc_tensor_op && pyfunc_tensor_op->IsRandom()) random_op_ = true;
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

#ifndef ENABLE_ANDROID
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
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> TypeCastOperation::Build() { return std::make_shared<TypeCastOp>(data_type_); }

Status TypeCastOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["data_type"] = data_type_.ToString();
  *out_json = args;
  return Status::OK();
}

Status TypeCastOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("data_type") != op_params.end(), "Failed tofind data_type");
  std::string data_type = op_params["data_type"];
  *operation = std::make_shared<transforms::TypeCastOperation>(data_type);
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// UniqueOperation
Status UniqueOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> UniqueOperation::Build() { return std::make_shared<UniqueOp>(); }
Status PluginOperation::ValidateParams() {
  std::string err_msg;
  err_msg += lib_path_.empty() ? "lib_path is empty, please specify a path to .so file. " : "";
  err_msg += func_name_.empty() ? "func_name_ is empty, please specify function name to load." : "";
  if (!err_msg.empty()) {
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}
std::shared_ptr<TensorOp> PluginOperation::Build() {
  return std::make_shared<PluginOp>(lib_path_, func_name_, user_args_);
}
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
