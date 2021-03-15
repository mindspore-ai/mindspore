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

#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

// Kernel data headers (in alphabetical order)
#include "minddata/dataset/kernels/data/compose_op.h"
#include "minddata/dataset/kernels/data/duplicate_op.h"
#include "minddata/dataset/kernels/data/one_hot_op.h"
#include "minddata/dataset/kernels/data/random_apply_op.h"
#include "minddata/dataset/kernels/data/random_choice_op.h"
#include "minddata/dataset/kernels/data/type_cast_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/data/unique_op.h"
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
                       [](std::shared_ptr<TensorOperation> op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  return std::make_shared<ComposeOp>(tensor_ops);
}

// DuplicateOperation
Status DuplicateOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DuplicateOperation::Build() { return std::make_shared<DuplicateOp>(); }

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

// PreBuiltOperation
PreBuiltOperation::PreBuiltOperation(std::shared_ptr<TensorOp> tensor_op) : op_(tensor_op) {
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
  return std::make_shared<RandomApplyOp>(prob_, tensor_ops);
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
                       [](std::shared_ptr<TensorOperation> op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  return std::make_shared<RandomChoiceOp>(tensor_ops);
}

// TypeCastOperation
TypeCastOperation::TypeCastOperation(std::string data_type) : data_type_(data_type) {}

Status TypeCastOperation::ValidateParams() {
  std::vector<std::string> predefine_type = {"bool",  "int8",   "uint8",   "int16",   "uint16",  "int32", "uint32",
                                             "int64", "uint64", "float16", "float32", "float64", "string"};
  auto itr = std::find(predefine_type.begin(), predefine_type.end(), data_type_);
  if (itr == predefine_type.end()) {
    std::string err_msg = "TypeCast: Invalid data type: " + data_type_;
    MS_LOG(ERROR) << "TypeCast: Only supports data type bool, int8, uint8, int16, uint16, int32, uint32, "
                  << "int64, uint64, float16, float32, float64, string, but got: " << data_type_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> TypeCastOperation::Build() { return std::make_shared<TypeCastOp>(data_type_); }

#ifndef ENABLE_ANDROID
// UniqueOperation
Status UniqueOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> UniqueOperation::Build() { return std::make_shared<UniqueOp>(); }
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
