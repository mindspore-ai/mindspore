/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/transforms.h"

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

namespace mindspore {
namespace dataset {

/* ####################################### Validator Functions ############################################ */
Status ValidateVectorFillvalue(const std::string &transform_name, const std::vector<uint8_t> &fill_value) {
  if (fill_value.empty() || (fill_value.size() != 1 && fill_value.size() != 3)) {
    std::string err_msg =
      transform_name + ": fill_value vector has incorrect size: " + std::to_string(fill_value.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (uint8_t single_fill_value : fill_value) {
    if (single_fill_value > 255) {
      std::string err_msg =
        transform_name + ": fill_value has to be between 0 and 255, got:" + std::to_string(single_fill_value);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

Status ValidateProbability(const std::string &transform_name, const float &probability) {
  if (probability < 0.0 || probability > 1.0) {
    std::string err_msg =
      transform_name + ": probability must be between 0.0 and 1.0, got: " + std::to_string(probability);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

Status ValidateVectorPadding(const std::string &transform_name, const std::vector<int32_t> &padding) {
  if (padding.empty() || padding.size() == 3 || padding.size() > 4) {
    std::string err_msg = transform_name + ": padding vector has incorrect size: " + std::to_string(padding.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < padding.size(); ++i) {
    if (padding[i] < 0) {
      std::string err_msg =
        transform_name +
        ": invalid padding, padding value must be greater than or equal to 0, got: " + std::to_string(padding[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (padding[i] == INT_MAX) {
      std::string err_msg =
        transform_name + ": invalid padding, padding value too large, got: " + std::to_string(padding[i]);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

Status ValidateVectorPositive(const std::string &transform_name, const std::vector<int32_t> &size) {
  for (int32_t i = 0; i < size.size(); ++i) {
    if (size[i] <= 0) {
      std::string err_msg =
        transform_name + ": Non-positive size value: " + std::to_string(size[i]) + " at element: " + std::to_string(i);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

Status ValidateVectorTransforms(const std::string &transform_name,
                                const std::vector<std::shared_ptr<TensorOperation>> &transforms) {
  if (transforms.empty()) {
    std::string err_msg = transform_name + ": transform list must not be empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < transforms.size(); ++i) {
    if (transforms[i] == nullptr) {
      std::string err_msg =
        transform_name + ": transform ops must not be null, got transform[" + std::to_string(i) + "] == nullptr.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

bool CmpFloat(const float &a, const float &b, float epsilon) { return (std::fabs(a - b) < epsilon); }

// Transform operations for data.
namespace transforms {

// FUNCTIONS TO CREATE DATA TRANSFORM OPERATIONS
// (In alphabetical order)

// Function to create ComposeOperation.
std::shared_ptr<ComposeOperation> Compose(const std::vector<std::shared_ptr<TensorOperation>> &transforms) {
  auto op = std::make_shared<ComposeOperation>(transforms);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create DuplicateOperation.
std::shared_ptr<DuplicateOperation> Duplicate() {
  auto op = std::make_shared<DuplicateOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create OneHotOperation.
std::shared_ptr<OneHotOperation> OneHot(int32_t num_classes) {
  auto op = std::make_shared<OneHotOperation>(num_classes);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomApplyOperation.
std::shared_ptr<RandomApplyOperation> RandomApply(const std::vector<std::shared_ptr<TensorOperation>> &transforms,
                                                  double prob) {
  auto op = std::make_shared<RandomApplyOperation>(transforms, prob);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create RandomChoiceOperation.
std::shared_ptr<RandomChoiceOperation> RandomChoice(const std::vector<std::shared_ptr<TensorOperation>> &transforms) {
  auto op = std::make_shared<RandomChoiceOperation>(transforms);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

// Function to create TypeCastOperation.
std::shared_ptr<TypeCastOperation> TypeCast(std::string data_type) {
  auto op = std::make_shared<TypeCastOperation>(data_type);
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}

#ifndef ENABLE_ANDROID
// Function to create UniqueOperation.
std::shared_ptr<UniqueOperation> Unique() {
  auto op = std::make_shared<UniqueOperation>();
  // Input validation
  return op->ValidateParams() ? op : nullptr;
}
#endif

/* ####################################### Validator Functions ############################################ */

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
PreBuiltOperation::PreBuiltOperation(std::shared_ptr<TensorOp> tensor_op) : op_(tensor_op) {}

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
