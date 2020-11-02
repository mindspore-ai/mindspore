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
#include "minddata/dataset/kernels/data/one_hot_op.h"
#include "minddata/dataset/kernels/data/type_cast_op.h"

namespace mindspore {
namespace dataset {

TensorOperation::TensorOperation() {}

// Transform operations for data.
namespace transforms {

// FUNCTIONS TO CREATE DATA TRANSFORM OPERATIONS
// (In alphabetical order)

// Function to create OneHotOperation.
std::shared_ptr<OneHotOperation> OneHot(int32_t num_classes) {
  auto op = std::make_shared<OneHotOperation>(num_classes);
  // Input validation
  if (!op->ValidateParams()) {
    return nullptr;
  }
  return op;
}

// Function to create TypeCastOperation.
std::shared_ptr<TypeCastOperation> TypeCast(std::string data_type) {
  auto op = std::make_shared<TypeCastOperation>(data_type);
  // Input validation
  if (!op->ValidateParams()) {
    return nullptr;
  }
  return op;
}

/* ####################################### Validator Functions ############################################ */

/* ####################################### Derived TensorOperation classes ################################# */

// (In alphabetical order)

// OneHotOperation
OneHotOperation::OneHotOperation(int32_t num_classes) : num_classes_(num_classes) {}

Status OneHotOperation::ValidateParams() {
  if (num_classes_ <= 0) {
    std::string err_msg =
      "OneHot: Number of classes must be greater than 0. num_classes: " + std::to_string(num_classes_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> OneHotOperation::Build() { return std::make_shared<OneHotOp>(num_classes_); }

// TypeCastOperation
TypeCastOperation::TypeCastOperation(std::string data_type) : data_type_(data_type) {}

Status TypeCastOperation::ValidateParams() {
  std::vector<std::string> predefine_type = {"bool",  "int8",   "uint8",   "int16",   "uint16",  "int32", "uint32",
                                             "int64", "uint64", "float16", "float32", "float64", "string"};
  auto itr = std::find(predefine_type.begin(), predefine_type.end(), data_type_);
  if (itr == predefine_type.end()) {
    std::string err_msg = "TypeCast: Invalid data type: " + data_type_;
    MS_LOG(ERROR) << "TypeCast: Only supports data type bool, int8, uint8, int16, uint16, int32, uint32, "
                  << "int64, uint64, float16, float32, float64, string, but got " << data_type_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> TypeCastOperation::Build() { return std::make_shared<TypeCastOp>(data_type_); }

}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
