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
#include "minddata/dataset/kernels/image/image_utils.h"

// Kernel data headers (in alphabetical order)
#include "minddata/dataset/kernels/data/one_hot_op.h"

namespace mindspore {
namespace dataset {
namespace api {

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

/* ####################################### Validator Functions ############################################ */

/* ####################################### Derived TensorOperation classes ################################# */

// (In alphabetical order)

// OneHotOperation
OneHotOperation::OneHotOperation(int32_t num_classes) : num_classes_(num_classes) {}

bool OneHotOperation::ValidateParams() {
  if (num_classes_ < 0) {
    MS_LOG(ERROR) << "OneHot: Number of classes cannot be negative. Number of classes: " << num_classes_;
    return false;
  }

  return true;
}

std::shared_ptr<TensorOp> OneHotOperation::Build() { return std::make_shared<OneHotOp>(num_classes_); }

}  // namespace transforms
}  // namespace api
}  // namespace dataset
}  // namespace mindspore
