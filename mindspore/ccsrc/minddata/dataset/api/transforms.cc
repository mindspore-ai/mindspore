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

#include "minddata/dataset/include/transforms.h"

namespace mindspore {
namespace dataset {

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
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
