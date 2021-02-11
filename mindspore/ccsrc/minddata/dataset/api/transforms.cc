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

// Constructor to Duplicate
Duplicate::Duplicate() {}

std::shared_ptr<TensorOperation> Duplicate::Parse() { return std::make_shared<DuplicateOperation>(); }

// Constructor to OneHot
OneHot::OneHot(int32_t num_classes) : num_classes_(num_classes) {}

std::shared_ptr<TensorOperation> OneHot::Parse() { return std::make_shared<OneHotOperation>(num_classes_); }

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

// Constructor to TypeCast
TypeCast::TypeCast(std::string data_type) : data_type_(data_type) {}

std::shared_ptr<TensorOperation> TypeCast::Parse() { return std::make_shared<TypeCastOperation>(data_type_); }

#ifndef ENABLE_ANDROID
// Constructor to Unique
Unique::Unique() {}

std::shared_ptr<TensorOperation> Unique::Parse() { return std::make_shared<UniqueOperation>(); }
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
