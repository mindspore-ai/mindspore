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

#include <algorithm>

#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

namespace mindspore {
namespace dataset {

// Transform operations for data.
namespace transforms {

// API CLASS FOR DATA TRANSFORM OPERATIONS
// (In alphabetical order)

// Constructor to Compose.
Compose::Compose(const std::vector<TensorTransform *> &transforms) {
  (void)std::transform(
    transforms.begin(), transforms.end(), std::back_inserter(transforms_),
    [](TensorTransform *op) -> std::shared_ptr<TensorOperation> { return op != nullptr ? op->Parse() : nullptr; });
}

Compose::Compose(const std::vector<std::shared_ptr<TensorTransform>> &transforms) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

Compose::Compose(const std::vector<std::reference_wrapper<TensorTransform>> &transforms) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> Compose::Parse() { return std::make_shared<ComposeOperation>(transforms_); }

// Constructor to Duplicate
Duplicate::Duplicate() {}

std::shared_ptr<TensorOperation> Duplicate::Parse() { return std::make_shared<DuplicateOperation>(); }

// Constructor to OneHot
OneHot::OneHot(int32_t num_classes) : num_classes_(num_classes) {}

std::shared_ptr<TensorOperation> OneHot::Parse() { return std::make_shared<OneHotOperation>(num_classes_); }

// Constructor to RandomApply.
RandomApply::RandomApply(const std::vector<TensorTransform *> &transforms, double prob) : prob_(prob) {
  (void)std::transform(
    transforms.begin(), transforms.end(), std::back_inserter(transforms_),
    [](TensorTransform *op) -> std::shared_ptr<TensorOperation> { return op != nullptr ? op->Parse() : nullptr; });
}

RandomApply::RandomApply(const std::vector<std::shared_ptr<TensorTransform>> &transforms, double prob) : prob_(prob) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

RandomApply::RandomApply(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, double prob)
    : prob_(prob) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> RandomApply::Parse() {
  return std::make_shared<RandomApplyOperation>(transforms_, prob_);
}

// Constructor to RandomChoice.
RandomChoice::RandomChoice(const std::vector<TensorTransform *> &transforms) {
  (void)std::transform(
    transforms.begin(), transforms.end(), std::back_inserter(transforms_),
    [](TensorTransform *op) -> std::shared_ptr<TensorOperation> { return op != nullptr ? op->Parse() : nullptr; });
}

RandomChoice::RandomChoice(const std::vector<std::shared_ptr<TensorTransform>> &transforms) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

RandomChoice::RandomChoice(const std::vector<std::reference_wrapper<TensorTransform>> &transforms) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> RandomChoice::Parse() { return std::make_shared<RandomChoiceOperation>(transforms_); }

// Constructor to TypeCast
TypeCast::TypeCast(std::string data_type) : data_type_(data_type) {}

std::shared_ptr<TensorOperation> TypeCast::Parse() { return std::make_shared<TypeCastOperation>(data_type_); }

// Constructor to Unique
Unique::Unique() {}

#ifndef ENABLE_ANDROID
std::shared_ptr<TensorOperation> Unique::Parse() { return std::make_shared<UniqueOperation>(); }
#else
std::shared_ptr<TensorOperation> Unique::Parse() {
  MS_LOG(ERROR) << "Unique op is not supported for Android.";
  return nullptr;
}
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
