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
struct Compose::Data {
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

Compose::Compose(const std::vector<TensorTransform *> &transforms) : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform *const op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

Compose::Compose(const std::vector<std::shared_ptr<TensorTransform>> &transforms) : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

Compose::Compose(const std::vector<std::reference_wrapper<TensorTransform>> &transforms)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> Compose::Parse() { return std::make_shared<ComposeOperation>(data_->transforms_); }

// Constructor to Duplicate
Duplicate::Duplicate() {}

std::shared_ptr<TensorOperation> Duplicate::Parse() { return std::make_shared<DuplicateOperation>(); }

// Constructor to OneHot
struct OneHot::Data {
  explicit Data(int32_t num_classes) : num_classes_(num_classes) {}
  float num_classes_;
};

OneHot::OneHot(int32_t num_classes) : data_(std::make_shared<Data>(num_classes)) {}

std::shared_ptr<TensorOperation> OneHot::Parse() { return std::make_shared<OneHotOperation>(data_->num_classes_); }

// Constructor to RandomApply.
struct RandomApply::Data {
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  double prob_;
};

RandomApply::RandomApply(const std::vector<TensorTransform *> &transforms, double prob)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform *const op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
  data_->prob_ = prob;
}

RandomApply::RandomApply(const std::vector<std::shared_ptr<TensorTransform>> &transforms, double prob)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
  data_->prob_ = prob;
}

RandomApply::RandomApply(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, double prob)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
  data_->prob_ = prob;
}

std::shared_ptr<TensorOperation> RandomApply::Parse() {
  return std::make_shared<RandomApplyOperation>(data_->transforms_, data_->prob_);
}

// Constructor to RandomChoice.
struct RandomChoice::Data {
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

RandomChoice::RandomChoice(const std::vector<TensorTransform *> &transforms) : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform *const op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

RandomChoice::RandomChoice(const std::vector<std::shared_ptr<TensorTransform>> &transforms)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

RandomChoice::RandomChoice(const std::vector<std::reference_wrapper<TensorTransform>> &transforms)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> RandomChoice::Parse() {
  return std::make_shared<RandomChoiceOperation>(data_->transforms_);
}

// Constructor to TypeCast
struct TypeCast::Data {
  explicit Data(const std::vector<char> &data_type) : data_type_(CharToString(data_type)) {}
  std::string data_type_;
};

TypeCast::TypeCast(const std::vector<char> &data_type) : data_(std::make_shared<Data>(data_type)) {}

std::shared_ptr<TensorOperation> TypeCast::Parse() { return std::make_shared<TypeCastOperation>(data_->data_type_); }

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
