/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/dataset/transforms.h"

#include <algorithm>

#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/general/transform/transforms_ir.h"
#include "ir/dtype/type_id.h"

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
                       [](const std::shared_ptr<TensorTransform> &op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

Compose::Compose(const std::vector<std::reference_wrapper<TensorTransform>> &transforms)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> Compose::Parse() { return std::make_shared<ComposeOperation>(data_->transforms_); }

// Constructor to Concatenate

// Constructor to Duplicate

// Constructor to Fill

// Constructor to Mask

// Constructor to OneHot

// Constructor to PadEnd

// Constructor to RandomApply.

// Constructor to RandomChoice.

// Constructor to Slice

// Constructor to TypeCast
struct TypeCast::Data {
  dataset::DataType data_type_;
};

TypeCast::TypeCast(mindspore::DataType data_type) : data_(std::make_shared<Data>()) {
  data_->data_type_ = dataset::MSTypeToDEType(static_cast<TypeId>(data_type));
}

std::shared_ptr<TensorOperation> TypeCast::Parse() { return std::make_shared<TypeCastOperation>(data_->data_type_); }

// Constructor to Unique

}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
