/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "kernel/kernel.h"

#include <algorithm>
#include <stack>
#include "utils/ms_context.h"
#include "utils/anf_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kInvalidShape = -2;

TypeId KernelTensor::GetDtype() const {
  if (tensor_info_.abstract_base == nullptr) {
    return TypeId::kTypeUnknown;
  }

  auto type_ptr = tensor_info_.abstract_base->BuildType();
  if (type_ptr == nullptr || !type_ptr->isa<TensorType>()) {
    return TypeId::kTypeUnknown;
  }

  auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
  auto elem = tensor_ptr->element();
  if (elem == nullptr) {
    return TypeId::kTypeUnknown;
  }
  return elem->type_id();
}

std::vector<size_t> KernelTensor::GetShapeVector() const {
  auto base_shape_ptr = GetBaseShape();
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::Shape>()) {
    return {};
  }
  auto shape = base_shape_ptr->cast<abstract::ShapePtr>()->shape();
  std::vector<size_t> out_shape;
  std::transform(shape.begin(), shape.end(), std::back_inserter(out_shape),
                 [](const int64_t &value) { return static_cast<size_t>(value); });
  return out_shape;
}

std::vector<TypeId> KernelTensor::GetListOrTupleDtype() const {
  if (tensor_info_.abstract_base == nullptr) {
    return {TypeId::kTypeUnknown};
  }

  auto type_ptr = tensor_info_.abstract_base->BuildType();
  if (type_ptr == nullptr || !type_ptr->isa<List>() || !type_ptr->isa<Tuple>()) {
    return {TypeId::kTypeUnknown};
  }

  std::vector<TypeId> types;
  if (type_ptr->isa<List>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                   [](const TypePtr &t) { return t->type_id(); });
  } else if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                   [](const TypePtr &t) { return t->type_id(); });
  } else {
    types.push_back(TypeId::kTypeUnknown);
  }

  return types;
}

std::vector<std::vector<size_t>> KernelTensor::GetListOrTupleShapeVector() const {
  auto base_shape_ptr = GetBaseShape();
  // ListShape or TupleShape is inherited from SequenceShape.
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::SequenceShape>()) {
    return {};
  }
  auto sequence_shape_ptr = base_shape_ptr->cast<abstract::SequenceShapePtr>();
  auto base_shape_list = sequence_shape_ptr->shape();
  std::vector<std::vector<size_t>> shape_vector_list;
  for (auto base_shape : base_shape_list) {
    if (base_shape == nullptr || !base_shape->isa<abstract::Shape>()) {
      return {};
    }
    auto tmp_shape = base_shape->cast<abstract::ShapePtr>()->shape();
    std::vector<size_t> cur_out_shape;
    std::transform(tmp_shape.begin(), tmp_shape.end(), std::back_inserter(cur_out_shape),
                   [](const int64_t &value) { return static_cast<size_t>(value); });
    shape_vector_list.push_back(cur_out_shape);
  }

  return shape_vector_list;
}

void KernelTensor::SetDtype(const TypePtr &dtype) {
  if (tensor_info_.abstract_base == nullptr) {
    return;
  }
  tensor_info_.abstract_base->set_type(dtype);
}

void KernelTensor::SetShapeVector(const std::vector<int64_t> &shape) {
  if (tensor_info_.abstract_base == nullptr) {
    return;
  }
  tensor_info_.abstract_base->set_shape(std::make_shared<abstract::Shape>(shape));
}

abstract::BaseShapePtr KernelTensor::GetBaseShape() const {
  if (tensor_info_.abstract_base == nullptr) {
    return nullptr;
  }
  return tensor_info_.abstract_base->BuildShape();
}

void KernelTensor::SetBaseShape(const abstract::BaseShapePtr &base_shape) {
  if (tensor_info_.abstract_base == nullptr) {
    return;
  }
  tensor_info_.abstract_base->set_shape(base_shape);
}
}  // namespace kernel
}  // namespace mindspore
