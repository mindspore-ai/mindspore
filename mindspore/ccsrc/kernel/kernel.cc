/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <functional>
#include <iterator>
#include <numeric>
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kInvalidShape = -2;

string KernelTensor::GetAbstractName() const {
  const TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return "null(no abstract base)";
  }
  return info.base_->ToString();
}

bool KernelTensor::IsDynamicShape() const {
  auto shape = this->GetShapeVector();
  return std::any_of(shape.cbegin(), shape.cend(), [](auto i) { return i < 0; });
}

size_t KernelTensor::GetSizeInBytes() const {
  auto unit_size = GetTypeByte(TypeIdToType(GetDtype()));
  auto shapes = this->GetShapeVector();
  if (shapes.size() == 0) {
    return unit_size;
  }

  auto cur_size = unit_size;
  for (const auto val : shapes) {
    if (val < 0) {
      MS_LOG_EXCEPTION << "Invalid shape value " << val << " for calculating size. Abstract name: " << GetAbstractName()
                       << ". Please contact MindSpore support.";
    }
    if (val == 0) {
      MS_LOG_WARNING << "One dim of the shape is 0. Abstract name: " << GetAbstractName() << ".";
    }
    cur_size *= val;
  }

  return cur_size;
}

TypeId GetSeqElementsDtype(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractSequence>()) {
    return TypeId::kTypeUnknown;
  }
  TypePtr type_ptr;
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    if (seq_abs->dynamic_len_element_abs() == nullptr) {
      return TypeId::kTypeUnknown;
    }
    type_ptr = seq_abs->dynamic_len_element_abs()->BuildType();
  } else {
    if (seq_abs->elements().empty() || seq_abs->elements()[0] == nullptr) {
      return TypeId::kTypeUnknown;
    }
    type_ptr = seq_abs->elements()[0]->BuildType();
  }
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (type_ptr->isa<TensorType>()) {
    auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    auto elem = tensor_ptr->element();
    if (elem == nullptr) {
      return TypeId::kTypeUnknown;
    }
    return elem->type_id();
  }
  return type_ptr->type_id();
}

TypeId KernelTensor::GetDtype() const {
  if (meta_type_ == kObjectTypeNumber) {
    // Scalar
    const ScalarInfo &info = std::get<ScalarInfo>(meta_);
    return info.base_->BuildType()->type_id();
  } else if (meta_type_ == kObjectTypeTuple) {
    // Tuple
    const TupleInfo &info = std::get<TupleInfo>(meta_);
    return GetSeqElementsDtype(info.base_);
  } else if (meta_type_ == kObjectTypeList) {
    // List
    const ListInfo &info = std::get<ListInfo>(meta_);
    return GetSeqElementsDtype(info.base_);
  } else if (meta_type_ == kMetaTypeNone) {
    return TypeId::kMetaTypeNone;
  } else {
    // Tensor
    const TensorInfo &info = std::get<TensorInfo>(meta_);
    if (info.base_ == nullptr) {
      return TypeId::kTypeUnknown;
    }
    auto type_ptr = info.base_->BuildType();
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
  return kTypeUnknown;
}

ShapeVector GetSequenceFlattenShape(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractSequence>()) {
    return {};
  }
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    return {-1};
  }
  if (seq_abs->elements().empty() || seq_abs->elements()[0] == nullptr) {
    MS_LOG(INFO) << "Empty sequence abstract:" << seq_abs->ToString();
    return {0};
  }
  auto type_ptr = seq_abs->elements()[0]->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (!type_ptr->isa<TensorType>()) {
    return {(int64_t)seq_abs->elements().size()};
  }
  // for tuple of tensor, the tensors shape must be same
  ShapeVector flatten_shp;
  (void)flatten_shp.emplace_back(seq_abs->elements().size());
  if (seq_abs->elements().empty()) {
    return flatten_shp;
  }
  auto tensor_shp_ptr = seq_abs->elements()[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(tensor_shp_ptr);
  MS_LOG(DEBUG) << "tensor shape:" << tensor_shp_ptr->ToString() << " for abstract:" << abs->ToString();
  MS_EXCEPTION_IF_NULL(tensor_shp_ptr->cast<abstract::ShapePtr>());
  auto shape = tensor_shp_ptr->cast<abstract::ShapePtr>()->shape();
  (void)flatten_shp.insert(flatten_shp.end(), shape.begin(), shape.end());
  return flatten_shp;
}

ShapeVector KernelTensor::GetShapeVector() const {
  if (meta_type_ == kObjectTypeTensorType) {
    // Tensor
    auto base_shape_ptr = GetBaseShape();
    if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::Shape>()) {
      return {};
    }
    auto shape = base_shape_ptr->cast<abstract::ShapePtr>()->shape();
    return shape;
  } else if (meta_type_ == kObjectTypeTuple) {
    const TupleInfo &tuple_info = std::get<TupleInfo>(meta_);
    return GetSequenceFlattenShape(tuple_info.base_);
  } else if (meta_type_ == kObjectTypeList) {
    const ListInfo &list_info = std::get<ListInfo>(meta_);
    return GetSequenceFlattenShape(list_info.base_);
  } else {
    // Scalar
    return {};
  }
}

ShapeVector KernelTensor::GetMaxShape() const {
  if (meta_type_ != kObjectTypeTensorType) {
    return {};
  }
  auto base_shape_ptr = GetBaseShape();
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::Shape>()) {
    return {};
  }

  return base_shape_ptr->cast<abstract::ShapePtr>()->max_shape();
}

std::vector<TypeId> KernelTensor::GetListOrTupleDtype() const {
  const TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return {TypeId::kTypeUnknown};
  }

  auto type_ptr = info.base_->BuildType();
  if (type_ptr == nullptr || !type_ptr->isa<List>() || !type_ptr->isa<Tuple>()) {
    return {TypeId::kTypeUnknown};
  }

  std::vector<TypeId> types;
  if (type_ptr->isa<List>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else if (type_ptr->isa<List>()) {
    auto list_ptr = type_ptr->cast<ListPtr>();
    auto elements = list_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else {
    types.push_back(TypeId::kTypeUnknown);
  }

  return types;
}

ShapeArray KernelTensor::GetListOrTupleShapeVector() const {
  auto base_shape_ptr = GetBaseShape();
  // ListShape or TupleShape is inherited from SequenceShape.
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::SequenceShape>()) {
    return {};
  }
  auto sequence_shape_ptr = base_shape_ptr->cast<abstract::SequenceShapePtr>();
  auto base_shape_list = sequence_shape_ptr->shape();
  std::vector<std::vector<int64_t>> shape_vector_list;
  for (auto base_shape : base_shape_list) {
    if (base_shape == nullptr || !base_shape->isa<abstract::Shape>()) {
      return {};
    }
    auto tmp_shape = base_shape->cast<abstract::ShapePtr>()->shape();
    shape_vector_list.push_back(tmp_shape);
  }

  return shape_vector_list;
}

void KernelTensor::SetDtype(const TypePtr &dtype) {
  TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return;
  }
  info.base_->set_type(dtype);
}

void KernelTensor::SetShapeVector(const std::vector<int64_t> &shape) const {
  auto base_shape_ptr = GetBaseShape();
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::Shape>()) {
    return;
  }
  auto shape_ptr = base_shape_ptr->cast<abstract::ShapePtr>();
  if (shape_ptr) {
    shape_ptr->set_shape(shape);
  }
}

abstract::BaseShapePtr KernelTensor::GetBaseShape() const {
  if (meta_type_ != kObjectTypeTensorType) {
    return nullptr;
  }
  const TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return nullptr;
  }
  return info.base_->BuildShape();
}

void KernelTensor::SetBaseShape(const abstract::BaseShapePtr &base_shape) {
  TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return;
  }
  info.base_->set_shape(base_shape);
}

const std::vector<int64_t> &KernelTensor::GetDeviceShapeAdaptively() const {
  const TensorInfo &info = std::get<TensorInfo>(meta_);
  return info.device_shape_adaptively;
}

void KernelTensor::SetDeviceShapeAdaptively(const std::vector<int64_t> &device_shape_adaptively) {
  TensorInfo &info = std::get<TensorInfo>(meta_);
  info.device_shape_adaptively = device_shape_adaptively;
}

int KernelMod::Resize(const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  return Resize(this->op_, this->inputs_, this->outputs_, inputsOnHost);
}

int KernelMod::Resize(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  inputs_ = inputs;
  outputs_ = outputs;
  return Resize(this->op_, inputs, outputs, inputsOnHost);
}

int KernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                      const std::vector<KernelTensorPtr> &outputs,
                      const std::map<uint32_t, tensor::TensorPtr> & /* inputsOnHost */) {
  MS_LOG(DEBUG) << "Resize start for operator:" << base_operator->name();
  auto ret = KRET_OK;
  workspace_size_list_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  for (size_t idx = 0; idx < inputs.size(); idx++) {
    auto &input = inputs[idx];
    size_t tensor_size = 0;
    MS_EXCEPTION_IF_NULL(input);
    size_t type_size = GetTypeByte(TypeIdToType(input->GetDtype()));
    auto shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      MS_LOG(DEBUG) << "input " << idx << " is not valid shape:" << shape << " for op:" << kernel_name_;
      // early stop if any input shape contains -1/-2, which means input shape is dynamic
      return KRET_UNKNOWN_SHAPE;
    } else {
      tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
    }
    (void)input_size_list_.emplace_back(tensor_size);
  }

  for (size_t idx = 0; idx < outputs.size(); idx++) {
    auto &output = outputs[idx];
    size_t tensor_size = 0;
    MS_EXCEPTION_IF_NULL(output);
    size_t type_size = GetTypeByte(TypeIdToType(output->GetDtype()));
    auto shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
      // Note:
      // If output shape is unknown, the op is a compute-depended op and max_shape should not be empty,
      // and the output_size_list_ can be set by max_shape
      auto max_shape = output->GetMaxShape();
      if (max_shape.empty()) {
        MS_LOG(DEBUG) << "For " << kernel_name_ << ", the max_shape should not be empty when input shape is known.";
        ret = KRET_UNKNOWN_OUT_SHAPE;
      } else {
        tensor_size = SizeOf(max_shape) * type_size;
        ret = KRET_UNKNOWN_OUT_SHAPE;
      }
    } else {
      if (shape.empty()) {
        tensor_size = type_size;
      } else {
        auto cur_out_shape_num = SizeOf(shape);
        tensor_size = cur_out_shape_num * type_size;
        if (type_size != 0 && tensor_size / type_size != cur_out_shape_num) {
          MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", the shape of outputs[" << output_size_list_.size()
                                   << "]: " << shape
                                   << " is too big, mindspore cannot apply for such a large amount of memory.";
        }
      }
      tensor_size = std::max(tensor_size, type_size);
    }
    (void)output_size_list_.emplace_back(tensor_size);
  }
  MS_LOG(DEBUG) << "Resize end for operator:" << base_operator->name();
  return static_cast<int>(ret);
}

std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensorPtr> &tensors) {
  std::vector<std::vector<int64_t>> shapes(tensors.size());
  for (size_t idx = 0; idx < shapes.size(); idx++) {
    shapes[idx] = tensors[idx]->GetShapeVector();
  }
  return shapes;
}
}  // namespace kernel
}  // namespace mindspore
