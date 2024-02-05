/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/dvm/dvm_kernel_mod.h"
#include <algorithm>

namespace mindspore {
namespace kernel {
BaseShapePtr DvmInfer::InferShape(const AbstractBasePtrList &args) { return kernel_->InferShape(args); }

DvmKernelMod::DvmKernelMod(bool is_dynamic) {
  auto kernel_type = is_dynamic ? dvm::KernelType::kDynShape : dvm::KernelType::kStaticShape;
  kernel_.Reset(kernel_type);
}

bool DvmKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  for (size_t i = 0; i < inputs_addr_.size(); ++i) {
    inputs_addr_[i] = inputs[inputs_idx_[i]]->device_ptr();
  }
  for (size_t i = 0; i < outputs_addr_.size(); ++i) {
    outputs_addr_[i] = outputs[outputs_idx_[i]]->device_ptr();
  }
  auto ret = kernel_.Launch(reloc_table_, inputs_addr_.data(), outputs_addr_.data(), stream_ptr);
  return ret == 0;
}

void DvmKernelMod::Initialize(const std::vector<TypeId> &inputs_type, const std::vector<TypeId> &outputs_type) {
  inputs_type_byte_.clear();
  inputs_type_byte_.reserve(inputs_type.size());
  (void)std::transform(inputs_type.begin(), inputs_type.end(), std::back_inserter(inputs_type_byte_),
                       [](const TypeId &type_id) { return GetTypeByte(TypeIdToType(type_id)); });
  outputs_type_byte_.clear();
  outputs_type_byte_.reserve(outputs_type.size());
  (void)std::transform(outputs_type.begin(), outputs_type.end(), std::back_inserter(outputs_type_byte_),
                       [](const TypeId &type_id) { return GetTypeByte(TypeIdToType(type_id)); });
  input_size_list_.resize(inputs_type.size(), 1);
  output_size_list_.resize(outputs_type.size(), 1);
  inputs_shape_.resize(inputs_type.size());
  outputs_shape_.resize(outputs_type.size());
  shapes_ref_source_.reserve(inputs_type.size());
  inputs_shape_ref_.resize(inputs_type.size());
  inputs_.reserve(inputs_type.size());
  outputs_.reserve(outputs_type.size());
  inputs_idx_.reserve(inputs_type.size());
  outputs_idx_.reserve(outputs_type.size());
}

void DvmKernelMod::CodeGen(const std::vector<ShapeVector> &inputs_shape,
                           const std::vector<ShapeVector> &outputs_shape) {
  for (size_t i = 0; i < inputs_shape.size(); ++i) {
    input_size_list_[i] = inputs_type_byte_[i];
    for (auto sh : inputs_shape[i]) {
      input_size_list_[i] *= LongToSize(sh);
    }
  }
  for (size_t i = 0; i < outputs_shape.size(); ++i) {
    output_size_list_[i] = outputs_type_byte_[i];
    for (auto sh : outputs_shape[i]) {
      output_size_list_[i] *= LongToSize(sh);
    }
  }
  kernel_.CodeGen();
}

BaseShapePtr DvmKernelMod::InferShape(const AbstractBasePtrList &inputs_abs) {
  BaseShapePtr result{nullptr};
  // update input shape
  for (size_t i = 0; i < inputs_abs.size(); ++i) {
    // to do: if input value is needed in infer shape, then we should fetch input value here
    inputs_shape_[i] = inputs_abs[i]->GetShape()->GetShapeVector();
    if (inputs_shape_ref_[i] != nullptr) {
      *inputs_shape_ref_[i] = inputs_shape_[i];
    }
    input_size_list_[i] = inputs_type_byte_[i];
    for (auto sh : inputs_shape_[i]) {
      input_size_list_[i] *= LongToSize(sh);
    }
  }
  // re-codegen by new input shape
  kernel_.CodeGen();
  // update output shape
  for (size_t i = 0; i < outputs_.size(); ++i) {
    auto idx = outputs_idx_[i];
    auto shape_ref = kernel_.GetShape(outputs_[i]);
    outputs_shape_[idx] = ShapeVector(shape_ref->data, shape_ref->data + shape_ref->size);
    output_size_list_[idx] = outputs_type_byte_[idx];
    for (auto sh : outputs_shape_[idx]) {
      output_size_list_[idx] *= LongToSize(sh);
    }
  }
  // update output abstract
  if (outputs_shape_.size() > 1) {
    abstract::BaseShapePtrList out_shapes(outputs_shape_.size());
    for (size_t i = 0; i < outputs_shape_.size(); ++i) {
      out_shapes[i] = std::make_shared<abstract::TensorShape>(outputs_shape_[i]);
    }
    result = std::make_shared<abstract::TupleShape>(out_shapes);
  } else {
    result = std::make_shared<abstract::TensorShape>(outputs_shape_.front());
  }
  return result;
}

void DvmKernelMod::CacheLoad(dvm::NDObject *obj, size_t idx) {
  inputs_.push_back(obj);
  inputs_idx_.push_back(idx);
}

void DvmKernelMod::CacheStore(dvm::NDObject *obj, size_t idx) {
  outputs_.push_back(obj);
  outputs_idx_.push_back(idx);
}

void DvmKernelMod::UpdateIO() {
  inputs_addr_.resize(inputs_.size());
  outputs_addr_.resize(outputs_.size());
  reloc_table_.inputs = inputs_.data();
  reloc_table_.outputs = outputs_.data();
  reloc_table_.inputs_size = inputs_.size();
  reloc_table_.outputs_size = outputs_.size();
}

void DvmKernelMod::UpdateInputShapeRef(size_t input_idx, dvm::ShapeRef *ref) { inputs_shape_ref_[input_idx] = ref; }
}  // namespace kernel
}  // namespace mindspore
