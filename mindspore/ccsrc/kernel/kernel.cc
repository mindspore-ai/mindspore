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

#include <functional>
#include <algorithm>
#include <iterator>
#include <numeric>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
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
  flatten_shp.emplace_back(seq_abs->elements().size());
  if (seq_abs->elements().empty()) {
    return flatten_shp;
  }
  auto tensor_shp_ptr = seq_abs->elements()[0]->BuildShape();
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

void KernelTensor::SetShapeVector(const std::vector<int64_t> &shape) {
  TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return;
  }
  info.base_->set_shape(std::make_shared<abstract::Shape>(shape));
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

int KernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                      const std::vector<KernelTensorPtr> &outputs,
                      const std::map<uint32_t, tensor::TensorPtr> & /* inputsOnHost */) {
  auto ret = KRET_OK;
  this->inputs_ = inputs;
  this->outputs_ = outputs;
  workspace_size_list_.clear();
  input_size_list_.clear();
  input_shapes_.clear();
  for (auto &input : inputs) {
    size_t tensor_size = 0;
    size_t type_size = GetTypeByte(TypeIdToType(input->GetDtype()));
    MS_EXCEPTION_IF_NULL(input);
    auto shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      // early stop if any input shape contains -1/-2, which means input shape is dynamic
      return KRET_UNKNOWN_SHAPE;
    } else {
      tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
    }
    (void)input_size_list_.emplace_back(tensor_size);
    input_shapes_.emplace_back(shape);
  }
  output_shapes_.clear();
  output_size_list_.clear();
  for (auto &output : outputs) {
    size_t tensor_size = 0;
    size_t type_size = GetTypeByte(TypeIdToType(output->GetDtype()));
    MS_EXCEPTION_IF_NULL(output);
    auto shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
      // Note:
      // If output shape is unknown, the op is a compute-depended op and max_shape should not be empty,
      // and the output_size_list_ can be set by max_shape
      auto max_shape = output->GetMaxShape();
      if (max_shape.empty()) {
        auto primitive = base_operator->GetPrim();
        MS_ERROR_IF_NULL(primitive);
        MS_LOG(DEBUG) << "For " << primitive->name()
                      << ", the max_shape should not be empty when input shape is known.";
        ret = KRET_UNKNOWN_OUT_SHAPE;
      } else {
        tensor_size = SizeOf(max_shape) * type_size;
        ret = KRET_UNKNOWN_OUT_SHAPE;
      }
    } else {
      tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
    }
    (void)output_size_list_.emplace_back(tensor_size);
    output_shapes_.emplace_back(shape);
  }
  return static_cast<int>(ret);
}

bool KernelMod::Launch(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
                       const std::vector<AddressPtr> &workspace, void *stream_ptr) {
  return false;
}
// deprecated
bool KernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_CHECK_FAIL(this->inputs_.size() == inputs.size(), "inputs size check failed");
  MS_EXCEPTION_IF_CHECK_FAIL(this->outputs_.size() == outputs.size(), "inputs size check failed");
  auto it1 = this->inputs_.begin();
  auto it2 = inputs.begin();
  for (; it1 != this->inputs_.end() && it2 != inputs.end(); it1++, it2++) {
    (*it1)->SetData((*it2));
  }

  it1 = this->outputs_.begin();
  it2 = outputs.begin();
  for (; it1 != this->outputs_.end() && it2 != outputs.end(); it1++, it2++) {
    (*it1)->SetData((*it2));
  }

  return Launch(this->inputs_, this->outputs_, workspace, stream_ptr);
}

std::vector<int64_t> GetIntValueFromData(void *const data_c, const TypeId &type_id, size_t data_size,
                                         const size_t input_index, const std::string &kernel_name) {
  std::vector<int64_t> tensor_value;
  MS_EXCEPTION_IF_NULL(data_c);
  if (type_id == kNumberTypeInt32) {
    auto tensor_data = reinterpret_cast<int32_t *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(int32_t));
  } else if (type_id == kNumberTypeInt64) {
    auto tensor_data = reinterpret_cast<int64_t *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(int64_t));
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name << "', the " << input_index
                            << "th input must be a Tensor[Int64] or Tensor[Int32] type, but got "
                            << TypeIdLabel(type_id);
  }
  return tensor_value;
}

std::optional<std::vector<int64_t>> TryGetIntValueFromInputs(const std::vector<KernelTensorPtr> &inputs,
                                                             const size_t input_index, const std::string &kernel_name,
                                                             bool data_from_host) {
  if (inputs.size() <= input_index) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', inputs size is " << inputs.size() << ", but require " << input_index;
    return std::nullopt;
  }

  AddressPtr data{nullptr};
  if (data_from_host) {
    data = inputs[input_index]->GetHostData();
  } else {
    data = inputs[input_index]->GetData();
  }

  // The value of dynamic attr can only be obtained after the InferOp() is executed.
  if (data == nullptr || data->addr == nullptr) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', fail to find the " << input_index << "th input's data.";
    return std::nullopt;
  }

  const auto &data_format = inputs[input_index]->GetFormat();
  if (data_format != mindspore::Format::DEFAULT_FORMAT && data_format != mindspore::Format::NCHW) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "',  the format of the " << input_index
                      << "th input currently should be the default format and does not support " << data_format;
  }

  return GetIntValueFromData(data->addr, inputs[input_index]->GetDtype(), data->size, input_index, kernel_name);
}

bool TryGetIntValue(const CNodePtr &kernel_node, const size_t input_index, std::vector<int64_t> *attr_value,
                    bool data_from_host) {
  auto args = GetArgsFromCNode(kernel_node);
  if (args == nullptr) {
    return false;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto res = TryGetIntValueFromInputs(args->inputs, input_index, op_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  *attr_value = res.value();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
