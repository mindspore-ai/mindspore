/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/extendrt/graph_compiler/anfnode_tensor_adapter.h"
#include "src/extendrt/graph_compiler/compile_result_builder.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "utils/ms_utils_secure.h"

using ShapePtr = mindspore::abstract::ShapePtr;
using AbstractBasePtr = mindspore::abstract::AbstractBasePtr;
using AbstractTensorPtr = mindspore::abstract::AbstractTensorPtr;

namespace mindspore {
namespace infer {
Tensor *TensorAdapter::Convert2Tensor(const ParameterPtr &param_node, const std::string &tensor_name) {
  auto adapter = TensorAdapter::Create(param_node);
  if (adapter == nullptr) {
    MS_LOG(ERROR) << "Create tensor-adapter from parameter failed, parameter : " << param_node;
    return nullptr;
  }
  return adapter->ToTensor(tensor_name);
}

Tensor *TensorAdapter::Convert2Tensor(const ValueNodePtr &value_node, const std::string &tensor_name) {
  auto adapter = TensorAdapter::Create(value_node);
  if (adapter == nullptr) {
    MS_LOG(ERROR) << "Create tensor-adapter from value-node failed, value-node : " << value_node;
    return nullptr;
  }
  return adapter->ToTensor(tensor_name);
}

Tensor *TensorAdapter::Convert2Tensor(const AbstractTensorPtr &abstract, const std::string &tensor_name) {
  auto adapter = TensorAdapter::Create(abstract);
  if (adapter == nullptr) {
    MS_LOG(ERROR) << "Create tensor-adapter from abstracttensor failed, abstract : " << abstract;
    return nullptr;
  }
  return adapter->ToTensor(tensor_name);
}

Tensor *TensorAdapter::Convert2Tensor(const AbstractBasePtr &abstract, const std::string &tensor_name) {
  auto adapter = TensorAdapter::Create(abstract);
  if (adapter == nullptr) {
    MS_LOG(ERROR) << "Create tensor-adapter from abstractbase failed, abstract : " << abstract;
    return nullptr;
  }
  return adapter->ToTensor(tensor_name);
}

Tensor *TensorAdapter::ToTensor(const std::string &tensor_name) {
  std::vector<int32_t> int32_shape(shape_.size());
  for (size_t i = 0; i < shape_.size(); i++) {
    int32_shape[i] = static_cast<int32_t>(shape_[i]);
  }
  auto tensor = Tensor::CreateTensor(tensor_name, data_type_, int32_shape, data_, data_len_);
  if (tensor == nullptr) {
    return nullptr;
  }
  // move data to tensor
  tensor->set_own_data(own_data_);
  own_data_ = false;
  return tensor;
}

TensorAdapterPtr TensorAdapter::Create(const ParameterPtr &param_node) {
  if (param_node == nullptr) {
    MS_LOG(ERROR) << "Input parameter is nullptr.";
    return nullptr;
  }
  ShapeVector shape_vector;
  TypeId data_type = kTypeUnknown;
  auto status = GetDTAndShapeFromParameter(param_node, &data_type, &shape_vector);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Get data type and shape from param node failed.";
    return nullptr;
  }
  if (data_type == kObjectTypeString) {
    MS_LOG(ERROR) << "Not support kObjectTypeString type DefaultParam.";
    return nullptr;
  }
  auto adapter = std::make_shared<TensorAdapter>();
  adapter->data_type_ = data_type;
  adapter->shape_ = shape_vector;
  adapter->format_ = NHWC;
  adapter->is_const_ = param_node->has_default();
  if (!adapter->is_const_) {
    return adapter;
  }
  auto tensor_info = std::dynamic_pointer_cast<tensor::Tensor>(param_node->default_param());
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Cast default-param to tensor failed.";
    return nullptr;
  }
  adapter->compress_type_ = tensor_info->compression_type();
  adapter->data_ = tensor_info->data_c();
  adapter->data_len_ = tensor_info->Size();
  adapter->own_data_ = false;
  return adapter;
}

TensorAdapterPtr TensorAdapter::CreateFromTensorValueNode(const ValueNodePtr &value_node) {
  auto value_abstract = value_node->abstract();
  if (value_abstract == nullptr) {
    MS_LOG(ERROR) << "Abstract of value is nullptr";
    return nullptr;
  }
  auto adapter = TensorAdapter::Create(value_abstract);
  if (adapter == nullptr) {
    MS_LOG(ERROR) << "Create tensor adapter from abstract of valuenode failed, valuenode: "
                  << value_node->fullname_with_scope();
    return nullptr;
  }
  adapter->is_const_ = true;

  auto value = value_node->value();
  if (value == nullptr) {
    MS_LOG(ERROR) << "Value of value-node is nullptr, " << value_node->fullname_with_scope();
    return nullptr;
  }
  auto data = value->cast<tensor::TensorPtr>();
  if (data == nullptr) {
    MS_LOG(ERROR) << "Value of tensor-type value-node is not a Tensor, " << value_node->fullname_with_scope();
    return nullptr;
  }
  adapter->data_ = data->data_c();
  adapter->data_len_ = data->Size();
  adapter->own_data_ = false;
  return adapter;
}

TensorAdapterPtr TensorAdapter::CreateFromInt32ImmValue(const ValueNodePtr &value_node) {
  MS_ASSERT(value_node != nullptr);
  auto adapter = std::make_shared<TensorAdapter>();
  adapter->is_const_ = true;
  adapter->data_type_ = kNumberTypeInt32;
  adapter->shape_ = {1};
  auto value = value_node->value();
  if (value == nullptr) {
    MS_LOG(ERROR) << "Value of value-node is nullptr, " << value_node->fullname_with_scope();
    return nullptr;
  }
  auto data = GetValue<int32_t>(value);
  adapter->data_ = &data;
  adapter->data_len_ = sizeof(int32_t);
  adapter->own_data_ = false;
  return adapter;
}

TensorAdapterPtr TensorAdapter::CreateFromInt64ImmValue(const ValueNodePtr &value_node) {
  MS_ASSERT(value_node != nullptr);
  auto adapter = std::make_shared<TensorAdapter>();
  adapter->is_const_ = true;
  adapter->data_type_ = kNumberTypeInt64;
  adapter->shape_ = {1};
  auto value = value_node->value();
  if (value == nullptr) {
    MS_LOG(ERROR) << "Value of value-node is nullptr, " << value_node->fullname_with_scope();
    return nullptr;
  }
  auto data = GetValue<int64_t>(value);
  adapter->data_ = &data;
  adapter->data_len_ = sizeof(int64_t);
  adapter->own_data_ = false;
  return adapter;
}

TensorAdapterPtr TensorAdapter::CreateFromBoolImmValue(const ValueNodePtr &value_node) {
  MS_ASSERT(value_node != nullptr);
  auto adapter = std::make_shared<TensorAdapter>();
  adapter->is_const_ = true;
  adapter->data_type_ = kNumberTypeBool;
  adapter->shape_ = {1};
  auto value = value_node->value();
  if (value == nullptr) {
    MS_LOG(ERROR) << "Value of value-node is nullptr, " << value_node->fullname_with_scope();
    return nullptr;
  }
  auto data = value->cast<mindspore::BoolImmPtr>();
  if (data == nullptr) {
    MS_LOG(ERROR) << "BoolImm Value of cast to BoolImmPtr failed, " << value_node->fullname_with_scope();
    return nullptr;
  }
  auto data_value = data->value();
  adapter->data_ = &data_value;
  adapter->data_len_ = sizeof(bool);
  adapter->own_data_ = false;
  return adapter;
}

TensorAdapterPtr TensorAdapter::CreateFromNumberTypeValue(const ValueNodePtr &value_node) {
  MS_ASSERT(value_node != nullptr);
  auto adapter = std::make_shared<TensorAdapter>();
  adapter->is_const_ = true;
  adapter->data_type_ = kNumberTypeInt32;
  adapter->shape_ = {1};
  auto data = utils::cast<NumberPtr>(value_node->value());
  if (data == nullptr) {
    MS_LOG(ERROR) << "Value of Number type value-node is not a NumberPtr, " << value_node->fullname_with_scope();
    return nullptr;
  }
  TypeId number_type = data->number_type();
  static const std::unordered_map<TypeId, TypeId> TypeToTypeMap = {
    {kNumberTypeInt, kNumberTypeInt32}, {kNumberTypeUInt, kNumberTypeUInt32}, {kNumberTypeFloat, kNumberTypeFloat32}};
  if (TypeToTypeMap.find(number_type) != TypeToTypeMap.end()) {
    number_type = TypeToTypeMap.at(number_type);
  }
  auto number_data = static_cast<int32_t>(number_type);
  adapter->data_ = &number_data;
  adapter->data_len_ = sizeof(int32_t);
  adapter->own_data_ = false;
  return adapter;
}

TensorAdapterPtr TensorAdapter::CreateFromIntSequenceValue(const ValueNodePtr &value_node) {
  MS_ASSERT(value_node != nullptr);
  auto value_seq = utils::cast<ValueSequencePtr>(value_node->value());
  if (value_seq == nullptr) {
    MS_LOG(ERROR) << "Value of Sequence type value-node is not a ValueSequencePtr, "
                  << value_node->fullname_with_scope();
    return nullptr;
  }
  auto adapter = std::make_shared<TensorAdapter>();
  adapter->is_const_ = true;
  if (!value_seq->value().empty()) {
    if (value_seq->value().front()->type()->number_type() == kNumberTypeInt32 ||
        value_seq->value().front()->type()->number_type() == kNumberTypeInt) {
      adapter->data_type_ = kNumberTypeInt32;
      auto data = GetValue<std::vector<int32_t>>(value_seq);
      adapter->shape_ = {static_cast<int64_t>(data.size())};
      adapter->data_ = data.data();
      adapter->data_len_ = data.size() * sizeof(int32_t);
      adapter->own_data_ = false;
    } else if (value_seq->value().front()->type()->number_type() == kNumberTypeInt64) {
      adapter->data_type_ = kNumberTypeInt64;
      auto data = GetValue<std::vector<int64_t>>(value_seq);
      adapter->shape_ = {static_cast<int64_t>(data.size())};
      adapter->data_ = data.data();
      adapter->data_len_ = data.size() * sizeof(int64_t);
      adapter->own_data_ = false;
    } else {
      MS_LOG(ERROR) << "only support integer value ValueSequence.";
      return nullptr;
    }
  }
  return adapter;
}

TensorAdapterPtr TensorAdapter::Create(const ValueNodePtr &value_node) {
  MS_ASSERT(value_node != nullptr);
  auto value = value_node->value();
  if (value->isa<tensor::Tensor>()) {
    return CreateFromTensorValueNode(value_node);
  } else if (value->isa<mindspore::Int32Imm>()) {
    return CreateFromInt32ImmValue(value_node);
  } else if (value->isa<mindspore::Int64Imm>()) {
    return CreateFromInt64ImmValue(value_node);
  } else if (value->isa<mindspore::BoolImm>()) {
    return CreateFromBoolImmValue(value_node);
  } else if (value->isa<mindspore::ValueSequence>()) {
    return CreateFromIntSequenceValue(value_node);
  } else if (value->isa<Number>()) {
    return CreateFromNumberTypeValue(value_node);
  } else {
    MS_LOG(ERROR) << "Not support value type: " << value->type();
    return nullptr;
  }
}

TensorAdapterPtr TensorAdapter::Create(const AbstractBasePtr &abs) {
  auto abs_tensor = utils::cast<AbstractTensorPtr>(abs);
  if (abs_tensor == nullptr) {
    MS_LOG(ERROR) << "Input abstract is not a AbstractTensor.";
    return nullptr;
  }
  return TensorAdapter::Create(abs_tensor);
}

TensorAdapterPtr TensorAdapter::Create(const AbstractTensorPtr &abs_tensor) {
  if (abs_tensor == nullptr) {
    MS_LOG(ERROR) << "Input abstract is not a AbstractTensor.";
    return nullptr;
  }
  ShapeVector shape_vector;
  TypeId data_type = kTypeUnknown;
  auto ret = GetDTAndShapeFromAbTensor(abs_tensor, &data_type, &shape_vector);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get data type and shape from value node failed.";
    return nullptr;
  }
  auto adapter = std::make_shared<TensorAdapter>();
  adapter->data_type_ = data_type;
  adapter->shape_ = shape_vector;
  return adapter;
}

StatusCode TensorAdapter::GetDTAndShapeFromAbTensor(const AbstractTensorPtr &abstract, TypeId *data_type,
                                                    ShapeVector *shape_vector) {
  MS_ASSERT(abstract != nullptr && data_type != nullptr && shape_vector != nullptr);
  if (abstract->element() == nullptr) {
    MS_LOG(ERROR) << "`element` of abstract is nullptr";
    return kLiteError;
  }
  auto type_ptr = abstract->element()->GetTypeTrack();
  if (type_ptr == nullptr) {
    MS_LOG(ERROR) << "Type of abstract is nullptr";
    return kLiteError;
  }
  *data_type = type_ptr->type_id();
  if (!utils::isa<ShapePtr>(abstract->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr";
    return kLiteError;
  }
  *shape_vector = utils::cast<ShapePtr>(abstract->BuildShape())->shape();
  return kSuccess;
}

StatusCode TensorAdapter::GetDTAndShapeFromParameter(const ParameterPtr &param_node, TypeId *data_type,
                                                     ShapeVector *shape_vector) {
  MS_ASSERT(param_node != nullptr && data_type != nullptr && shape_vector != nullptr);
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
    return kLiteError;
  }
  auto abstract_tensor = utils::cast<AbstractTensorPtr>(abstract_base);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, " << param_node->name();
    return kLiteError;
  }
  return GetDTAndShapeFromAbTensor(abstract_tensor, data_type, shape_vector);
}
}  // namespace infer
}  // namespace mindspore
