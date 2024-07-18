/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "include/common/utils/convert_utils.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/common/utils/utils.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindspore/core/ops/sparse_ops.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "utils/hashing.h"

namespace mindspore {
bool ValueToBool(const ValuePtr &v, bool *value) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<BoolImm>()) {
    *value = v->cast<BoolImmPtr>()->value();
  } else if (v->isa<Int32Imm>()) {
    *value = v->cast<Int32ImmPtr>()->value() != 0;
  } else if (v->isa<UInt32Imm>()) {
    *value = v->cast<UInt32ImmPtr>()->value() != 0;
  } else if (v->isa<FP32Imm>()) {
    *value = fabs(v->cast<FP32ImmPtr>()->value()) > FLT_EPSILON;
  } else if (v->isa<FP64Imm>()) {
    *value = fabs(v->cast<FP64ImmPtr>()->value()) > DBL_EPSILON;
  } else if (v->isa<StringImm>()) {
    std::string str = v->cast<StringImmPtr>()->value();
    *value = str.length() != 0;
  } else if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->data_sync();
    bool *tensor_data = static_cast<bool *>(tensor->data_c());
    // maybe need to support if tensor is a bool array
    auto vb = tensor_data[0];
    *value = vb;
  } else {
    MS_LOG(WARNING) << "value is not supported to cast to be bool";
    return false;
  }
  return true;
}

bool BaseRefToInt(const ValuePtr &v, int64_t *value) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    tensor->data_sync();
    if (tensor->Dtype()->ToString() == "Int32") {
      auto *tensor_data = static_cast<int32_t *>(tensor->data_c());
      auto vb = tensor_data[0];
      *value = static_cast<int64_t>(vb);
    } else if (tensor->Dtype()->ToString() == "Int64") {
      auto *tensor_data = static_cast<int64_t *>(tensor->data_c());
      auto vb = tensor_data[0];
      *value = vb;
    } else {
      MS_LOG(ERROR) << "Index must be Int type.";
    }
    return true;
  }
  MS_LOG(ERROR) << "Index must be tensor type.";
  return false;
}

bool BaseRefToBool(const BaseRef &v, bool *value) {
  if (utils::isa<ValuePtr>(v)) {
    return ValueToBool(utils::cast<ValuePtr>(v), value);
  } else if (utils::isa<bool>(v)) {
    auto vb = utils::cast<bool>(v);
    *value = vb;
  } else if (utils::isa<int>(v)) {
    auto vb = utils::cast<int>(v);
    *value = vb != 0;
  } else if (utils::isa<unsigned int>(v)) {
    auto vb = utils::cast<unsigned int>(v);
    *value = vb != 0;
  } else if (utils::isa<float>(v)) {
    auto vb = utils::cast<float>(v);
    *value = !(vb >= -FLT_EPSILON && vb <= FLT_EPSILON);
  } else if (utils::isa<double>(v)) {
    auto vb = utils::cast<double>(v);
    *value = !(vb >= -DBL_EPSILON && vb <= DBL_EPSILON);
  } else {
    MS_LOG(DEBUG) << "value is not supported to cast to be bool";
    return false;
  }
  return true;
}

namespace {
// Isomorphism
bool SameNode(const AnfNodePtr &node1, const AnfNodePtr &node2, FuncGraphPairMapEquiv *equiv_func_graph,
              NodeMapEquiv *const equiv_node);

bool SameValueNode(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  auto a1 = GetValueNode(node1);
  auto a2 = GetValueNode(node2);
  if (a1->isa<Primitive>() && a2->isa<Primitive>()) {
    return a1->cast<PrimitivePtr>()->name() == a2->cast<PrimitivePtr>()->name();
  } else if (a1->isa<tensor::Tensor>() && a2->isa<tensor::Tensor>()) {
    return a1->cast<tensor::TensorPtr>()->ValueEqual(*(a2->cast<tensor::TensorPtr>()));
  }
  return *a1 == *a2;
}

bool SameNodeShallow(const AnfNodePtr &node1, const AnfNodePtr &node2, FuncGraphPairMapEquiv *equiv_func_graph,
                     NodeMapEquiv *const equiv_node) {
  if (equiv_node == nullptr) {
    MS_LOG(ERROR) << "Invalid equiv_node";
    return false;
  }
  if (equiv_node->count(node1) > 0 && (*equiv_node)[node1] == node2) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node1) && IsValueNode<FuncGraph>(node2)) {
    return Isomorphic(GetValueNode<FuncGraphPtr>(node1), GetValueNode<FuncGraphPtr>(node2), equiv_func_graph,
                      equiv_node);
  }
  if (node1->isa<ValueNode>() && node2->isa<ValueNode>()) {
    return SameValueNode(node1, node2);
  }
  if (node1->isa<Parameter>() && node2->isa<Parameter>()) {
    auto para1 = node1->cast<ParameterPtr>();
    auto para2 = node2->cast<ParameterPtr>();
    if (para1->name() == para2->name()) {
      return true;
    }
    MS_LOG(DEBUG) << "two parameters are not equal.";
    return false;
  }
  if (AnfUtils::IsCustomActorNode(node1) && AnfUtils::IsCustomActorNode(node2)) {
    return AnfUtils::IsCutomActorNodeSame(node1, node2);
  }
  if (node1->isa<CNode>() && node2->isa<CNode>()) {
    return SameNode(node1, node2, equiv_func_graph, equiv_node);
  }
  MS_LOG(ERROR) << "type error";
  return false;
}

bool SameNode(const AnfNodePtr &node1, const AnfNodePtr &node2, FuncGraphPairMapEquiv *equiv_func_graph,
              NodeMapEquiv *const equiv_node) {
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  if (node1->isa<CNode>() && node2->isa<CNode>()) {
    auto &inputs1 = node1->cast<CNodePtr>()->inputs();
    auto &inputs2 = node2->cast<CNodePtr>()->inputs();
    for (std::size_t i = 0; i < inputs1.size(); ++i) {
      if (!SameNodeShallow(inputs1[i], inputs2[i], equiv_func_graph, equiv_node)) {
        return false;
      }
    }
    return true;
  }
  return SameNodeShallow(node1, node2, equiv_func_graph, equiv_node);
}

bool SameSubgraph(const AnfNodePtr &root1, const AnfNodePtr &root2, FuncGraphPairMapEquiv *equiv_func_graph,
                  NodeMapEquiv *const equiv_node) {
  mindspore::HashSet<AnfNodePtr> done;
  std::stack<std::pair<AnfNodePtr, AnfNodePtr>> todo;

  todo.push(std::make_pair(root1, root2));
  while (!todo.empty()) {
    AnfNodePtr node1 = todo.top().first;
    if (done.count(node1) > 0) {
      todo.pop();
      continue;
    }
    AnfNodePtr node2 = todo.top().second;

    bool condition = false;
    const auto &s1 = GetInputs(node1);
    const auto &s2 = GetInputs(node2);

    if (s1.size() != s2.size()) {
      return false;
    }
    for (std::size_t i = 0; i < s1.size(); ++i) {
      if (done.count(s1[i]) == 0) {
        todo.push(std::make_pair(s1[i], s2[i]));
        condition = true;
      }
    }
    if (condition) {
      continue;
    }
    (void)done.insert(node1);

    auto res = SameNode(node1, node2, equiv_func_graph, equiv_node);
    if (res) {
      (*equiv_node)[node1] = node2;
    } else {
      return false;
    }
    todo.pop();
  }
  return true;
}
}  // namespace

bool Isomorphic(const FuncGraphPtr &fg1, const FuncGraphPtr &fg2, FuncGraphPairMapEquiv *equiv_func_graph,
                NodeMapEquiv *const equiv_node) {
  auto fg1_fg2 = std::make_pair(fg1, fg2);
  if (equiv_func_graph == nullptr) {
    MS_LOG(ERROR) << "equiv_func_graph not init";
    return false;
  }
  if (equiv_func_graph->find(fg1_fg2) != equiv_func_graph->end()) {
    return (*equiv_func_graph)[fg1_fg2] != kNotEquiv;
  }
  if (fg1 == nullptr || fg2 == nullptr) {
    MS_LOG(ERROR) << "Invalid function graph";
    return false;
  }
  if (fg1->parameters().size() != fg2->parameters().size()) {
    MS_LOG(DEBUG) << "parameters size not match";
    return false;
  }
  if (equiv_node != nullptr) {
    for (std::size_t i = 0; i < fg1->parameters().size(); ++i) {
      (*equiv_node)[fg1->parameters()[i]] = fg2->parameters()[i];
    }
    (*equiv_func_graph)[fg1_fg2] = kPending;
    auto result = SameSubgraph(fg1->get_return(), fg2->get_return(), equiv_func_graph, equiv_node);
    (*equiv_func_graph)[fg1_fg2] = EquivState(result);
    return result;
  }

  MS_LOG(ERROR) << "equiv_node not init";
  return false;
}

tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    MS_EXCEPTION(ArgumentError) << "Nullptr Error!";
  }
  TypePtr data_type = scalar->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeBool:
      return std::make_shared<tensor::Tensor>(GetValue<bool>(scalar), data_type);
    case kNumberTypeInt8:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int8_t>(scalar)), data_type);
    case kNumberTypeInt16:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int16_t>(scalar)), data_type);
    case kNumberTypeInt32:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int32_t>(scalar)), data_type);
    case kNumberTypeInt64:
      return std::make_shared<tensor::Tensor>(GetValue<int64_t>(scalar), data_type);
    case kNumberTypeUInt8:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint8_t>(scalar)), data_type);
    case kNumberTypeUInt16:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint16_t>(scalar)), data_type);
    case kNumberTypeUInt32:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(GetValue<uint32_t>(scalar)), data_type);
    case kNumberTypeUInt64:
      return std::make_shared<tensor::Tensor>(GetValue<uint64_t>(scalar), data_type);
    case kNumberTypeFloat32:
      return std::make_shared<tensor::Tensor>(GetValue<float>(scalar), data_type);
    case kNumberTypeFloat64:
      return std::make_shared<tensor::Tensor>(GetValue<double>(scalar), data_type);
    default:
      MS_LOG(EXCEPTION) << "When convert scalar to tensor, the scalar type: " << data_type << " is invalid.";
  }
}

template <typename T>
std::vector<T> ConvertValueListToVector(const ValuePtrList &seq_values) {
  size_t element_num = seq_values.size();
  std::vector<T> array_data(element_num);
  for (size_t i = 0; i < element_num; i++) {
    const auto &element = seq_values[i];
    MS_EXCEPTION_IF_NULL(element);
    array_data[i] = GetValue<T>(element);
  }
  return array_data;
}

tensor::TensorPtr SequenceToTensor(const ValueSequencePtr &sequence) {
  MS_EXCEPTION_IF_NULL(sequence);
  const auto &element_values = sequence->value();
  if (element_values.empty()) {
    std::vector<int32_t> array_data;
    MS_LOG(WARNING) << "The value sequence is empty.";
    return std::make_shared<tensor::Tensor>(std::move(array_data), TypeIdToType(kNumberTypeInt32));
  }

  const auto &first_element = element_values[0];
  if (!first_element->isa<Scalar>()) {
    MS_LOG(EXCEPTION) << "For sequence value, only sequence of scalar can convert to TensorValue, but got: "
                      << sequence->ToString();
  }

  TypePtr data_type = first_element->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeInt32:
      return std::make_shared<tensor::Tensor>(ConvertValueListToVector<int32_t>(element_values), data_type);
    case kNumberTypeInt64:
      return std::make_shared<tensor::Tensor>(ConvertValueListToVector<int64_t>(element_values), data_type);
    case kNumberTypeFloat64:
      return std::make_shared<tensor::Tensor>(ConvertValueListToVector<double>(element_values), data_type);
    default:
      MS_LOG(EXCEPTION) << "When convert sequence to tensor, the sequence type: " << data_type << " is invalid.";
  }
}

namespace {
KernelTensorValuePtr ConvertScalarToKernelTensorValue(const ValuePtr &scalar) {
  MS_EXCEPTION_IF_NULL(scalar);
  TypePtr data_type = scalar->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeBool:
      return std::make_shared<KernelTensorValue>(GetValue<bool>(scalar), data_type);
    case kNumberTypeInt8:
      return std::make_shared<KernelTensorValue>(GetValue<int8_t>(scalar), data_type);
    case kNumberTypeInt16:
      return std::make_shared<KernelTensorValue>(GetValue<int16_t>(scalar), data_type);
    case kNumberTypeInt32:
      return std::make_shared<KernelTensorValue>(GetValue<int32_t>(scalar), data_type);
    case kNumberTypeInt64:
      return std::make_shared<KernelTensorValue>(GetValue<int64_t>(scalar), data_type);
    case kNumberTypeUInt8:
      return std::make_shared<KernelTensorValue>(GetValue<uint8_t>(scalar), data_type);
    case kNumberTypeUInt16:
      return std::make_shared<KernelTensorValue>(GetValue<uint16_t>(scalar), data_type);
    case kNumberTypeUInt32:
      return std::make_shared<KernelTensorValue>(GetValue<uint32_t>(scalar), data_type);
    case kNumberTypeUInt64:
      return std::make_shared<KernelTensorValue>(GetValue<uint64_t>(scalar), data_type);
    case kNumberTypeFloat32:
      return std::make_shared<KernelTensorValue>(GetValue<float>(scalar), data_type);
    case kNumberTypeFloat64:
      return std::make_shared<KernelTensorValue>(GetValue<double>(scalar), data_type);
    default:
      MS_LOG(EXCEPTION) << "When convert scalar to KernelTensorValue, the scalar type: " << data_type->ToString()
                        << " is invalid.";
  }
}

template <typename T>
KernelTensorValuePtr ConvertValueListToKernelTensorValue(const ValuePtrList &seq_values, const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  size_t element_num = seq_values.size();
  std::vector<uint8_t> array_data(element_num * sizeof(T));
  T *array_data_ptr = reinterpret_cast<T *>(array_data.data());
  MS_EXCEPTION_IF_NULL(array_data_ptr);

  for (size_t i = 0; i < element_num; i++) {
    const auto &element = seq_values[i];
    MS_EXCEPTION_IF_NULL(element);
    array_data_ptr[i] = GetValue<T>(element);
  }
  return std::make_shared<KernelTensorValue>(std::move(array_data), type);
}

KernelTensorValuePtr ConvertSequenceToKernelTensorValue(const ValueSequencePtr &value_seq) {
  MS_EXCEPTION_IF_NULL(value_seq);
  const auto &element_values = value_seq->value();
  std::vector<uint8_t> array_data;
  if (element_values.empty()) {
    MS_LOG(INFO) << "The value sequence is empty.";
    return std::make_shared<KernelTensorValue>(std::move(array_data), value_seq->type());
  }

  const auto &first_element = element_values[0];
  if (!first_element->isa<Scalar>()) {
    MS_LOG(EXCEPTION) << "For sequence value, only sequence of scalar can convert to KernelTensorValue, but got: "
                      << value_seq->ToString();
  }

  TypePtr data_type = first_element->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();

  switch (type_id) {
    case kNumberTypeBool:
      return ConvertValueListToKernelTensorValue<bool>(element_values, value_seq->type());
    case kNumberTypeInt8:
      return ConvertValueListToKernelTensorValue<int8_t>(element_values, value_seq->type());
    case kNumberTypeInt16:
      return ConvertValueListToKernelTensorValue<int16_t>(element_values, value_seq->type());
    case kNumberTypeInt32:
      return ConvertValueListToKernelTensorValue<int32_t>(element_values, value_seq->type());
    case kNumberTypeInt64:
      return ConvertValueListToKernelTensorValue<int64_t>(element_values, value_seq->type());
    case kNumberTypeUInt8:
      return ConvertValueListToKernelTensorValue<uint8_t>(element_values, value_seq->type());
    case kNumberTypeUInt16:
      return ConvertValueListToKernelTensorValue<uint16_t>(element_values, value_seq->type());
    case kNumberTypeUInt32:
      return ConvertValueListToKernelTensorValue<uint32_t>(element_values, value_seq->type());
    case kNumberTypeUInt64:
      return ConvertValueListToKernelTensorValue<uint64_t>(element_values, value_seq->type());
    case kNumberTypeFloat32:
      return ConvertValueListToKernelTensorValue<float>(element_values, value_seq->type());
    case kNumberTypeFloat64:
      return ConvertValueListToKernelTensorValue<double>(element_values, value_seq->type());
    default:
      MS_LOG(EXCEPTION) << "When convert sequence to KernelTensorValue, the element type: " << data_type->ToString()
                        << " is invalid.";
  }
}
}  // namespace

KernelTensorValuePtr ConvertValueToKernelTensorValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<Scalar>()) {
    return ConvertScalarToKernelTensorValue(value);
  } else if (value->isa<ValueSequence>()) {
    auto value_seq = value->cast<ValueSequencePtr>();
    return ConvertSequenceToKernelTensorValue(value_seq);
  } else if (value->isa<tensor::BaseTensor>()) {
    auto tensor_ptr = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    return std::make_shared<KernelTensorValue>(tensor_ptr->data_ptr(), tensor_ptr->type());
  } else if (value->isa<StringImm>()) {
    auto string_ptr = value->cast<StringImmPtr>();
    MS_EXCEPTION_IF_NULL(string_ptr);
    return std::make_shared<KernelTensorValue>(string_ptr, string_ptr->type());
  } else if (value->isa<Type>()) {
    return nullptr;
  } else {
    MS_LOG(WARNING) << "KernelTensorValue not support the value type: " << value->ToString();
    return nullptr;
  }
}

template <typename T, typename Scalar>
ValuePtr GetTensorValue(const tensor::TensorPtr &tensor) {
  ValuePtr ret;
  auto tensor_value = TensorValueToVector<T>(tensor);
  if (tensor_value.size() == 1) {
    ret = std::make_shared<Scalar>(tensor_value[0]);
  } else {
    std::vector<ValuePtr> value_vec;
    for (const auto &elem : tensor_value) {
      auto value = std::make_shared<Scalar>(elem);
      MS_EXCEPTION_IF_NULL(value);
      value_vec.push_back(value);
    }
    ret = std::make_shared<ValueTuple>(value_vec);
  }
  return ret;
}

ValuePtr CreateValueFromTensor(const tensor::TensorPtr &tensor) {
  ValuePtr ret;
  if (tensor->has_user_data(kTensorValueIsType)) {
    ret = tensor->user_data<mindspore::Type>(kTensorValueIsType);
    return ret;
  }

  if (tensor->has_user_data(kTensorValueIsEmpty)) {
    ret = tensor->user_data<mindspore::Value>(kTensorValueIsEmpty);
    return ret;
  }

  TypePtr data_type = tensor->Dtype();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeBool: {
      ret = GetTensorValue<bool, BoolImm>(tensor);
      break;
    }

    case kNumberTypeInt8: {
      ret = GetTensorValue<int8_t, Int8Imm>(tensor);
      break;
    }

    case kNumberTypeUInt8: {
      ret = GetTensorValue<uint8_t, UInt8Imm>(tensor);
      break;
    }

    case kNumberTypeInt16: {
      ret = GetTensorValue<int16_t, Int16Imm>(tensor);
      break;
    }

    case kNumberTypeUInt16: {
      ret = GetTensorValue<uint16_t, UInt16Imm>(tensor);
      break;
    }

    case kNumberTypeInt32: {
      ret = GetTensorValue<int32_t, Int32Imm>(tensor);
      break;
    }

    case kNumberTypeUInt32: {
      ret = GetTensorValue<uint32_t, UInt32Imm>(tensor);
      break;
    }

    case kNumberTypeInt64: {
      ret = GetTensorValue<int64_t, Int64Imm>(tensor);
      break;
    }

    case kNumberTypeUInt64: {
      ret = GetTensorValue<uint64_t, UInt64Imm>(tensor);
      break;
    }

    case kNumberTypeFloat32: {
      ret = GetTensorValue<float, FP32Imm>(tensor);
      break;
    }

    case kNumberTypeFloat64: {
      ret = GetTensorValue<double, FP64Imm>(tensor);
      break;
    }

    default:
      MS_LOG(EXCEPTION) << "Can't parse attr value :" << tensor->ToString() << ", Type:" << tensor->type_name();
  }
  return ret;
}

void TensorValueToTensor(const ValuePtr &value, std::vector<tensor::BaseTensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(tensors);
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensors->emplace_back(tensor);
  } else if (value->isa<Scalar>()) {
    auto tensor = ScalarToTensor(value->cast<ScalarPtr>());
    MS_EXCEPTION_IF_NULL(tensor);
    tensors->emplace_back(tensor);
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);
    for (const auto &v : value_seq->value()) {
      TensorValueToTensor(v, tensors);
    }
  }
}

size_t CountValueNum(const ValueSequencePtr &value_sequence) {
  MS_EXCEPTION_IF_NULL(value_sequence);
  size_t cnt = 0;
  const auto &value_list = value_sequence->value();
  for (const auto &value : value_list) {
    if (value->isa<ValueSequence>()) {
      cnt += CountValueNum(value->cast<ValueSequencePtr>());
    } else {
      cnt++;
    }
  }
  return cnt;
}

bool IsAKGSparseOP(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const PrimitiveSet prims{prim::kPrimCSRReduceSum, prim::kPrimCSRMul,  prim::kPrimCSRMV,  prim::kPrimCSRGather,
                           prim::kPrimCSR2COO,      prim::kPrimCOO2CSR, prim::kPrimCSRDiv, prim::kPrimCSRMM};
  return IsOneOfPrimitiveCNode(cnode, prims);
}

namespace {
ShapeVector ConvertTensorListToShapeVector(const tensor::TensorPtrList &tensor_list, size_t index) {
  ShapeVector shape;
  if (index >= tensor_list.size()) {
    MS_LOG(EXCEPTION) << "Index " << index << " is out of range of " << tensor_list.size();
    return shape;
  }

  auto converter = [](const tensor::TensorPtr tensorptr) {
    MS_EXCEPTION_IF_NULL(tensorptr);
    if (tensorptr->DataDim() != 0) {
      MS_LOG(EXCEPTION) << "Element must be scalar!";
    }
    tensorptr->data_sync(false);
    return *(static_cast<int64_t *>(tensorptr->data_c()));
  };
  std::transform(tensor_list.begin() + index, tensor_list.end(), std::back_inserter(shape), converter);
  if (shape.empty()) {
    MS_LOG(ERROR) << "ShapeVector is empty!";
  }
  return shape;
}
tensor::CSRTensorPtr TensorListToCSRTensor(const tensor::TensorPtrList &tensor_list) {
  tensor::TensorPtr indptr = utils::cast<tensor::TensorPtr>(tensor_list[tensor::CSRTensor::kIndptrIdx]);
  tensor::TensorPtr indices = utils::cast<tensor::TensorPtr>(tensor_list[tensor::CSRTensor::kIndicesIdx]);
  tensor::TensorPtr values = utils::cast<tensor::TensorPtr>(tensor_list[tensor::CSRTensor::kValuesIdx]);
  ShapeVector shape = ConvertTensorListToShapeVector(tensor_list, tensor::CSRTensor::kShapeIdx);
  auto csr_tensor_ptr = std::make_shared<tensor::CSRTensor>(indptr, indices, values, shape);
  return csr_tensor_ptr;
}

tensor::COOTensorPtr TensorListToCOOTensor(const tensor::TensorPtrList &tensor_list) {
  tensor::TensorPtr indices = utils::cast<tensor::TensorPtr>(tensor_list[tensor::COOTensor::kIndicesIdx]);
  tensor::TensorPtr values = utils::cast<tensor::TensorPtr>(tensor_list[tensor::COOTensor::kValuesIdx]);
  ShapeVector shape = ConvertTensorListToShapeVector(tensor_list, tensor::COOTensor::kShapeIdx);
  auto coo_tensor_ptr = std::make_shared<tensor::COOTensor>(indices, values, shape);
  return coo_tensor_ptr;
}
}  // namespace

tensor::MetaSparseTensorPtr TensorListToSparseTensor(const abstract::AbstractBasePtr &abs_sparse,
                                                     const tensor::TensorPtrList &tensor_list) {
  if (abs_sparse->isa<abstract::AbstractCOOTensor>()) {
    return TensorListToCOOTensor(tensor_list);
  }
  return TensorListToCSRTensor(tensor_list);
}

std::vector<ShapeVector> BaseShapeToShapeVector(const abstract::BaseShapePtr &base_shape) {
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    const auto &shape = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    return {shape->shape()};
  } else if (base_shape->isa<abstract::SequenceShape>()) {
    const auto &tuple_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (tuple_shape->size() == 0) {
      return {};
    }
    // If the shape is a tuple shape, all shapes need to be consistent.
    auto element_base_shape = (*tuple_shape)[0];
    if (element_base_shape->isa<abstract::Shape>()) {
      const auto &element_shape = element_base_shape->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(element_shape);
      return std::vector<ShapeVector>(tuple_shape->size(), element_shape->shape());
    } else if (element_base_shape->isa<abstract::NoShape>()) {
      return std::vector<ShapeVector>(tuple_shape->size(), {1});
    }
  } else if (base_shape->isa<abstract::NoShape>() || base_shape->isa<abstract::DynamicSequenceShape>()) {
    return {};
  }
  MS_LOG(WARNING) << "Invalid shape:" << base_shape->ToString();
  return {};
}

ShapeVector BaseShapeToShape(const abstract::BaseShapePtr &base_shape) {
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    const auto &shape = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    return shape->shape();
  } else if (base_shape->isa<abstract::NoShape>()) {
    return {};
  }
  MS_LOG(WARNING) << "Invalid shape:" << base_shape->ToString();
  return {};
}

ValuePtr UpdateValueByAttrDataType(const ValuePtr &value, const std::string &attr_data_type) {
  static std::set<std::string> kListDataType = {"listInt", "listStr", "listBool", "listFloat"};
  auto iter = kListDataType.find(attr_data_type);
  ValuePtr ret = value;
  if (iter != kListDataType.end()) {
    if (!value->isa<ValueSequence>()) {
      std::vector<ValuePtr> value_vec;
      value_vec.push_back(value);
      ret = std::make_shared<ValueTuple>(value_vec);
    }
  }
  return ret;
}

namespace {
size_t GetHashId(int a, int b) { return a < b ? hash_combine(a, b) : hash_combine(b, a); }

static const std::map<size_t, TypeId> tensor_tensor_convert_map = {
  // Bool
  {GetHashId(kNumberTypeBool, kNumberTypeBool), kNumberTypeBool},
  {GetHashId(kNumberTypeBool, kNumberTypeInt8), kNumberTypeInt8},
  {GetHashId(kNumberTypeBool, kNumberTypeInt16), kNumberTypeInt16},
  {GetHashId(kNumberTypeBool, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeBool, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt8), kNumberTypeUInt8},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt16), kNumberTypeUInt16},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt32), kNumberTypeUInt32},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt64), kNumberTypeUInt64},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeBool, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeBool, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeBool, kNumberTypeComplex128), kNumberTypeComplex128},
  // Int8
  {GetHashId(kNumberTypeInt8, kNumberTypeInt8), kNumberTypeInt8},
  {GetHashId(kNumberTypeInt8, kNumberTypeInt16), kNumberTypeInt16},
  {GetHashId(kNumberTypeInt8, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeInt8, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt8, kNumberTypeUInt8), kNumberTypeInt16},
  {GetHashId(kNumberTypeInt8, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeInt8, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeInt8, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeInt8, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeInt8, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeInt8, kNumberTypeComplex128), kNumberTypeComplex128},
  // Int16
  {GetHashId(kNumberTypeInt16, kNumberTypeInt16), kNumberTypeInt16},
  {GetHashId(kNumberTypeInt16, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeInt16, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt16, kNumberTypeUInt8), kNumberTypeInt16},
  {GetHashId(kNumberTypeInt16, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeInt16, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeInt16, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeInt16, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeInt16, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeInt16, kNumberTypeComplex128), kNumberTypeComplex128},
  // Int32
  {GetHashId(kNumberTypeInt32, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeInt32, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt32, kNumberTypeUInt8), kNumberTypeInt32},
  {GetHashId(kNumberTypeInt32, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeInt32, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeInt32, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeInt32, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeInt32, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeInt32, kNumberTypeComplex128), kNumberTypeComplex128},
  // Int64
  {GetHashId(kNumberTypeInt64, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt64, kNumberTypeUInt8), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeInt64, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeInt64, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeInt64, kNumberTypeComplex128), kNumberTypeComplex128},
  // UInt8
  {GetHashId(kNumberTypeUInt8, kNumberTypeUInt8), kNumberTypeUInt8},
  {GetHashId(kNumberTypeUInt8, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeUInt8, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeUInt8, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeUInt8, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeUInt8, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeUInt8, kNumberTypeComplex128), kNumberTypeComplex128},
  // UInt16
  {GetHashId(kNumberTypeUInt16, kNumberTypeUInt16), kNumberTypeUInt16},
  // UInt32
  {GetHashId(kNumberTypeUInt32, kNumberTypeUInt32), kNumberTypeUInt32},
  // UInt64
  {GetHashId(kNumberTypeUInt64, kNumberTypeUInt64), kNumberTypeUInt64},
  // Float16
  {GetHashId(kNumberTypeFloat16, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeFloat16, kNumberTypeBFloat16), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat16, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat16, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeFloat16, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeFloat16, kNumberTypeComplex128), kNumberTypeComplex128},
  // BFloat16
  {GetHashId(kNumberTypeBFloat16, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeBFloat16, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeBFloat16, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeBFloat16, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeBFloat16, kNumberTypeComplex128), kNumberTypeComplex128},
  // Float32
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeFloat32, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeFloat32, kNumberTypeComplex128), kNumberTypeComplex128},
  // Float64
  {GetHashId(kNumberTypeFloat64, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeFloat64, kNumberTypeComplex64), kNumberTypeComplex128},
  {GetHashId(kNumberTypeFloat64, kNumberTypeComplex128), kNumberTypeComplex128},
  // Complex64
  {GetHashId(kNumberTypeComplex64, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeComplex64, kNumberTypeComplex128), kNumberTypeComplex128},
  // Complex128
  {GetHashId(kNumberTypeComplex128, kNumberTypeComplex128), kNumberTypeComplex128},
};

static const std::map<size_t, TypeId> scalar_tensor_convert_map = {
  // Scalar is bool.
  {GetHashId(kNumberTypeBool, kNumberTypeBool), kNumberTypeBool},
  {GetHashId(kNumberTypeBool, kNumberTypeInt8), kNumberTypeInt8},
  {GetHashId(kNumberTypeBool, kNumberTypeInt16), kNumberTypeInt16},
  {GetHashId(kNumberTypeBool, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeBool, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt8), kNumberTypeUInt8},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt16), kNumberTypeUInt16},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt32), kNumberTypeUInt32},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt64), kNumberTypeUInt64},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeBool, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeBool, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeBool, kNumberTypeComplex128), kNumberTypeComplex128},
  // Scalar is int.
  {GetHashId(kNumberTypeInt64, kNumberTypeBool), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt8), kNumberTypeInt8},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt16), kNumberTypeInt16},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt64, kNumberTypeUInt8), kNumberTypeUInt8},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeInt64, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeInt64, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeInt64, kNumberTypeComplex128), kNumberTypeComplex128},
  // Scalar is float.
  {GetHashId(kNumberTypeFloat32, kNumberTypeBool), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt8), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt16), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt64), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeUInt8), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeFloat32, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeFloat32, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeFloat32, kNumberTypeComplex128), kNumberTypeComplex128},
};

TypeId ConvertTypeForTensorsOrScalars(const TypeId &current, const TypeId &other, const size_t hash_id) {
  auto iter = tensor_tensor_convert_map.find(hash_id);
  if (iter != tensor_tensor_convert_map.end()) {
    return iter->second;
  }
  MS_EXCEPTION(TypeError) << "Type implicit conversion between " << TypeIdToString(current) << " and "
                          << TypeIdToString(other) << " is not supported.";
}

TypeId ConvertTypeBetweenTensorAndScalar(const TypeId &tensor_type_id, const TypeId &scalar_type_id,
                                         const size_t hash_id) {
  auto iter = scalar_tensor_convert_map.find(hash_id);
  if (iter != scalar_tensor_convert_map.end()) {
    return iter->second;
  }
  MS_EXCEPTION(TypeError) << "Type implicit conversion between Tensor[" << TypeIdToString(tensor_type_id) << "] and "
                          << TypeIdToString(scalar_type_id) << " is not supported.";
}

TypeId GetConversionType(const TypeId &current, bool current_arg_is_tensor, bool is_parameter,
                         const std::pair<TypeId, bool> &sig_type, const TypeId &ref_type_id) {
  TypeId saved_type_id = sig_type.first;
  bool saved_has_tensor = sig_type.second;
  if (current == saved_type_id) {
    return current;
  }

  if (current != kTypeUnknown && saved_type_id != kTypeUnknown) {
    auto hash_id = GetHashId(current, saved_type_id);
    // Tensor + Scalar, Scalar + Tensor
    if (MS_UNLIKELY(current_arg_is_tensor ^ saved_has_tensor)) {
      return ConvertTypeBetweenTensorAndScalar(current, saved_type_id, hash_id);
    }
    // Tensor + Tensor, Scalar + Scalar
    if ((is_parameter || saved_type_id == ref_type_id) &&
        hash_id == GetHashId(kNumberTypeFloat16, kNumberTypeBFloat16)) {
      // "saved_type_id == ref_type_id": if Parameter exists, its type_id should be equal to the saved_type_id,
      // otherwise it means that the wrong type cast will be performed on the Parameter.
      static bool already_printed = false;
      if (!already_printed) {
        already_printed = true;
        MS_LOG(WARNING) << "For operators with side effects, there is an implicit type conversion between "
                        << TypeIdToString(current) << " and " << TypeIdToString(saved_type_id)
                        << ", which may result in loss of precision. It is recommended to use Float32.";
      }
      return is_parameter ? current : saved_type_id;
    }
    return ConvertTypeForTensorsOrScalars(current, saved_type_id, hash_id);
  }
  return current != kTypeUnknown ? current : saved_type_id;
}
}  // namespace

std::map<SignatureEnumDType, std::pair<TypeId, bool>> GetSignatureTypeMap(const std::vector<SignatureEnumDType> &dtypes,
                                                                          const std::vector<TypeId> &args_type_id,
                                                                          const std::vector<bool> &args_is_tensor,
                                                                          const std::set<size_t> &write_indices) {
  // {T0: (target_type_id=Int32, has_tensor=true), T1: (target_type_id=Float32, has_tensor=false), ...}
  std::map<SignatureEnumDType, std::pair<TypeId, bool>> sig_type_map;
  std::map<SignatureEnumDType, TypeId> ref_type_map;
  size_t args_size = args_type_id.size();
  for (size_t i = 0; i < args_size; ++i) {
    bool is_parameter = write_indices.find(i) != write_indices.end();
    const auto &it = sig_type_map.find(dtypes[i]);
    if (it == sig_type_map.end()) {
      (void)sig_type_map.insert(std::make_pair(dtypes[i], std::make_pair(args_type_id[i], args_is_tensor[i])));
      (void)ref_type_map.insert(std::make_pair(dtypes[i], is_parameter ? args_type_id[i] : kTypeUnknown));
    } else {
      it->second.first =
        GetConversionType(args_type_id[i], args_is_tensor[i], is_parameter, it->second, ref_type_map[dtypes[i]]);
      it->second.second = args_is_tensor[i] || it->second.second;
      if (is_parameter && ref_type_map[dtypes[i]] == kTypeUnknown) {
        ref_type_map[dtypes[i]] = args_type_id[i];
      }
    }
  }
  return sig_type_map;
}

TypeId ConvertTypeForTensorsOrScalars(const TypeId &type1, const TypeId &type2) {
  return ConvertTypeForTensorsOrScalars(type1, type2, GetHashId(type1, type2));
}

std::string ValueSimpleInfoToString(const ValueSimpleInfo &value_simple_info) {
  std::ostringstream buf;
  buf << "Value simple info element size : " << value_simple_info.size_;
  for (size_t i = 0; i < value_simple_info.size_; ++i) {
    buf << ". The " << i << "th shape: " << value_simple_info.shape_vector_[i] << ", dtype "
        << value_simple_info.dtype_vector_[i];
    if (!value_simple_info.object_type_vector_.empty()) {
      buf << ", object type " << value_simple_info.object_type_vector_[i];
    }
  }
  return buf.str();
}

abstract::AbstractBasePtr TransformValueSimpleInfoToAbstract(const ValueSimpleInfo &value_simple_info) {
  if (value_simple_info.size_ < 1) {
    MS_LOG(EXCEPTION) << "Simple infer info size must greater than 1, but got " << value_simple_info.size_;
  }
  abstract::AbstractBasePtr out_abs;
  if (value_simple_info.size_ == 1 && !value_simple_info.is_tuple_output_) {
    out_abs = std::make_shared<abstract::AbstractTensor>(value_simple_info.dtype_vector_[kIndex0],
                                                         value_simple_info.shape_vector_[kIndex0]);
  } else {
    AbstractBasePtrList out_abs_list;
    out_abs_list.resize(value_simple_info.size_);
    for (size_t i = 0; i < value_simple_info.size_; ++i) {
      out_abs_list[i] = std::make_shared<abstract::AbstractTensor>(value_simple_info.dtype_vector_[i],
                                                                   value_simple_info.shape_vector_[i]);
    }
    out_abs = std::make_shared<abstract::AbstractTuple>(out_abs_list);
  }
  return out_abs;
}
}  // namespace mindspore
