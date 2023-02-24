/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <utility>
#include <cfloat>
#include <cmath>

#include "ir/value.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"
#include "include/common/utils/utils.h"

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

void TensorValueToTensor(const ValuePtr &value, std::vector<tensor::TensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(tensors);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
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

ValuePtr ShallowCopyTensorValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor_value = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_value);
    auto shallow_tensor = std::make_shared<tensor::Tensor>(*tensor_value);
    shallow_tensor->set_base_shape(tensor_value->base_shape_ptr());
    return shallow_tensor;
  } else if (value->isa<ValueSequence>()) {
    std::vector<ValuePtr> values;
    auto value_tuple = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    (void)std::transform(value_tuple->value().begin(), value_tuple->value().end(), std::back_inserter(values),
                         [](const ValuePtr &elem) { return ShallowCopyTensorValue(elem); });
    return std::make_shared<ValueTuple>(values);
  } else {
    return value;
  }
}

size_t CountValueNum(const ValueSequencePtr &value_sequence) {
  MS_EXCEPTION_IF_NULL(value_sequence);
  size_t cnt = 0;
  const auto &value_list = value_sequence->value();
  for (const auto &value : value_list) {
    if (value->isa<None>()) {
      continue;
    } else if (value->isa<ValueSequence>()) {
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
  } else if (base_shape->isa<abstract::TupleShape>()) {
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
}  // namespace mindspore
