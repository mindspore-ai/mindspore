/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "utils/convert_utils.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <list>
#include <utility>
#include <cfloat>

#include "abstract/abstract_value.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"

namespace mindspore {
bool ValueToBool(const ValuePtr &v, bool *value) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<BoolImm>()) {
    *value = v->cast<BoolImmPtr>()->value();
  } else if (v->isa<Int32Imm>()) {
    *value = v->cast<Int32ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<UInt32Imm>()) {
    *value = v->cast<UInt32ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<FP32Imm>()) {
    *value = v->cast<FP32ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<FP64Imm>()) {
    *value = v->cast<FP64ImmPtr>()->value() == 0 ? false : true;
  } else if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    (void)tensor->data_sync();
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

bool BaseRefToInt(const ValuePtr &v, int *value) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    (void)tensor->data_sync();
    int *tensor_data = static_cast<int *>(tensor->data_c());
    auto vb = tensor_data[0];
    *value = vb;
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
    if (vb == true) {
      *value = true;
    } else {
      *value = false;
    }
  } else if (utils::isa<int>(v)) {
    auto vb = utils::cast<int>(v);
    if (vb == 0) {
      *value = false;
    } else {
      *value = true;
    }
  } else if (utils::isa<unsigned int>(v)) {
    auto vb = utils::cast<unsigned int>(v);
    if (vb == 0) {
      *value = false;
    } else {
      *value = true;
    }
  } else if (utils::isa<float>(v)) {
    auto vb = utils::cast<float>(v);
    if (vb >= -FLT_EPSILON && vb <= FLT_EPSILON) {
      *value = false;
    } else {
      *value = true;
    }
  } else if (utils::isa<double>(v)) {
    auto vb = utils::cast<double>(v);
    if (vb >= -DBL_EPSILON && vb <= DBL_EPSILON) {
      *value = false;
    } else {
      *value = true;
    }
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
    auto a1 = GetValueNode(node1);
    auto a2 = GetValueNode(node2);
    if (a1->isa<Primitive>() && a2->isa<Primitive>()) {
      return a1->cast<PrimitivePtr>()->name() == a2->cast<PrimitivePtr>()->name();
    } else if (a1->isa<tensor::Tensor>() && a2->isa<tensor::Tensor>()) {
      return a1->cast<tensor::TensorPtr>()->ValueEqual(*(a2->cast<tensor::TensorPtr>()));
    } else {
      return *a1 == *a2;
    }
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

bool SameSubgraph(AnfNodePtr root1, AnfNodePtr root2, FuncGraphPairMapEquiv *equiv_func_graph,
                  NodeMapEquiv *const equiv_node) {
  std::unordered_set<AnfNodePtr> done;
  std::stack<std::pair<AnfNodePtr, AnfNodePtr>> todo;

  todo.push(std::make_pair(root1, root2));
  while (todo.size() > 0) {
    AnfNodePtr node1 = todo.top().first;
    if (done.count(node1) > 0) {
      todo.pop();
      continue;
    }
    AnfNodePtr node2 = todo.top().second;

    bool condition = false;
    std::vector<AnfNodePtr> s1 = SuccIncoming(node1);
    std::vector<AnfNodePtr> s2 = SuccIncoming(node2);

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

bool Isomorphic(FuncGraphPtr fg1, FuncGraphPtr fg2, FuncGraphPairMapEquiv *equiv_func_graph,
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
  tensor::TensorPtr tensor = nullptr;
  if (scalar->isa<FloatImm>()) {
    tensor = std::make_shared<tensor::Tensor>(static_cast<double>(GetValue<float>(scalar)), kFloat32);
  } else if (scalar->isa<Int32Imm>()) {
    tensor = std::make_shared<tensor::Tensor>(static_cast<int64_t>(GetValue<int>(scalar)), kInt32);
  } else if (scalar->isa<Int64Imm>()) {
    tensor = std::make_shared<tensor::Tensor>(GetValue<int64_t>(scalar), kInt64);
  } else if (scalar->isa<BoolImm>()) {
    const int64_t bool_value = GetValue<bool>(scalar) ? 1 : 0;
    tensor = std::make_shared<tensor::Tensor>(bool_value, kBool);
  } else {
    auto type = scalar->type();
    auto type_str = (type == nullptr) ? "nullptr" : type->ToString();
    MS_LOG(EXCEPTION) << "Invalid scalar type: " << type_str;
  }
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor;
}

void TensorValueToTensor(const ValuePtr &value, std::vector<tensor::TensorPtr> *tensors) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(tensors);
  if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      if (element->isa<tensor::Tensor>()) {
        auto tensor = element->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        tensors->push_back(tensor);
      }
    }
  } else if (value->isa<tensor::Tensor>()) {
    tensor::TensorPtr tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensors->push_back(tensor);
  }
}
}  // namespace mindspore
