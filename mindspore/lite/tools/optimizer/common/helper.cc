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

#define USE_DEPRECATED_API
#include <memory>
#include <vector>
#include <algorithm>
#include "tools/optimizer/common/helper.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
ValueNodePtr Helper::CreateValueNodeWithSexp(const BaseRef &sexp) {
  if (utils::isa<int>(sexp)) {
    return NewValueNode(utils::cast<int>(sexp));
  }
  if (utils::isa<float>(sexp)) {
    return NewValueNode(utils::cast<float>(sexp));
  }
  if (utils::isa<bool>(sexp)) {
    return NewValueNode(utils::cast<bool>(sexp));
  }
  if (utils::isa<ValuePtr>(sexp)) {
    return NewValueNode(utils::cast<ValuePtr>(sexp));
  }
  return nullptr;
}

CNodePtr Helper::CreateCNodeWithGraph(const std::vector<AnfNodePtr> &input_nodes, const BaseRef &graph) {
  if (utils::isa<FuncGraphPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<FuncGraphPtr>(graph));
  }
  if (utils::isa<VarPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<VarPtr>(graph));
  }
  return nullptr;
}

VarNodePtr Helper::CreateVarNodeWithSexp(const BaseRef &sexp, const BaseRef &graph) {
  if (utils::isa<VarPtr>(graph)) {
    MS_LOG(DEBUG) << "make VarPtr " + graph.ToString();
    return std::make_shared<VarNode>(utils::cast<VarPtr>(sexp), nullptr);
  }
  if (utils::isa<FuncGraphPtr>(graph)) {
    MS_LOG(DEBUG) << "VarNode, should input a Var in graph. It's GraphPtr: " + graph.ToString();
    return std::make_shared<VarNode>(utils::cast<VarPtr>(sexp), utils::cast<FuncGraphPtr>(graph));
  }
  MS_LOG(ERROR) << "VarNode, should input a Var in graph. It's " + graph.ToString();
  return nullptr;
}

AnfNodePtr Helper::HandleSexpVector(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                                    bool multigraph) {
  if (primitive_vars == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    MS_LOG(ERROR) << "primitive_vars is nullptr";
    return nullptr;
  }
  MS_LOG(DEBUG) << "HandleSexpVector sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  std::vector<AnfNodePtr> input_nodes;
  const auto &tuple = utils::cast<VectorRef>(sexp);
  if (multigraph && utils::isa<VarPtr>(graph)) {
    for (auto &x : tuple) {
      auto is_var = std::make_shared<Var>("G");
      MS_CHECK_TRUE_RET(is_var != nullptr, nullptr);
      AnfNodePtr node = SexpToNode(x, is_var, primitive_vars, true);
      input_nodes.push_back(node);
    }
    auto var_ptr = utils::cast<VarPtr>(graph);
    return std::make_shared<CNode>(input_nodes, var_ptr);
  }

  for (auto &x : tuple) {
    AnfNodePtr node = SexpToNode(x, graph, primitive_vars, multigraph);
    input_nodes.push_back(node);
  }
  return CreateCNodeWithGraph(input_nodes, graph);
}

namespace {
bool AnfEqualPrimitive(const AnfNodePtr &a_node, const AnfNodePtr &b_node) {
  auto a_value_node = a_node->cast<ValueNodePtr>();
  auto b_value_node = b_node->cast<ValueNodePtr>();
  if (a_value_node == nullptr || b_value_node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }

  auto a_value = a_value_node->value();
  auto b_value = b_value_node->value();
  if (a_value == nullptr || b_value == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }

  auto a_prim = a_value->cast<PrimitivePtr>();
  auto b_prim = b_value->cast<PrimitivePtr>();
  if (a_prim == nullptr || b_prim == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  return a_prim->name() == b_prim->name();
}

bool AnfEqualValueNode(const AnfNodePtr &a_node, const AnfNodePtr &b_node) {
  auto a_value_node_ptr = a_node->cast<ValueNodePtr>();
  auto b_value_node_ptr = b_node->cast<ValueNodePtr>();
  if (a_value_node_ptr == nullptr || b_value_node_ptr == nullptr) {
    MS_LOG(ERROR) << "cast value node ptr fail";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  auto a_value_ptr = a_value_node_ptr->value();
  auto b_value_ptr = b_value_node_ptr->value();
  if (a_value_ptr == nullptr || b_value_ptr == nullptr) {
    MS_LOG(ERROR) << "value ptr is nullptr";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }

  if (utils::isa<ops::PrimitiveC>(a_value_ptr) && utils::isa<ops::PrimitiveC>(b_value_ptr)) {
    auto a_obj = static_cast<ops::PrimitiveC *>(a_value_ptr.get());
    auto b_obj = static_cast<ops::PrimitiveC *>(b_value_ptr.get());
    return (*a_obj) == (*b_obj);
  } else {
    return (*a_value_ptr) == (*b_value_ptr);
  }
}
}  // namespace

// not implement for lite, just for api compatible
CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg,
                  const std::vector<AnfNodePtr> &orig_nodes) {
  return fg->NewCNode(inputs);
}

// not implement for lite, just for api compatible
CNodePtr NewCNode(const CNodePtr &cnode, const KernelGraphPtr &fg, const std::vector<AnfNodePtr> &orig_nodes) {
  MS_LOG(DEBUG) << "Not implement for lite, just for api compatible.";
  return nullptr;
}

// not implement for lite, just for api compatible
AbstractBasePtr CppInferShapeAndType(const PrimitivePtr &prim, const AbstractBasePtrList &args_spec_list) {
  MS_LOG(DEBUG) << "Not implement for lite, just for api compatible.";
  return nullptr;
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedList(const FuncGraphPtr &graph,
                                                                             const AnfNodePtr &node) {
  return Helper::GetRealNodeUsedList(graph, node);
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> Helper::GetRealNodeUsedList(const FuncGraphPtr &graph,
                                                                                     const AnfNodePtr &node) {
  if (graph == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_CHECK_TRUE_RET(output_node_list != nullptr, nullptr);
  auto manager = graph->manager();
  if (manager == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(ERROR) << "node has no output in manager";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_ERROR);
    return nullptr;
  }
  auto output_info_list = iter->second;
  std::copy(output_info_list.begin(), output_info_list.end(), std::back_inserter(*output_node_list));
  return output_node_list;
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> Helper::GetRealNodeUsedListByOutputIdx(
  const FuncGraphPtr &graph, const AnfNodePtr &node, size_t output_index) {
  if (graph == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr.";
    return nullptr;
  }
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_CHECK_TRUE_RET(output_node_list != nullptr, nullptr);
  auto manager = graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(ERROR) << "node has no output in manager";
    return output_node_list;
  }
  auto output_info_list = iter->second;
  for (const auto &output_info : output_info_list) {
    size_t used_output_index;
    if (CheckPrimitiveType(output_info.first, prim::kPrimTupleGetItem)) {
      used_output_index = GetTupleGetItemOutIndex(utils::cast<CNodePtr>(output_info.first));
    } else if (CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      used_output_index = output_index;
    } else {
      if (output_index != 0) {
        MS_LOG(ERROR) << "node has no output in manager";
        return output_node_list;
      }
      return output_node_list;
    }
    if (used_output_index == output_index) {
      output_node_list->push_back(output_info);
    }
  }
  return output_node_list;
}

bool AnfEqual(const BaseRef &a, const BaseRef &b) {
  if (utils::isa<AnfNodePtr>(a) && utils::isa<AnfNodePtr>(b)) {
    auto a_node = utils::cast<AnfNodePtr>(a);
    auto b_node = utils::cast<AnfNodePtr>(b);
    if (a_node == nullptr || b_node == nullptr) {
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
      MS_LOG(ERROR) << "a or b is nullptr.";
      return false;
    }
    if (IsValueNode<Primitive>(a_node) && IsValueNode<Primitive>(b_node)) {
      return AnfEqualPrimitive(a_node, b_node);
    }
    if (a_node->isa<ValueNode>() && b_node->isa<ValueNode>()) {
      return AnfEqualValueNode(a_node, b_node);
    }
  }
  if (a.m_ptr->isa<mindspore::ops::PrimitiveC>() && b.m_ptr->isa<mindspore::ops::PrimitiveC>()) {
    auto a_value_node_ptr = a.m_ptr->cast<PrimitiveCPtr>();
    auto b_value_node_ptr = b.m_ptr->cast<PrimitiveCPtr>();
    return a_value_node_ptr->name() == b_value_node_ptr->name();
  }

  return a == b;
}

AnfNodePtr SexpToNode(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars, bool multigraph) {
  return Helper::SexpToNode(sexp, graph, primitive_vars, multigraph);
}

AnfNodePtr Helper::SexpToNode(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                              bool multigraph) {
  MS_LOG(DEBUG) << "SexpToNode sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  if (primitive_vars == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (utils::isa<VectorRef>(sexp)) {
    return HandleSexpVector(sexp, graph, primitive_vars, multigraph);
  }
  if (utils::isa<VarPtr>(sexp)) {
    auto var_ptr = utils::cast<VarPtr>(sexp);
    if (var_ptr == nullptr) {
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
      MS_LOG(ERROR) << "var_ptr is nullptr.";
      return nullptr;
    }
    if (var_ptr->primitive()) {
      (*primitive_vars)[var_ptr->primitive()] = var_ptr;
      return NewValueNode(var_ptr->primitive());
    }
    return CreateVarNodeWithSexp(sexp, graph);
  }
  if (utils::isa<AnfNodePtr>(sexp)) {
    return utils::cast<AnfNodePtr>(sexp);
  }
  auto value_node = CreateValueNodeWithSexp(sexp);
  if (value_node == nullptr) {
    MS_LOG(WARNING) << "sexp cannot converted. sexp: " << sexp.ToString();
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  return value_node;
}
}  // namespace opt
}  // namespace mindspore
