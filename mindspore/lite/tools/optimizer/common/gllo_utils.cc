/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/common/gllo_utils.h"
#include <vector>
#include <algorithm>
#include <utility>
#include "src/ir/primitive_t_value.h"
#include "frontend/operator/ops.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAnfPrimitiveIndex = 0;
bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
}

bool IsRealKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // parameter and value node is not a real kernel too
  if (!node->isa<CNode>()) {
    return true;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode(%s)" << node->DebugString();
  }
  auto input = cnode->inputs()[0];
  bool is_virtual_node = IsPrimitive(input, prim::kPrimImageSummary) || IsPrimitive(input, prim::kPrimScalarSummary) ||
      IsPrimitive(input, prim::kPrimTensorSummary) ||
      IsPrimitive(input, prim::kPrimHistogramSummary) || IsPrimitive(input, prim::kPrimMakeTuple) ||
      IsPrimitive(input, prim::kPrimStateSetItem) || IsPrimitive(input, prim::kPrimDepend) ||
      IsPrimitive(input, prim::kPrimTupleGetItem) || IsPrimitive(input, prim::kPrimControlDepend) ||
      IsPrimitive(input, prim::kPrimReturn) || IsPrimitive(input, prim::kPrimPartial);
  return !is_virtual_node;
}

ValueNodePtr CreateValueNodeWithSexp(const BaseRef &sexp) {
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

CNodePtr CreateCNodeWithGraph(const std::vector<AnfNodePtr> &input_nodes, const BaseRef &graph) {
  if (utils::isa<FuncGraphPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<FuncGraphPtr>(graph));
  }
  if (utils::isa<VarPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<VarPtr>(graph));
  }
  return nullptr;
}

VarNodePtr CreateVarNodeWithSexp(const BaseRef &sexp, const BaseRef &graph) {
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

AnfNodePtr HandleSexpVector(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                            bool multigraph) {
  MS_LOG(DEBUG) << "HandleSexpVector sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  std::vector<AnfNodePtr> input_nodes;
  const auto &tuple = utils::cast<VectorRef>(sexp);
  if (multigraph && utils::isa<VarPtr>(graph)) {
    for (auto &x : tuple) {
      AnfNodePtr node = SexpToNode(x, std::make_shared<Var>("G"), primitive_vars, true);
      input_nodes.push_back(node);
    }
    VarPtr var_ptr = utils::cast<VarPtr>(graph);
    return std::make_shared<CNode>(input_nodes, var_ptr);
  }

  for (auto &x : tuple) {
    AnfNodePtr node = SexpToNode(x, graph, primitive_vars, multigraph);
    input_nodes.push_back(node);
  }
  return CreateCNodeWithGraph(input_nodes, graph);
}
}  // namespace

bool AnfEqual(const BaseRef &a, const BaseRef &b) {
  if (utils::isa<AnfNodePtr>(a) && utils::isa<AnfNodePtr>(b)) {
    auto a_node = utils::cast<AnfNodePtr>(a);
    auto b_node = utils::cast<AnfNodePtr>(b);
    MS_EXCEPTION_IF_NULL(a_node);
    MS_EXCEPTION_IF_NULL(b_node);
    if (IsValueNode<Primitive>(a_node) && IsValueNode<Primitive>(b_node)) {
      auto a_value_node = a_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(a_value_node);
      auto a_value = a_value_node->value();
      MS_EXCEPTION_IF_NULL(a_value);
      auto a_prim = a_value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(a_prim);

      auto b_value_node = b_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(b_value_node);
      auto b_value = b_value_node->value();
      MS_EXCEPTION_IF_NULL(b_value);
      auto b_prim = b_value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(b_prim);

      return a_prim->name() == b_prim->name();
    } else if (a_node->isa<ValueNode>() && b_node->isa<ValueNode>()) {
      auto a_value_node_ptr = a_node->cast<ValueNodePtr>();
      if (a_value_node_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "cast value node ptr fail";
      }
      auto a_value_ptr = a_value_node_ptr->value();
      if (a_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "value ptr is nullptr";
      }

      auto b_value_node_ptr = b_node->cast<ValueNodePtr>();
      if (b_value_node_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "cast value node ptr fail";
      }
      auto b_value_ptr = b_value_node_ptr->value();
      if (b_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "value ptr is nullptr";
      }

      if (utils::isa<lite::PrimitiveTValue>(a_value_ptr) && utils::isa<lite::PrimitiveTValue>(b_value_ptr)) {
        auto a_obj = (lite::PrimitiveTValue *) (a_value_ptr.get());
        auto b_obj = (lite::PrimitiveTValue *) (b_value_ptr.get());
        return (*a_obj) == (*b_obj);
      } else {
        return (*a_value_ptr) == (*b_value_ptr);
      }
    }
  }
  if (a.m_ptr->isa<lite::PrimitiveTValue>() && b.m_ptr->isa<lite::PrimitiveTValue>()) {
    auto a_value_node_ptr = a.m_ptr->cast<PrimitiveTValuePtr>();
    auto b_value_node_ptr = b.m_ptr->cast<PrimitiveTValuePtr>();
    return a_value_node_ptr->GetPrimitiveT()->value.type == b_value_node_ptr->GetPrimitiveT()->value.type;
  }

  return a == b;
}

bool CNodeTypeEqual(const BaseRef &a, const BaseRef &b) {
  // To matchCNode and Kernel's type
  if (utils::isa<CNode>(a) && utils::isa<CNode>(b)) {
    return true;
  }
  return a.type() == b.type();
}

AnfNodePtr SexpToNode(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars, bool multigraph) {
  MS_LOG(DEBUG) << "SexpToNode sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  MS_EXCEPTION_IF_NULL(primitive_vars);
  if (utils::isa<VectorRef>(sexp)) {
    return HandleSexpVector(sexp, graph, primitive_vars, multigraph);
  }
  if (utils::isa<VarPtr>(sexp)) {
    auto var_ptr = utils::cast<VarPtr>(sexp);
    MS_EXCEPTION_IF_NULL(var_ptr);
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
    MS_LOG(EXCEPTION) << "sexp cannot converted. sexp: " + sexp.ToString();
  }
  return value_node;
}

bool IsRealCNodeKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // parameter and value node is not a real cnode kernel
  if (!node->isa<CNode>()) {
    return false;
  }
  // return considered as a real node
  if (CheckPrimitiveType(node, prim::kPrimReturn)) {
    return true;
  }
  return IsRealKernel(node);
}
bool IsGraphKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // graph kernel should be a real cnode kernel.
  if (!IsRealCNodeKernel(node)) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input = cnode->input(kAnfPrimitiveIndex);
  // graph kernel should has func_graph as first input.
  if (!IsValueNode<FuncGraph>(input)) {
    return false;
  }

  auto func_graph = GetValueNode<FuncGraphPtr>(input);
  MS_EXCEPTION_IF_NULL(func_graph);
  return func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
}

void CheckIfFuncGraphIsNull(const FuncGraphPtr &graph) {
  if (graph == nullptr) {
    MS_LOG(EXCEPTION) << "The graph is null.";
  }
}

void CheckIfAnfNodeIsNull(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(EXCEPTION) << "The AnfNode is null.";
  }
}

void CheckIfCNodeIsNull(const CNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(EXCEPTION) << "The CNode is null.";
  }
}

void CheckIfVarIsNull(const VarPtr &var) {
  if (var == nullptr) {
    MS_LOG(EXCEPTION) << "The Var is null.";
  }
}

void CheckIfNodeIsParam(const AnfNodePtr &node) {
  if (node != nullptr && !utils::isa<ParameterPtr>(node)) {
    MS_LOG(EXCEPTION) << "The Node is not param.";
  }
}

void CheckInputSize(const CNodePtr &node, const int size) {
  if (node->inputs().size() != size) {
    MS_LOG(EXCEPTION) << "The input size of node must be " << size << ", but it is" << node->inputs().size();
  }
}

void CheckLeastInputSize(const CNodePtr &node, const int size) {
  if (node->inputs().size() < size) {
    MS_LOG(EXCEPTION) << "The input size of node must be " << size << ", but it is" << node->inputs().size();
  }
}

AnfNodePtr AddNewBiasNode(float *bias_data, const FuncGraphPtr &func_graph, int kernel_num,
                          const ParamValueLitePtr &weight_tensor) {
  auto bias_parameter = func_graph->add_parameter();
  MS_ASSERT(bias_parameter != nullptr);
  std::vector<int> shape = {kernel_num};
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(TypeIdToType(weight_tensor->tensor_type()), shape);
  bias_parameter->set_abstract(abstract_tensor);

  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_addr(bias_data);
  param_value->set_tensor_size(kernel_num * sizeof(float) / sizeof(uint8_t));
  bias_parameter->set_default_param(param_value);
  return bias_parameter;
}

schema::PrimitiveType GetCNodeType(const BaseRef &n) {
  ValueNodePtr value_node;
  if (utils::isa<CNodePtr>(n)) {
    auto in = utils::cast<CNodePtr>(n);
    value_node = in->input(0)->cast<ValueNodePtr>();
  } else if (utils::isa<ValueNodePtr>(n)) {
    value_node = utils::cast<ValueNodePtr>(n);
  } else {
    MS_LOG(EXCEPTION) << "only value node or cnode has type";
    return schema::PrimitiveType_NONE;
  }
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_ASSERT(value != nullptr);
  if (utils::isa<PrimitiveTValuePtr>(value)) {
    auto primitive = value->cast<PrimitiveTValuePtr>();
    MS_ASSERT(primitive != nullptr);
    return primitive->GetPrimitiveT()->value.type;
  } else if (utils::isa<Primitive>(value)) {
    auto primitive = value->cast<PrimitivePtr>();
    MS_ASSERT(primitive != nullptr);
    MS_LOG(INFO) << "anf primitive node type:" << primitive->name();
    return schema::PrimitiveType_NONE;
  }
  return schema::PrimitiveType_NONE;
}

bool IsParamNode(const BaseRef &n) {
  return utils::isa<ParameterPtr>(n);
}

bool IsConvNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Conv2D || type == schema::PrimitiveType_DepthwiseConv2D;
  }
  return false;
}

bool CheckIsAllInputsParam(const AnfNodePtr &node) {
  if (utils::isa<CNode>(node)) {
    auto cnode = node->cast<CNodePtr>();
    for (auto i = 1; i < cnode->inputs().size(); i++) {
      if (!utils::isa<Parameter>(cnode->input(i))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

size_t GetOutputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto type = node->Type();
  if (type == nullptr) {
    return 1;
  }
  if (type->isa<Tuple>()) {
    auto tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    return tuple_type->size();
  } else if (type->isa<TensorType>() || type->isa<Number>()) {
    return 1;
  } else if (type->isa<TypeNone>()) {
    return 0;
  } else {
    return 1;
  }
}

bool IsMultiOutputTensors(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  auto output_node_list = GetRealNodeUsedList(graph, node);
  if (output_node_list->size() != 1) {
      MS_LOG(DEBUG) << "fusion node has multi output nodes";
      return true;
  }
  return false;
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedList(const FuncGraphPtr &graph,
                                                                             const AnfNodePtr &node) {
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }
  auto output_info_list = iter->second;
  std::copy(output_info_list.begin(), output_info_list.end(), std::back_inserter(*output_node_list));
  return output_node_list;
}
}  // namespace opt
}  // namespace mindspore
