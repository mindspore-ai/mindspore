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
#include "src/ops/primitive_c.h"
#include "frontend/operator/ops.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAnfPrimitiveIndex = 0;
bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
}

bool IsRealKernel(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  // parameter and value node is not a real kernel too
  if (!node->isa<CNode>()) {
    return true;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  if (cnode->inputs().empty()) {
    MS_LOG(ERROR) << "Illegal null input of cnode(%s)" << node->DebugString();
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INPUT_TENSOR_ERROR);
    return false;
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
    if (a_node == nullptr || b_node == nullptr) {
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
      return false;
    }
    if (IsValueNode<Primitive>(a_node) && IsValueNode<Primitive>(b_node)) {
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
      return a_prim->cast<PrimitiveCPtr>()->Type() == b_prim->cast<PrimitiveCPtr>()->Type();
    } else if (a_node->isa<ValueNode>() && b_node->isa<ValueNode>()) {
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
      if (utils::isa<lite::PrimitiveC>(a_value_ptr) && utils::isa<lite::PrimitiveC>(b_value_ptr)) {
        auto a_obj = (lite::PrimitiveC *)(a_value_ptr.get());
        auto b_obj = (lite::PrimitiveC *)(b_value_ptr.get());
        return (*a_obj) == (*b_obj);
      } else {
        return (*a_value_ptr) == (*b_value_ptr);
      }
    }
  }
  if (a.m_ptr->isa<lite::PrimitiveC>() && b.m_ptr->isa<lite::PrimitiveC>()) {
    auto a_value_node_ptr = a.m_ptr->cast<PrimitiveCPtr>();
    auto b_value_node_ptr = b.m_ptr->cast<PrimitiveCPtr>();
    return a_value_node_ptr->Type() == b_value_node_ptr->Type();
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
    MS_LOG(ERROR) << "sexp cannot converted. sexp: " << sexp.ToString();
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  return value_node;
}

bool IsRealCNodeKernel(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
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
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  // graph kernel should be a real cnode kernel.
  if (!IsRealCNodeKernel(node)) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  auto input = cnode->input(kAnfPrimitiveIndex);
  // graph kernel should has func_graph as first input.
  if (!IsValueNode<FuncGraph>(input)) {
    return false;
  }

  auto func_graph = GetValueNode<FuncGraphPtr>(input);
  if (func_graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  return func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
}

int CheckIfFuncGraphIsNull(const FuncGraphPtr &graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "The graph is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  return lite::RET_OK;
}

int CheckIfAnfNodeIsNull(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "The AnfNode is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  return lite::RET_OK;
}

int CheckIfCNodeIsNull(const CNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "The CNode is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  return lite::RET_OK;
}

int CheckIfVarIsNull(const VarPtr &var) {
  if (var == nullptr) {
    MS_LOG(ERROR) << "The Var is null.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  return lite::RET_OK;
}

int CheckIfNodeIsParam(const AnfNodePtr &node) {
  if (node != nullptr && !utils::isa<ParameterPtr>(node)) {
    MS_LOG(ERROR) << "The Node is not param.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_INVALID_OP_ATTR;
  }
  return lite::RET_OK;
}

int CheckInputSize(const CNodePtr &node, const int size) {
  if (static_cast<int>(node->inputs().size()) != size) {
    MS_LOG(ERROR) << "The input size of node must be " << size << ", but it is" << node->inputs().size();
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_INVALID_OP_ATTR;
  }
  return lite::RET_OK;
}

int CheckLeastInputSize(const CNodePtr &node, const int size) {
  if (static_cast<int>(node->inputs().size()) < size) {
    MS_LOG(ERROR) << "The input size of node must be " << size << ", but it is" << node->inputs().size();
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_INVALID_OP_ATTR;
  }
  return lite::RET_OK;
}

ParameterPtr AddNewBiasNode(float *bias_data, const FuncGraphPtr &func_graph, int kernel_num,
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
  param_value->set_format(weight_tensor->format());
  param_value->set_tensor_type(weight_tensor->tensor_type());
  param_value->set_tensor_shape(shape);
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
    MS_LOG(ERROR) << "only value node or cnode has type";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return schema::PrimitiveType_NONE;
  }
  if (value_node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return schema::PrimitiveType_NONE;
  }
  auto value = value_node->value();
  MS_ASSERT(value != nullptr);
  if (utils::isa<PrimitiveCPtr>(value)) {
    auto primitive = value->cast<PrimitiveCPtr>();
    MS_ASSERT(primitive != nullptr);
    return (schema::PrimitiveType)primitive->Type();
  } else if (utils::isa<Primitive>(value)) {
    auto primitive = value->cast<PrimitivePtr>();
    MS_ASSERT(primitive != nullptr);
    MS_LOG(INFO) << "anf primitive node type:" << primitive->name();
    return schema::PrimitiveType_NONE;
  }
  return schema::PrimitiveType_NONE;
}

bool IsParamNode(const BaseRef &n) {
  if (!utils::isa<ParameterPtr>(n)) {
    return false;
  }
  auto param = utils::cast<ParameterPtr>(n)->default_param();
  auto tensor = std::dynamic_pointer_cast<ParamValueLite>(param);
  if (tensor == nullptr) {
    return false;
  }
  return tensor->tensor_addr() != nullptr;
}

bool IsConvNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Conv2D || type == schema::PrimitiveType_DepthwiseConv2D;
  }
  return false;
}

bool IsPoolingNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Pooling;
  }
  return false;
}

bool IsQuantNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_QuantDTypeCast;
  }
  return false;
}

bool CheckIsAllInputsParam(const AnfNodePtr &node) {
  if (utils::isa<CNode>(node)) {
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      if (!utils::isa<Parameter>(cnode->input(i))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

size_t GetOutputTensorNum(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return 0;
  }
  auto type = node->Type();
  if (type == nullptr) {
    return 1;
  }
  if (type->isa<Tuple>()) {
    auto tuple_type = type->cast<TuplePtr>();
    if (tuple_type == nullptr) {
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
      return 0;
    }
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
  if (graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(ERROR) << "node has no output in manager";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NOT_FIND_OP);
    return nullptr;
  }
  auto output_info_list = iter->second;
  std::copy(output_info_list.begin(), output_info_list.end(), std::back_inserter(*output_node_list));
  return output_node_list;
}
size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  MS_ASSERT(tuple_get_item != nullptr);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(ERROR) << "The node tuple_get_item must have 2 inputs!";
    return -1;
  }
  auto output_index_value_node = tuple_get_item->input(kInputNodeOutputIndexInTupleGetItem);
  MS_ASSERT(output_index_value_node != nullptr);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_ASSERT(value_node != nullptr);
  return IntToSize(GetValue<int>(value_node->value()));
}
std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedListByOutputIdx(const FuncGraphPtr &graph,
                                                                                        const AnfNodePtr &node,
                                                                                        size_t output_index) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(ERROR) << "node has no output in manager";
    return output_node_list;
  }
  auto output_info_list = iter->second;
  for (const auto &output_info : output_info_list) {
    size_t used_output_index;
    if (GetCNodeType(output_info.first) == schema::PrimitiveType_TupleGetItem) {
      used_output_index = GetTupleGetItemOutIndex(utils::cast<CNodePtr>(output_info.first));
    } else if (GetCNodeType(node) == schema::PrimitiveType_TupleGetItem) {
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
}  // namespace opt
}  // namespace mindspore
