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
#include "src/common/common.h"
#include "frontend/operator/ops.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAnfPrimitiveIndex = 0;
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
  if (primitive_vars == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
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
    MS_LOG(DEBUG) << "The Node is not param.";
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
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto abstract_tensor =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(weight_tensor->tensor_type()), shape_vector);
  bias_parameter->set_abstract(abstract_tensor);

  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->SetTensorData(bias_data, kernel_num * sizeof(float) / sizeof(uint8_t));
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
    MS_LOG(INFO) << "only value node or cnode has type";
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
ParamValueLitePtr GetLiteParamValue(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  if (!utils::isa<ParameterPtr>(node)) {
    if (utils::isa<ValueNodePtr>(node)) {
      auto valueNode = node->cast<ValueNodePtr>();
      auto value = std::dynamic_pointer_cast<ParamValueLite>(valueNode->value());
      if (value != nullptr) {
        return value;
      }
    }
    MS_LOG(DEBUG) << "get lite param value node neither parameternode or valuenode";
    return nullptr;
  }
  auto param = node->cast<ParameterPtr>();
  MS_ASSERT(param != nullptr);
  auto param_value = std::dynamic_pointer_cast<ParamValueLite>(param->default_param());
  return param_value;
}

AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "CNodePtr is nullptr";
    return nullptr;
  }
  auto inputs = cnode->inputs();
  if (!(0 < index && index < inputs.size())) {
    return nullptr;
  }
  auto input = inputs[index];
  if (input == nullptr) {
    MS_LOG(ERROR) << "CNode input is nullptr";
    return nullptr;
  }

  AbstractBasePtr abstract = nullptr;
  if (utils::isa<ParameterPtr>(input)) {
    auto parameter = input->cast<ParameterPtr>();
    abstract = parameter->abstract();
  } else if (utils::isa<CNodePtr>(input)) {
    auto input_cnode = input->cast<CNodePtr>();
    if (GetCNodeType(input_cnode) == schema::PrimitiveType_TupleGetItem) {
      auto tuple_inputs = input_cnode->inputs();
      MS_ASSERT(tuple_inputs.size() == kTupleGetItemInputSize);
      auto get_item_input_cnode = tuple_inputs.at(1);
      MS_ASSERT(get_item_input_cnode != nullptr);
      auto idx = GetTupleGetItemOutIndex(input_cnode);
      if (!utils::isa<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
        MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple";
        return nullptr;
      }
      auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract());
      auto abstract_list = abstract_tuple->elements();
      if (abstract_list.size() <= idx) {
        MS_LOG(ERROR) << "AbstractTuple's size is smaller than expect";
        return nullptr;
      }
      abstract = abstract_list[idx];
    } else {
      abstract = input_cnode->abstract();
    }
  } else {
    MS_LOG(ERROR) << "unsupported input node type";
    return nullptr;
  }
  return abstract;
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

bool IsActivationNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Activation;
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
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return 0;
  }
  if (utils::isa<CNode>(node)) {
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      if (!utils::isa<Parameter>(cnode->input(i)) && !utils::isa<ValueNodePtr>(cnode->input(i))) {
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
  if (node == nullptr || graph == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return 0;
  }
  auto output_node_list = GetRealNodeUsedList(graph, node);
  if (output_node_list == nullptr) {
    MS_LOG(ERROR) << "output node list is nullptr";
    return false;
  }
  if (output_node_list->size() != 1) {
    MS_LOG(DEBUG) << "fusion node has multi output nodes";
    return true;
  }
  return false;
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedList(const FuncGraphPtr &graph,
                                                                             const AnfNodePtr &node) {
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  if (graph == nullptr || node == nullptr) {
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
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_ERROR);
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
  return IntToSize(lite::CastToInt(value_node->value()).front());
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
STATUS GetFilterDim(const std::vector<int32_t> &oriDims, kTransFilterType type, int32_t *filterK, int32_t *filterC,
                    int32_t *filterH, int32_t *filterW) {
  MS_ASSERT(oriDims.size() == 4);
  if (type == kKCHW2HWCK || type == kKCHW2HWKC || type == kKCHW2KHWC || type == kKCHW2CKHW) {
    *filterK = oriDims.at(lite::KCHW_K);
    *filterC = oriDims.at(lite::KCHW_C);
    *filterH = oriDims.at(lite::KCHW_H);
    *filterW = oriDims.at(lite::KCHW_W);
  } else if (type == kCKHW2HWCK || type == kCKHW2HWKC || type == kCKHW2KHWC) {
    *filterC = oriDims.at(lite::CKHW_C);
    *filterK = oriDims.at(lite::CKHW_K);
    *filterH = oriDims.at(lite::CKHW_H);
    *filterW = oriDims.at(lite::CKHW_W);
  } else if (type == kHWCK2KCHW || type == kHWCK2CKHW || type == kHWCK2KHWC) {
    *filterH = oriDims.at(lite::HWCK_H);
    *filterW = oriDims.at(lite::HWCK_W);
    *filterC = oriDims.at(lite::HWCK_C);
    *filterK = oriDims.at(lite::HWCK_K);
  } else if (type == kHWKC2KCHW || type == kHWKC2CKHW || type == kHWKC2KHWC) {
    *filterH = oriDims.at(lite::HWKC_H);
    *filterW = oriDims.at(lite::HWKC_W);
    *filterK = oriDims.at(lite::HWKC_K);
    *filterC = oriDims.at(lite::HWKC_C);
  } else if (type == kNHWC2KCHW || type == kNHWC2HWCK || type == kNHWC2CKHW) {
    *filterK = oriDims.at(lite::NHWC_N);
    *filterH = oriDims.at(lite::NHWC_H);
    *filterW = oriDims.at(lite::NHWC_W);
    *filterC = oriDims.at(lite::NHWC_C);
  } else if (type == kCHWK2HWCK || type == kCHWK2KHWC) {
    *filterC = oriDims.at(lite::CHWK_C);
    *filterH = oriDims.at(lite::CHWK_H);
    *filterW = oriDims.at(lite::CHWK_W);
    *filterK = oriDims.at(lite::CHWK_K);
  } else if (type == kKHWC2HWCK || type == kKHWC2CHWK) {
    *filterK = oriDims.at(lite::KHWC_K);
    *filterH = oriDims.at(lite::KHWC_H);
    *filterW = oriDims.at(lite::KHWC_W);
    *filterC = oriDims.at(lite::KHWC_C);
  } else {
    MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS SetFilterDim(const ParamValueLitePtr &tensor, kTransFilterType type, int32_t filterK, int32_t filterC,
                    int32_t filterH, int32_t filterW) {
  MS_ASSERT(tensor != nullptr);
  if (type == kKCHW2HWCK || type == kCKHW2HWCK || type == kNHWC2HWCK || type == kKHWC2HWCK || type == kCHWK2HWCK) {
    tensor->set_tensor_shape({filterH, filterW, filterC, filterK});
  } else if (type == kKCHW2HWKC || type == kCKHW2HWKC) {
    tensor->set_tensor_shape({filterH, filterW, filterK, filterC});
  } else if (type == kHWCK2KCHW || type == kHWKC2KCHW || type == kNHWC2KCHW) {
    tensor->set_tensor_shape({filterK, filterC, filterH, filterW});
  } else if (type == kHWCK2CKHW || type == kHWKC2CKHW || type == kNHWC2CKHW || type == kKCHW2CKHW) {
    tensor->set_tensor_shape({filterC, filterK, filterH, filterW});
  } else if (type == kKHWC2CHWK) {
    tensor->set_tensor_shape({filterC, filterH, filterW, filterK});
  } else if (type == kKCHW2KHWC || type == kCKHW2KHWC || type == kCHWK2KHWC || type == kHWCK2KHWC ||
             type == kHWKC2KHWC) {
    tensor->set_tensor_shape({filterK, filterH, filterW, filterC});
  } else {
    MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
    return RET_ERROR;
  }
  return RET_OK;
}
template <typename T>
static STATUS TransFilterData(const ParamValueLitePtr &tensor, kTransFilterType type, int32_t filterK, int32_t filterC,
                              int32_t filterH, int32_t filterW) {
  MS_ASSERT(tensor != nullptr);
  int count = filterH * filterW * filterC * filterK;
  if (count <= 0) {
    MS_LOG(ERROR) << "Dim size invalid";
    return RET_ERROR;
  }
  std::unique_ptr<T[]> buf(new (std::nothrow) T[count]);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "new buf failed";
    return RET_ERROR;
  }

  void *originWeightData = tensor->tensor_addr();
  T *weightData = static_cast<T *>(originWeightData);

  if (weightData == nullptr) {
    MS_LOG(ERROR) << "weightData is nullptr";
    return RET_ERROR;
  }
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  switch (type) {
    case kCHWK2HWCK:
    case kCHWK2KHWC: {
      for (int c = 0; c < filterC; ++c) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int k = 0; k < filterK; ++k) {
              p1Buff = weightData + ((c * filterH * filterW * filterK) + (h * filterW * filterK) + (w * filterK) + (k));
              if (type == kCHWK2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kCHWK2KHWC) {
                p2Buff =
                  buf.get() + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kKHWC2HWCK: {
      for (int k = 0; k < filterK; ++k) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int c = 0; c < filterC; ++c) {
              p1Buff = weightData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              p2Buff = buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kKCHW2HWCK:
    case kKCHW2CKHW:
    case kKCHW2KHWC:
    case kKCHW2HWKC: {
      for (int k = 0; k < filterK; ++k) {
        for (int c = 0; c < filterC; ++c) {
          for (int h = 0; h < filterH; ++h) {
            for (int w = 0; w < filterW; ++w) {
              p1Buff = weightData + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              if (type == kKCHW2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kKCHW2KHWC) {
                p2Buff =
                  buf.get() + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              } else if (type == kKCHW2CKHW) {
                p2Buff =
                  buf.get() + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              } else {
                p2Buff =
                  buf.get() + ((h * filterW * filterK * filterC) + (w * filterK * filterC) + (k * filterC) + (c));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kCKHW2HWCK:
    case kCKHW2KHWC:
    case kCKHW2HWKC: {
      for (int c = 0; c < filterC; ++c) {
        for (int k = 0; k < filterK; ++k) {
          for (int h = 0; h < filterH; ++h) {
            for (int w = 0; w < filterW; ++w) {
              p1Buff = weightData + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              if (type == kCKHW2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kCKHW2KHWC) {
                p2Buff =
                  buf.get() + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              } else {
                p2Buff =
                  buf.get() + ((h * filterW * filterK * filterC) + (w * filterK * filterC) + (k * filterC) + (c));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kHWCK2KCHW:
    case kHWCK2CKHW:
    case kHWCK2KHWC: {
      for (int h = 0; h < filterH; ++h) {
        for (int w = 0; w < filterW; ++w) {
          for (int c = 0; c < filterC; ++c) {
            for (int k = 0; k < filterK; ++k) {
              p1Buff = weightData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              if (type == kHWCK2KCHW) {
                p2Buff =
                  buf.get() + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              } else if (type == kHWCK2CKHW) {
                p2Buff =
                  buf.get() + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              } else {
                p2Buff =
                  buf.get() + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kHWKC2KCHW:
    case kHWKC2KHWC:
    case kHWKC2CKHW: {
      for (int h = 0; h < filterH; ++h) {
        for (int w = 0; w < filterW; ++w) {
          for (int c = 0; c < filterC; ++c) {
            for (int k = 0; k < filterK; ++k) {
              p1Buff = weightData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (k * filterC) + (c));
              if (type == kHWKC2KCHW) {
                p2Buff =
                  buf.get() + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              } else {
                p2Buff =
                  buf.get() + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kNHWC2HWCK:
    case kNHWC2KCHW:
    case kNHWC2CKHW: {
      for (int k = 0; k < filterK; ++k) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int c = 0; c < filterC; ++c) {
              p1Buff = weightData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (k * filterC) + (c));
              if (type == kNHWC2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kNHWC2CKHW) {
                p2Buff =
                  buf.get() + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              } else {
                p2Buff =
                  buf.get() + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kKHWC2CHWK: {
      for (int k = 0; k < filterK; ++k) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int c = 0; c < filterC; ++c) {
              p1Buff = weightData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              p2Buff = buf.get() + ((c * filterK * filterH * filterW) + (h * filterK * filterW) + (w * filterK) + (k));
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
      return RET_ERROR;
    }
  }

  auto ret = ::memcpy_s(tensor->tensor_addr(), count * sizeof(T), buf.get(), count * sizeof(T));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

template <typename T>
static STATUS TransFilterFormat(const ParamValueLitePtr &tensor, kTransFilterType type) {
  MS_ASSERT(tensor != nullptr);
  auto oriDims = tensor->tensor_shape();
  if (oriDims.size() != (size_t)lite::DIM_DEFAULT_SIZE) {
    MS_LOG(ERROR) << "Filter dim-num is not supported, dim-num: " << oriDims.size();
    return lite::RET_ERROR;
  }

  int32_t filterH;
  int32_t filterW;
  int32_t filterC;
  int32_t filterK;
  auto status = GetFilterDim(oriDims, type, &filterK, &filterC, &filterH, &filterW);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "GetFilterDim failed: " << status;
    return status;
  }
  status = SetFilterDim(tensor, type, filterK, filterC, filterH, filterW);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "SetFilterDim failed: " << status;
    return status;
  }
  status = TransFilterData<T>(tensor, type, filterK, filterC, filterH, filterW);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "TransFilterData failed: " << status;
    return status;
  }

  return lite::RET_OK;
}

STATUS TransFilterFormat(const ParamValueLitePtr &tensor, schema::Format dst_format) {
  if (tensor == nullptr) {
    return lite::RET_NULL_PTR;
  }
  auto ori_dims = tensor->tensor_shape();
  if (ori_dims.size() != (size_t)lite::DIM_DEFAULT_SIZE) {
    MS_LOG(ERROR) << "Filter dim-num is not supported, dim-num: " << ori_dims.size();
    return lite::RET_ERROR;
  }
  auto src_format = tensor->format();
  auto data_type = tensor->tensor_type();
  lite::STATUS status;
  switch (dst_format) {
    case schema::Format::Format_KHWC: {
      switch (src_format) {
        case schema::Format::Format_KCHW:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKCHW2KHWC);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKCHW2KHWC);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKCHW2KHWC);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kKCHW2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CKHW:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCKHW2KHWC);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCKHW2KHWC);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCKHW2KHWC);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kCKHW2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CHWK:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCHWK2KHWC);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCHWK2KHWC);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCHWK2KHWC);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kCHWK2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_HWCK:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWCK2KHWC);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWCK2KHWC);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWCK2KHWC);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kHWCK2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_HWKC:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWKC2KHWC);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWKC2KHWC);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWKC2KHWC);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kHWKC2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_KHWC:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << src_format << " to " << EnumNameFormat(dst_format);
          return RET_ERROR;
      }
    } break;
    case schema::Format::Format_HWCK: {
      switch (src_format) {
        case schema::Format::Format_KCHW:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKCHW2HWCK);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKCHW2HWCK);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKCHW2HWCK);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kKCHW2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_KHWC:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKHWC2HWCK);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKHWC2HWCK);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKHWC2HWCK);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kKHWC2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CKHW:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCKHW2HWCK);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCKHW2HWCK);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCKHW2HWCK);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kCKHW2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CHWK:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCHWK2HWCK);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCHWK2HWCK);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCHWK2HWCK);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kCHWK2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return lite::RET_ERROR;
          }
          break;
        case schema::Format::Format_HWCK:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << src_format << " to " << EnumNameFormat(dst_format);
          return RET_ERROR;
      }
    } break;
    case schema::Format::Format_KCHW: {
      switch (src_format) {
        case schema::Format::Format_KCHW:
          return RET_OK;
        case schema::Format::Format_HWCK:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWCK2KCHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWCK2KCHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWCK2KCHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kHWCK2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_HWKC:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWKC2KCHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWKC2KCHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWKC2KCHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kHWCK2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_KHWC:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKHWC2KCHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKHWC2KCHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKHWC2KCHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kKHWC2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CKHW:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCKHW2KCHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCKHW2KCHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCKHW2KCHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kCKHW2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CHWK:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCHWK2KCHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCHWK2KCHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCHWK2KCHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kCKHW2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << src_format << " to " << EnumNameFormat(dst_format);
          return RET_ERROR;
      }
    } break;
    case schema::Format::Format_CKHW: {
      switch (src_format) {
        case schema::Format::Format_HWCK:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWCK2CKHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWCK2CKHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWCK2CKHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kHWCK2CKHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_HWKC:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWKC2CKHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWKC2CKHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWKC2CKHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kHWKC2CKHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_KCHW:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKCHW2CKHW);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKCHW2CKHW);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKCHW2CKHW);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kKCHW2CKHW);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CKHW:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << src_format << " to " << EnumNameFormat(dst_format);
          return RET_ERROR;
      }
    } break;
    case schema::Format::Format_CHWK: {
      switch (src_format) {
        case schema::Format::Format_KHWC:
          if (data_type == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKHWC2CHWK);
          } else if (data_type == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKHWC2CHWK);
          } else if (data_type == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKHWC2CHWK);
          } else if (data_type == kNumberTypeFloat16) {
            status = TransFilterFormat<float16>(tensor, kKHWC2CHWK);
          } else {
            MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
            return RET_ERROR;
          }
          break;
        case schema::Format::Format_CHWK:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << src_format << " to " << EnumNameFormat(dst_format);
          return RET_ERROR;
      }
    } break;
    default:
      MS_LOG(ERROR) << "Unsupported transform from " << src_format << " to " << EnumNameFormat(dst_format);
      return RET_ERROR;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "TransFilterData failed: " << status;
    return status;
  }
  return RET_OK;
}
}  // namespace opt
}  // namespace mindspore
