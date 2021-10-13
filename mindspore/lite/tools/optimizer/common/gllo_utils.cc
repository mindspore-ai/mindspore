/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <vector>
#include <utility>
#include <unordered_map>
#include <functional>
#include <string>
#include "base/float16.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/transpose.h"
#include "ops/gather.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/common/tensor_util.h"
#include "frontend/operator/ops.h"
#include "backend/optimizer/common/helper.h"
#include "tools/converter/quant_param_holder.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kDeviceTypeNone = -1;
int DeduceDimConvertion(schema::Format src_format, schema::Format dst_format, std::vector<int> *perm) {
  MS_ASSERT(perm != nullptr);
  auto src_format_str = std::string(schema::EnumNameFormat(src_format));
  auto dst_format_str = std::string(schema::EnumNameFormat(dst_format));
  if (src_format_str.empty() || dst_format_str.empty() || src_format_str.size() != dst_format_str.size()) {
    MS_LOG(ERROR) << "src_format or dst_format is error.";
    return lite::RET_ERROR;
  }
  std::replace(src_format_str.begin(), src_format_str.end(), 'K', 'N');
  std::replace(dst_format_str.begin(), dst_format_str.end(), 'K', 'N');
  perm->clear();
  std::unordered_map<char, int> dim_map;
  for (size_t i = 0; i < src_format_str.size(); ++i) {
    dim_map[src_format_str[i]] = i;
  }
  for (size_t i = 0; i < dst_format_str.size(); ++i) {
    if (dim_map.find(dst_format_str[i]) == dim_map.end()) {
      MS_LOG(ERROR) << "src_format and dst_format cannot match, please check.";
      return RET_ERROR;
    }
    perm->push_back(dim_map[dst_format_str[i]]);
  }
  return lite::RET_OK;
}

template <typename T>
void TransposeData(const ShapeVector &origin_shape, const ShapeVector &cur_shape, const std::vector<int> &perm,
                   T *weight_data, std::vector<T> *buf) {
  MS_ASSERT(weight_data != nullptr && buf != nullptr);
  MS_ASSERT(origin_shape.size() == cur_shape.size() && cur_shape.size() == perm.size());
  int count = std::accumulate(origin_shape.begin(), origin_shape.end(), 1, std::multiplies<int>());
  ShapeVector post_multiply(cur_shape.size());
  std::unordered_map<int, int> dim_map;
  for (int i = cur_shape.size() - 1; i >= 0; --i) {
    if (i == static_cast<int>(cur_shape.size() - 1)) {
      post_multiply[i] = 1;
    } else {
      post_multiply[i] = cur_shape[i + 1] * post_multiply[i + 1];
    }
    dim_map[perm[i]] = i;
  }
  std::unordered_map<int, int> position_map;
  for (int i = 0; i < count; ++i) {
    int temp = i;
    for (int j = static_cast<int>(origin_shape.size()) - 1; j >= 0; --j) {
      MS_ASSERT(origin_shape[j] > 0);
      position_map[j] = temp % origin_shape[j];
      temp /= origin_shape[j];
    }
    int64_t new_pos = std::accumulate(position_map.begin(), position_map.end(), 0,
                                      [&post_multiply, &dim_map](int64_t res, const std::pair<int, int> &pair_y) {
                                        return res + post_multiply[dim_map[pair_y.first]] * pair_y.second;
                                      });
    buf->at(new_pos) = weight_data[i];
  }
}

template <typename T>
STATUS DoTransposeData(const tensor::TensorPtr &tensor, schema::Format src_format, schema::Format dst_format) {
  MS_ASSERT(tensor != nullptr);
  auto origin_shape = tensor->shape_c();
  if (origin_shape.size() != kInputSizeFour) {
    MS_LOG(ERROR) << "Filter dim-num is not supported, dim-num: " << origin_shape.size();
    return lite::RET_ERROR;
  }
  if (std::any_of(origin_shape.begin(), origin_shape.end(), [](int64_t val) { return val <= 0; })) {
    MS_LOG(ERROR) << "the tensor's shape is invalid.";
    return lite::RET_ERROR;
  }
  std::vector<int> perm;
  if (DeduceDimConvertion(src_format, dst_format, &perm) != RET_OK) {
    MS_LOG(ERROR) << "deduce perm failed.";
    return lite::RET_ERROR;
  }
  ShapeVector new_shape;
  for (auto &val : perm) {
    if (val < 0 || static_cast<size_t>(val) >= origin_shape.size()) {
      MS_LOG(ERROR) << "deduce perm is invalid.";
      return lite::RET_ERROR;
    }
    new_shape.push_back(origin_shape[val]);
  }
  auto count = std::accumulate(origin_shape.begin(), origin_shape.end(), 1LL, std::multiplies<int64_t>());
  if (count <= 0 || count > static_cast<int64_t>(INT32_MAX)) {
    MS_LOG(ERROR) << "tensor element num is too big, which should be smaller than int32_max.";
    return RET_ERROR;
  }
  std::vector<T> buf(count);

  void *originWeightData = tensor->data_c();
  MS_CHECK_TRUE_RET(originWeightData != nullptr, RET_ERROR);
  T *weightData = static_cast<T *>(originWeightData);
  TransposeData<T>(origin_shape, new_shape, perm, weightData, &buf);
  if (memcpy_s(tensor->data_c(), tensor->Size(), buf.data(), count * sizeof(T)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return RET_ERROR;
  }
  tensor->set_shape(new_shape);
  return RET_OK;
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
#ifndef ENABLE_SECURITY
  bool is_virtual_node = IsPrimitive(input, prim::kPrimImageSummary) || IsPrimitive(input, prim::kPrimScalarSummary) ||
                         IsPrimitive(input, prim::kPrimTensorSummary) ||
                         IsPrimitive(input, prim::kPrimHistogramSummary) || IsPrimitive(input, prim::kPrimMakeTuple) ||
                         IsPrimitive(input, prim::kPrimStateSetItem) || IsPrimitive(input, prim::kPrimDepend) ||
                         IsPrimitive(input, prim::kPrimTupleGetItem) || IsPrimitive(input, prim::kPrimReturn) ||
                         IsPrimitive(input, prim::kPrimPartial);
#else
  bool is_virtual_node = IsPrimitive(input, prim::kPrimMakeTuple) || IsPrimitive(input, prim::kPrimStateSetItem) ||
                         IsPrimitive(input, prim::kPrimDepend) || IsPrimitive(input, prim::kPrimTupleGetItem) ||
                         IsPrimitive(input, prim::kPrimReturn) || IsPrimitive(input, prim::kPrimPartial);
#endif
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
    auto a_obj = (ops::PrimitiveC *)(a_value_ptr.get());
    auto b_obj = (ops::PrimitiveC *)(b_value_ptr.get());
    return (*a_obj) == (*b_obj);
  } else {
    return (*a_value_ptr) == (*b_value_ptr);
  }
}
}  // namespace

bool CheckInputs(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr.";
    return false;
  }
  if (std::any_of(cnode->inputs().begin(), cnode->inputs().end(),
                  [](const AnfNodePtr &anf_node) { return anf_node == nullptr; })) {
    MS_LOG(ERROR) << "input is nullptr.";
    return false;
  }
  return true;
}

std::vector<int> CastToInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }
  std::vector<int> cur_value = {};
  if (utils::isa<ValueSequeuePtr>(value)) {
    if (!value->cast<ValueSequeuePtr>()->value().empty()) {
      auto data_type = value->cast<ValueSequeuePtr>()->value().front()->type()->number_type();
      if (data_type == kNumberTypeInt64) {
        auto origin_value = GetValue<std::vector<int64_t>>(value);
        std::transform(origin_value.begin(), origin_value.end(), std::back_inserter(cur_value),
                       [](int64_t index) { return static_cast<int>(index); });
      } else if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
        cur_value = GetValue<std::vector<int>>(value);
      } else {
        MS_LOG(ERROR) << "he function only process integer data.";
        return {};
      }
    }
  } else {
    auto data_type = value->type()->number_type();
    if (data_type == kNumberTypeInt64) {
      cur_value.push_back(static_cast<int>(GetValue<int64_t>(value)));
    } else if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
      cur_value.push_back(GetValue<int>(value));
    } else {
      MS_LOG(ERROR) << "the function only process integer data.";
      return {};
    }
  }
  return cur_value;
}

std::vector<std::vector<int>> CastToVec2DInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }

  std::vector<std::vector<int>> result_value;
  if (utils::isa<ValueSequeuePtr>(value)) {
    auto data_type =
      value->cast<ValueSequeuePtr>()->value().front()->cast<ValueSequeuePtr>()->value().front()->type()->number_type();
    if (data_type == kNumberTypeInt64) {
      auto origin_value = GetValue<std::vector<std::vector<int64_t>>>(value);
      for (auto &i : origin_value) {
        std::vector<int> cur_value;
        std::transform(i.begin(), i.end(), std::back_inserter(cur_value),
                       [](int64_t j) { return static_cast<int>(j); });
        result_value.push_back(cur_value);
      }
    } else if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
      result_value = GetValue<std::vector<std::vector<int>>>(value);
    } else {
      MS_LOG(ERROR) << "he function only process integer data.";
      return result_value;
    }
  }
  return result_value;
}

std::vector<float> CastToFloat(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "valueptr is nullptr.";
    return {};
  }
  std::vector<float> cur_value = {};
  if (utils::isa<ValueSequeuePtr>(value)) {
    if (!value->cast<ValueSequeuePtr>()->value().empty()) {
      auto data_type = value->cast<ValueSequeuePtr>()->value().front()->type()->number_type();
      if (data_type == kNumberTypeFloat || data_type == kNumberTypeFloat32) {
        cur_value = GetValue<std::vector<float>>(value);
      } else {
        MS_LOG(ERROR) << "the function only process float data.";
        return {};
      }
    }
  } else {
    auto data_type = value->type()->number_type();
    if (data_type == kNumberTypeFloat || data_type == kNumberTypeFloat32) {
      cur_value.push_back(GetValue<float>(value));
    } else {
      MS_LOG(ERROR) << "the function only process float data.";
      return {};
    }
  }
  return cur_value;
}

bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  if (node == nullptr || primitive_type == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
  } else if (node->isa<ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
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

bool IsOpType(const BaseRef &n, const PrimitivePtr &prim) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim);
  }
  return false;
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

ParameterPtr AddNewBiasNode(float *bias_data, const FuncGraphPtr &func_graph, int kernel_num, TypeId type_id) {
  if (bias_data == nullptr || func_graph == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr.";
    return nullptr;
  }
  auto bias_parameter = func_graph->add_parameter();
  MS_ASSERT(bias_parameter != nullptr);
  std::vector<int64_t> shape_vector = {kernel_num};
  auto tensor_info =
    lite::CreateTensorInfo(bias_data, kernel_num * sizeof(float) / sizeof(uint8_t), shape_vector, type_id);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return nullptr;
  }
  auto status = lite::InitParameterFromTensorInfo(bias_parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }

  return bias_parameter;
}

tensor::TensorPtr GetTensorInfo(const AnfNodePtr &node) {
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  if (!utils::isa<ParameterPtr>(node)) {
    if (utils::isa<ValueNodePtr>(node)) {
      auto valueNode = node->cast<ValueNodePtr>();
      auto value_ptr = valueNode->value();
      MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
      auto value = value_ptr->cast<tensor::TensorPtr>();
      if (value != nullptr) {
        return value;
      }
    }
    MS_LOG(DEBUG) << "get lite param value node neither parameternode or valuenode";
    return nullptr;
  }
  auto param = node->cast<ParameterPtr>();
  MS_ASSERT(param != nullptr);
  if (!param->has_default() || param->default_param() == nullptr) {
    return nullptr;
  }
  auto tensor_info = param->default_param()->cast<tensor::TensorPtr>();
  return tensor_info;
}

AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "CNodePtr is nullptr";
    return nullptr;
  }
  auto inputs = cnode->inputs();
  if (!(index > 0 && index < inputs.size())) {
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
    if (CheckPrimitiveType(input_cnode, prim::kPrimTupleGetItem)) {
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
  auto parameter = utils::cast<ParameterPtr>(n);
  if (!parameter->has_default() || parameter->default_param() == nullptr) {
    return false;
  }
  auto tensor = parameter->default_param()->cast<tensor::TensorPtr>();
  if (tensor == nullptr) {
    return false;
  }
  return tensor->data_c() != nullptr;
}

STATUS GetTensorInfoFromAbstract(tensor::TensorPtr *tensor_info, const CNodePtr &cnode, size_t index) {
  CHECK_NULL_RETURN(tensor_info);
  CHECK_NULL_RETURN(cnode);
  AbstractBasePtr abstract = GetCNodeInputAbstract(cnode, index);
  if (abstract == nullptr) {
    MS_LOG(WARNING) << "Abstract of CNode: " << cnode->fullname_with_scope() << " is nullptr, infershape is delayed.";
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "Abstract of parameter should be abstract tensor";
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  if (!utils::isa<tensor::TensorPtr>(abstract_tensor->GetValueTrack())) {  // input node not complete infershape
    MS_LOG(DEBUG) << "Value of abstract is not tensor::Tensor, indicate that infershape has failed";
    return RET_ERROR;
  }
  *tensor_info = utils::cast<tensor::TensorPtr>(abstract_tensor->GetValueTrack());
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor::Tensor of abstract is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

bool IsParamOrValueNodeWithData(const BaseRef &n) {
  if (utils::isa<ValueNode>(n)) {
    auto value_node = utils::cast<ValueNodePtr>(n);
    auto value = value_node->value();
    if (value != nullptr && value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      if (tensor == nullptr || tensor->data_c() == nullptr) {
        return false;
      }
      return true;
    } else {
      return false;
    }
  }
  if (utils::isa<ParameterPtr>(n)) {
    return IsParamNode(n);
  }
  return false;
}

bool IsParallelSplitConvNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    PrimitivePtr prim;
    if (utils::isa<CNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node->cast<CNodePtr>()->input(kAnfPrimitiveIndex));
    }
    if (utils::isa<ValueNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node);
    }
    if (prim == nullptr) {
      return false;
    }
    int device_type =
      prim->GetAttr(ops::kDeviceType) != nullptr ? GetValue<int32_t>(prim->GetAttr(ops::kDeviceType)) : kDeviceTypeNone;
    if (device_type != kDeviceTypeNone) {
      return false;
    }
    return CheckPrimitiveType(anf_node, prim::kPrimConv2DFusion) || CheckPrimitiveType(anf_node, prim::kPrimConv2D);
  }
  return false;
}

bool IsConvNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    PrimitivePtr prim;
    if (utils::isa<CNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node->cast<CNodePtr>()->input(kAnfPrimitiveIndex));
    }
    if (utils::isa<ValueNodePtr>(anf_node)) {
      prim = GetValueNode<PrimitivePtr>(anf_node);
    }
    if (prim == nullptr) {
      return false;
    }

    if (prim->GetAttr(ops::kActivationType) != nullptr &&
        GetValue<int64_t>(prim->GetAttr(ops::kActivationType)) != NO_ACTIVATION) {
      return false;
    }

    bool is_depth_wise =
      prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
    return CheckPrimitiveType(anf_node, prim::kPrimConv2DFusion) ||
           (CheckPrimitiveType(anf_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise);
  }
  return false;
}

bool CheckIsAllInputsParam(const AnfNodePtr &node) {
  if (node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
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
  if (graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
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

size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  if (tuple_get_item == nullptr || tuple_get_item->size() != kInputSizeThree) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  auto output_index_value_node = tuple_get_item->input(kInputIndexTwo);
  if (output_index_value_node == nullptr) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  auto indexes = CastToInt(value_node->value());
  if (indexes.empty()) {
    MS_LOG(ERROR) << "The node tuple_get_item is invalid.";
    return -1;
  }
  return indexes.front();
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedListByOutputIdx(const FuncGraphPtr &graph,
                                                                                        const AnfNodePtr &node,
                                                                                        size_t output_index) {
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

STATUS TransFilterFormat(const tensor::TensorPtr &tensor, schema::Format src_format, schema::Format dst_format) {
  MS_CHECK_TRUE_RET(tensor != nullptr, RET_ERROR);
  std::unordered_map<TypeId, std::function<STATUS(const tensor::TensorPtr &, schema::Format, schema::Format)>>
    trans_func = {{kNumberTypeFloat32, DoTransposeData<float>},
                  {kNumberTypeUInt8, DoTransposeData<uint8_t>},
                  {kNumberTypeInt8, DoTransposeData<int8_t>},
                  {kNumberTypeFloat16, DoTransposeData<float16>}};
  auto data_type = tensor->data_type();
  auto iter = trans_func.find(data_type);
  if (iter == trans_func.end()) {
    MS_LOG(ERROR) << "Unsupported data_type: " << data_type;
    return RET_ERROR;
  }
  return iter->second(tensor, src_format, dst_format);
}

ParameterPtr BuildParameterNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                const tensor::TensorPtr &tensor_info) {
  if (func_graph == nullptr || node == nullptr || tensor_info == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr.";
    return nullptr;
  }
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  auto shape = tensor_info->shape();
  std::vector<int64_t> shape_vector;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                 [](const int &val) { return static_cast<int64_t>(val); });
  auto data_type = tensor_info->data_type() == kNumberTypeInt64 ? kNumberTypeInt32 : tensor_info->data_type();
  param_node->set_name(node->fullname_with_scope());
  auto tensor_info_new = std::make_shared<tensor::Tensor>(data_type, shape_vector);
  if (tensor_info_new == nullptr) {
    MS_LOG(ERROR) << "new tensor::Tensor failed.";
    return nullptr;
  }
  size_t data_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  if (tensor_info->Size() == 0) {
    auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info_new);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "init parameter from tensor info failed";
      return nullptr;
    }
    return param_node;
  }
  if (tensor_info->data_type() == kNumberTypeInt64) {
    auto *tensor_data = reinterpret_cast<int *>(tensor_info_new->data_c());
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return nullptr;
    }
    auto *origin_data = reinterpret_cast<int64_t *>(tensor_info->data_c());
    for (size_t i = 0; i < data_count; ++i) {
      if (origin_data[i] > static_cast<int64_t>(INT32_MAX) || origin_data[i] < static_cast<int64_t>(INT32_MIN)) {
        MS_LOG(WARNING) << "int64 data " << origin_data[i] << "too big to fit into int32";
        tensor_data[i] = origin_data[i] > 0 ? INT32_MAX : INT32_MIN;
      } else {
        tensor_data[i] = static_cast<int>(origin_data[i]);
      }
    }
  } else {
    tensor_info_new->set_data_type(tensor_info->data_type());
    auto *tensor_data = reinterpret_cast<int8_t *>(tensor_info_new->data_c());
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return nullptr;
    }
    if (memcpy_s(tensor_data, tensor_info_new->Size(), tensor_info->data_c(), tensor_info->Size()) != lite::RET_OK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      return nullptr;
    }
  }
  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info_new);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  param_node->set_default_param(tensor_info_new);
  return param_node;
}

ParameterPtr BuildIntValueParameterNode(const FuncGraphPtr &func_graph, const int32_t &data,
                                        const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  auto tensor_info = lite::CreateTensorInfo(&data, sizeof(int32_t), {1}, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildIntVecParameterNode(const FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                      const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  std::vector<int64_t> shape_vector{static_cast<int64_t>(data.size())};
  auto tensor_info = lite::CreateTensorInfo(data.data(), data.size() * sizeof(int32_t), shape_vector, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }

  return param_node;
}

ParameterPtr BuildIntVec2DParameterNode(const FuncGraphPtr &func_graph, const std::vector<std::vector<int32_t>> &data,
                                        const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  std::vector<int64_t> shape_vector;
  shape_vector.push_back(data.size());
  shape_vector.push_back(2);

  std::vector<int32_t> data_1d;
  for (auto pair : data) {
    data_1d.insert(data_1d.end(), pair.begin(), pair.end());
  }

  auto size = data_1d.size() * sizeof(int32_t);
  auto tensor_info = lite::CreateTensorInfo(data_1d.data(), size, shape_vector, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildFloatValueParameterNode(const FuncGraphPtr &func_graph, const float &data,
                                          const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  auto tensor_info = lite::CreateTensorInfo(&data, sizeof(float), {1}, kNumberTypeFloat32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

ParameterPtr BuildFloatVecParameterNode(const FuncGraphPtr &func_graph, const std::vector<float> &data,
                                        const std::string &node_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);

  std::vector<int64_t> shape_vector{static_cast<int64_t>(data.size())};
  auto tensor_info = lite::CreateTensorInfo(data.data(), data.size() * sizeof(float), shape_vector, kNumberTypeFloat);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }

  return param_node;
}

CNodePtr GenTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &perm,
                          const std::string &cnode_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  auto perm_node = BuildIntVecParameterNode(func_graph, perm, cnode_name + "_perm");
  MS_ASSERT(perm_node != nullptr);
  auto trans_prim = std::make_shared<ops::Transpose>();
  MS_CHECK_TRUE_RET(trans_prim != nullptr, nullptr);
  auto cnode = func_graph->NewCNode(trans_prim, {input_node, perm_node});
  MS_ASSERT(cnode != nullptr);
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(cnode, 1, input_node);
  manager->SetEdge(cnode, kInputIndexTwo, perm_node);
  cnode->set_fullname_with_scope(cnode_name);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(kInputSizeTwo, 1);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, nullptr);
  trans_prim->AddAttr("quant_params", quant_params_holder);
  return cnode;
}

CNodePtr GenGatherNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &indices,
                       const std::string &cnode_name) {
  if (func_graph == nullptr || input_node == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr, which is invalid.";
    return nullptr;
  }
  auto indices_node = BuildIntVecParameterNode(func_graph, indices, cnode_name + "_indices");
  if (indices_node == nullptr) {
    MS_LOG(ERROR) << "make indices node failed.";
    return nullptr;
  }
  auto axis_node = BuildIntVecParameterNode(func_graph, {0}, cnode_name + "_indices");
  if (axis_node == nullptr) {
    MS_LOG(ERROR) << "make indices node failed.";
    return nullptr;
  }
  auto gather_prim = std::make_shared<ops::Gather>();
  MS_CHECK_TRUE_RET(gather_prim != nullptr, nullptr);
  auto cnode = func_graph->NewCNode(gather_prim, {input_node, indices_node, axis_node});
  MS_ASSERT(cnode != nullptr);
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(cnode, 1, input_node);
  manager->SetEdge(cnode, kInputIndexTwo, indices_node);
  manager->SetEdge(cnode, kInputIndexThree, axis_node);
  cnode->set_fullname_with_scope(cnode_name);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(kInputSizeThree, 1);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, nullptr);
  gather_prim->AddAttr("quant_params", quant_params_holder);
  return cnode;
}

CNodePtr GenTupleGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &input, size_t index) {
  if (func_graph == nullptr || input == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr, which is invalid.";
    return nullptr;
  }
  auto tuple_get_item_prim = std::make_shared<lite::TupleGetItem>();
  MS_CHECK_TRUE_RET(tuple_get_item_prim != nullptr, nullptr);
  auto second_input = NewValueNode(MakeValue<int>(index));
  MS_CHECK_TRUE_RET(second_input != nullptr, nullptr);
  auto tuple_cnode = func_graph->NewCNode(tuple_get_item_prim, {input, second_input});
  MS_ASSERT(tuple_cnode != nullptr);
  tuple_cnode->set_fullname_with_scope(input->fullname_with_scope() + "_getitem_" + std::to_string(index));
  return tuple_cnode;
}

STATUS FetchShapeFromAbstract(const abstract::AbstractBasePtr &abstract, ShapeVector *shape) {
  if (abstract == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr, which is invalid.";
    return lite::RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensor>(abstract)) {
    MS_LOG(ERROR) << "abstract of cnode is invalid.";
    return lite::RET_ERROR;
  }
  auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
  if (abstract_tensor->BuildShape() == nullptr || !utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "shape of cnode's output is invalid.";
    return lite::RET_ERROR;
  }
  *shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return lite::RET_OK;
}

bool IsTrainOp(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto cnode_type = prim->type_name();
  // optimizer op
  if (cnode_type == "Adam" || cnode_type == "SGD" || cnode_type == "ApplyMomentum") {
    return true;
  }
  // loss op
  if (cnode_type == "SoftmaxCrossEntropyWithLogits" || cnode_type == "SpareSoftmaxCrossEntropyWithLogits" ||
      cnode_type == "SmoothL1Loss" || cnode_type == "SmoothL1LossGrad" ||
      cnode_type == "SigmoidCrossEntropyWithLogits" || cnode_type == "SigmoidCrossEntropyWithLogpitsGrad") {
    return true;
  }
  // grad op
  if (cnode_type.find("Grad") != std::string::npos ||
      cnode->fullname_with_scope().find("Gradients") != std::string::npos) {
    return true;
  }
  return false;
}

bool IsMarkedTrainOp(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  if (prim->GetAttr("trainOp") != nullptr && GetValue<bool>(prim->GetAttr("trainOp"))) {
    MS_LOG(DEBUG) << "train op not fusion.";
    return true;
  }
  return false;
}

int GetDataTypeFromAnfNode(const AnfNodePtr &anf_node, TypeId *type_id) {
  if (anf_node == nullptr || type_id == nullptr) {
    MS_LOG(ERROR) << "anf_node or type_id is nullptr.";
    return RET_ERROR;
  }
  auto abstract_base = anf_node->abstract();
  // used for multi output e.g. split.
  if (utils::isa<abstract::AbstractTuple>(abstract_base)) {
    auto abstract_tuple = abstract_base->cast<abstract::AbstractTuplePtr>();
    if (abstract_tuple->elements().empty()) {
      MS_LOG(ERROR) << "abstract_tuple elements is empty.";
      return RET_ERROR;
    }
    abstract_base = abstract_tuple->elements().front();
  }
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << anf_node->fullname_with_scope();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << anf_node->fullname_with_scope();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, RET_ERROR, "type_ptr is nullptr");
  *type_id = type_ptr->type_id();
  return RET_OK;
}
}  // namespace opt
}  // namespace mindspore
