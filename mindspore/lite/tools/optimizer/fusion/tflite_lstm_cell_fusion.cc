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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include <memory>
#include <algorithm>
#include <functional>
#include "ops/lstm.h"
#include "ops/squeeze.h"
#include "ops/tuple_get_item.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/helper.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kWhileInputsLength = 23;
constexpr size_t kWhileInputsVarNum = 21;
constexpr size_t kCondNodesNum = 12;
constexpr size_t kCondCNodesNum = 4;
constexpr size_t kBodyNodesNum = 95;
constexpr size_t kBodyCNodesNum = 34;
constexpr size_t kLSTMOutputNum = 3;
constexpr auto kUnidirectionalGateNum = 4;
const auto &p1 = std::placeholders::_1;
constexpr float EPSILON = 1e-5;
bool IsParameterNode(const BaseRef &n) { return utils::isa<ParameterPtr>(n); }

std::vector<VectorRef> GenerateBodyGraphCellPattern(const std::vector<CondVarPtr> &placeholders) {
  MS_CHECK_TRUE_RET(placeholders.size() > 19, {});
  auto is_var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var1 != nullptr, {});
  VectorRef concat_i_w = VectorRef({is_var1, placeholders[8], placeholders[12]});
  auto is_var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var2 != nullptr, {});
  VectorRef concat_f_w = VectorRef({is_var2, placeholders[9], placeholders[13]});
  auto is_var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var3 != nullptr, {});
  VectorRef concat_c_w = VectorRef({is_var3, placeholders[10], placeholders[14]});
  auto is_var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var4 != nullptr, {});
  VectorRef concat_o_w = VectorRef({is_var4, placeholders[11], placeholders[15]});

  auto is_var_getitem = std::make_shared<Var>("GetItem");
  MS_CHECK_TRUE_RET(is_var_getitem != nullptr, {});
  auto is_param3 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, {});
  VectorRef get_item = VectorRef({is_var_getitem, placeholders[7], placeholders[2], is_param3});
  auto is_var5 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var5 != nullptr, {});
  VectorRef concat_input_h = VectorRef({is_var5, get_item, placeholders[5]});

  auto is_var6 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var6 != nullptr, {});
  VectorRef matmul_input = VectorRef({is_var6, concat_input_h, concat_i_w});
  auto is_var7 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var7 != nullptr, {});
  VectorRef matmul_forget = VectorRef({is_var7, concat_input_h, concat_f_w});
  auto is_var8 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var8 != nullptr, {});
  VectorRef matmul_cell = VectorRef({is_var8, concat_input_h, concat_c_w});
  auto is_var9 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var9 != nullptr, {});
  VectorRef matmul_output = VectorRef({is_var9, concat_input_h, concat_o_w});

  auto is_var10 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var10 != nullptr, {});
  VectorRef bias_input = VectorRef({is_var10, matmul_input, placeholders[16]});
  auto is_var11 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var11 != nullptr, {});
  VectorRef bias_forget = VectorRef({is_var11, matmul_forget, placeholders[17]});
  auto is_var12 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var12 != nullptr, {});
  VectorRef bias_cell = VectorRef({is_var12, matmul_cell, placeholders[18]});
  auto is_var13 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var13 != nullptr, {});
  VectorRef bias_output = VectorRef({is_var13, matmul_output, placeholders[19]});

  auto is_var_tanh = std::make_shared<Var>("Tanh");
  MS_CHECK_TRUE_RET(is_var_tanh != nullptr, {});
  VectorRef cell = VectorRef({is_var_tanh, bias_cell});
  auto is_var_sigmoid1 = std::make_shared<Var>("Sigmoid");
  MS_CHECK_TRUE_RET(is_var_sigmoid1 != nullptr, {});
  VectorRef input_gate = VectorRef({is_var_sigmoid1, bias_input});
  auto is_var_mul1 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul1 != nullptr, {});
  VectorRef cell_input = VectorRef({is_var_mul1, input_gate, cell});
  auto is_var_sigmoid2 = std::make_shared<Var>("Sigmoid");
  MS_CHECK_TRUE_RET(is_var_sigmoid2 != nullptr, {});
  VectorRef forget_gate = VectorRef({is_var_sigmoid2, bias_forget});
  auto is_var_mul2 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul2 != nullptr, {});
  VectorRef cell_forgeted = VectorRef({is_var_mul2, forget_gate, placeholders[4]});
  auto is_var_add = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add != nullptr, {});
  VectorRef cell_new = VectorRef({is_var_add, cell_forgeted, cell_input});
  return {bias_output, cell_new};
}
}  // namespace

STATUS TfliteLstmCellFusion::GetFloatScalarFromTensorInfo(const AnfNodePtr &tensor_info, float *v) {
  if (tensor_info == nullptr || v == nullptr) {
    MS_LOG(ERROR) << "tensor_info or v is nullptr";
    return RET_ERROR;
  }
  if (!utils::isa<ParameterPtr>(tensor_info)) {
    MS_LOG(DEBUG) << "tensor_info is not tensor::TensorPtr";
    return RET_ERROR;
  }
  auto param_ptr = utils::cast<ParameterPtr>(tensor_info);
  if (!param_ptr->has_default() || param_ptr->default_param() == nullptr) {
    MS_LOG(DEBUG) << "param not have default";
    return RET_ERROR;
  }
  auto default_param = param_ptr->default_param();
  if (!utils::isa<tensor::TensorPtr>(default_param)) {
    MS_LOG(DEBUG) << "tensor_info is not tensor::TensorPtr";
    return RET_ERROR;
  }
  auto default_param_ptr = utils::cast<tensor::TensorPtr>(default_param);
  auto tensor_shape = default_param_ptr->shape();
  if (!(tensor_shape.empty() || (tensor_shape.size() == 1 && tensor_shape[0] == 1))) {
    MS_LOG(DEBUG) << "default param is not scalar";
    return RET_ERROR;
  }
  if (default_param_ptr->data_type() != kNumberTypeFloat32 && default_param_ptr->data_type() != kNumberTypeFloat) {
    MS_LOG(DEBUG) << "default param is not float";
    return RET_ERROR;
  }
  *v = *(reinterpret_cast<float *>(default_param_ptr->data_c()));
  return RET_OK;
}

bool TfliteLstmCellFusion::Init() const {
  for (size_t i = 0; i < this->while_input_var_num_; ++i) {
    auto is_var = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var != nullptr, false);
    while_input_vars_.emplace_back(is_var);
  }
  cell_zoneout_old_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(cell_zoneout_old_ != nullptr, false);
  cell_zoneout_new_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(cell_zoneout_new_ != nullptr, false);
  hidden_zoneout_old_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(hidden_zoneout_old_ != nullptr, false);
  hidden_zoneout_new_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(hidden_zoneout_new_ != nullptr, false);
  return true;
}

TfliteLstmCellFusion::TfliteLstmCellFusion(const std::string &name, bool multigraph, int input_length, int var_num,
                                           int cond_nodes_num, int cond_cnodes_num, int body_nodes_num,
                                           int body_cnodes_num)
    : LitePatternProcessPass(name, multigraph) {
  /*
   * input vars for lstm while node
   * 0:cond_ 1:body_ 2:time_ 3:limit1_ 4:output_ 5:cell_ 6:hidden_ 7:limit2_ 8:input_
   * 9:i2i_  10:i2f_ 11:i2c_ 12:i2o_   13:c2i_   14:c2f_ 15:c2c_   16:c2o_   17:i_bias_ 18:f_bias_ 19:c_bias_ 20:o_bias_
   */
  this->while_inputs_num_ = input_length == 0 ? kWhileInputsLength : input_length;
  this->while_input_var_num_ = var_num == 0 ? kWhileInputsVarNum : var_num;
  this->cond_nodes_num_ = cond_nodes_num == 0 ? kCondNodesNum : cond_nodes_num;
  this->cond_cnodes_num_ = cond_cnodes_num == 0 ? kCondCNodesNum : cond_cnodes_num;
  this->body_nodes_num_ = body_nodes_num == 0 ? kBodyNodesNum : body_nodes_num;
  this->body_cnodes_num_ = body_cnodes_num == 0 ? kBodyCNodesNum : body_cnodes_num;
}

AnfNodePtr TfliteLstmCellFusion::GetCondGraphPattern(const PrimitiveVarMapPtr &primitive_vars) {
  MS_ASSERT(primitive_vars != nullptr);
  auto is_parameter1 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_parameter1 != nullptr, nullptr);
  auto is_parameter2 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_parameter2 != nullptr, nullptr);
  auto is_parameter3 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_parameter3 != nullptr, nullptr);
  auto is_parameter4 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_parameter4 != nullptr, nullptr);
  auto is_less1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess));
  MS_CHECK_TRUE_RET(is_less1 != nullptr, nullptr);
  auto is_less2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess));
  MS_CHECK_TRUE_RET(is_less2 != nullptr, nullptr);
  auto is_logical_and = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLogicalAnd));
  MS_CHECK_TRUE_RET(is_logical_and != nullptr, nullptr);
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReturn));
  MS_CHECK_TRUE_RET(is_return != nullptr, nullptr);
  VectorRef less1_ref = VectorRef({is_less1, is_parameter1, is_parameter2});
  VectorRef less2_ref = VectorRef({is_less2, is_parameter3, is_parameter4});
  VectorRef logicaland_ref = VectorRef({is_logical_and, less1_ref, less2_ref});
  VectorRef return_ref = VectorRef({is_return, logicaland_ref});
  VarPtr fg = std::make_shared<Var>("RootG");
  MS_CHECK_TRUE_RET(fg != nullptr, nullptr);
  auto pattern = Helper::SexpToNode(return_ref, fg, primitive_vars.get(), true);
  return pattern;
}

AnfNodePtr TfliteLstmCellFusion::GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  std::vector<CondVarPtr> placeholders;
  for (int i = 0; i < 20; ++i) {
    placeholders.emplace_back(std::make_shared<CondVar>(IsParameterNode));
  }
  auto is_var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var1 != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, nullptr);
  VectorRef add2 = VectorRef({is_var1, placeholders[2], is_param1});
  auto is_var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var2 != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, nullptr);
  VectorRef add3 = VectorRef({is_var2, placeholders[0], is_param2});

  auto hidden_cells = GenerateBodyGraphCellPattern(placeholders);
  MS_CHECK_TRUE_RET(hidden_cells.size() == kInputSizeTwo, {});
  auto is_var_mul1 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul1 != nullptr, {});
  VectorRef zoneout_cell_old = VectorRef({is_var_mul1, cell_zoneout_old_, placeholders[4]});
  auto is_var_mul2 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul2 != nullptr, {});
  auto cell_new = hidden_cells[1];
  MS_CHECK_TRUE_RET(!cell_new.empty(), {});
  VectorRef zoneout_cell_new = VectorRef({is_var_mul2, cell_zoneout_new_, cell_new});
  auto is_var_add1 = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add1 != nullptr, {});
  VectorRef cell_output = VectorRef({is_var_add1, zoneout_cell_new, zoneout_cell_old});

  auto is_var_sigmoid = std::make_shared<Var>("Sigmoid");
  MS_CHECK_TRUE_RET(is_var_sigmoid != nullptr, {});
  auto bias_output = hidden_cells[0];
  MS_CHECK_TRUE_RET(!bias_output.empty(), {});
  VectorRef output_gate = VectorRef({is_var_sigmoid, bias_output});
  auto is_var_tanh = std::make_shared<Var>("Tanh");
  MS_CHECK_TRUE_RET(is_var_tanh != nullptr, {});
  VectorRef cell_to_output = VectorRef({is_var_tanh, cell_new});
  auto is_var_mul3 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul3 != nullptr, {});
  VectorRef output = VectorRef({is_var_mul3, output_gate, cell_to_output});

  auto is_var_mul4 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul4 != nullptr, {});
  VectorRef zoneout_hidden_old = VectorRef({is_var_mul4, hidden_zoneout_old_, placeholders[5]});
  auto is_var_mul5 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul5 != nullptr, {});
  VectorRef zoneout_hidden_new = VectorRef({is_var_mul5, hidden_zoneout_new_, output});
  auto is_var_add2 = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add2 != nullptr, {});
  VectorRef hidden_output = VectorRef({is_var_add2, zoneout_hidden_new, zoneout_hidden_old});

  auto is_var_setitem = std::make_shared<Var>("SetItem");
  MS_CHECK_TRUE_RET(is_var_setitem != nullptr, {});
  VectorRef set_item = VectorRef({is_var_setitem, placeholders[3], placeholders[2], output});

  auto is_make_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMakeTuple));
  MS_CHECK_TRUE_RET(is_make_tuple != nullptr, nullptr);
  std::vector<BaseRef> outputs = {is_make_tuple, add3, placeholders[1], add2, set_item, cell_output, hidden_output};
  outputs.insert(outputs.end(), placeholders.begin() + 6, placeholders.end());
  VectorRef make_tuple_node = VectorRef(outputs);
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReturn));
  MS_CHECK_TRUE_RET(is_return != nullptr, nullptr);
  VectorRef return_node = VectorRef({is_return, make_tuple_node});

  VarPtr fg = std::make_shared<Var>("RootG");
  MS_CHECK_TRUE_RET(fg != nullptr, nullptr);
  auto pattern = Helper::SexpToNode(return_node, fg, primitive_vars.get(), true);
  return pattern;
}

const BaseRef TfliteLstmCellFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }
  auto is_while_node = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimWhile));
  MS_CHECK_TRUE_RET(is_while_node != nullptr, {});
  VectorRef while_node = VectorRef({is_while_node});
  auto while_inputs = while_input_vars_;
  MS_CHECK_TRUE_RET(while_inputs.size() > kInputSizeThree, {});
  while_inputs.insert(while_inputs.begin() + 4, while_input_vars_[2]);
  while_node.insert(while_node.end(), while_inputs.begin(), while_inputs.end());

  auto is_tuple_get_item = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem));
  MS_CHECK_TRUE_RET(is_tuple_get_item != nullptr, {});
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  VectorRef while_output = VectorRef({is_tuple_get_item, while_node, is_var});

  auto is_tensor_list_stack = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListStack));
  MS_CHECK_TRUE_RET(is_tensor_list_stack != nullptr, {});
  auto is_parameter = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_parameter != nullptr, {});
  VectorRef tensor_list_stack_node = VectorRef({is_tensor_list_stack, while_output, is_parameter});

  return tensor_list_stack_node;
}

EquivPtr TfliteLstmCellFusion::MatchGraph(const FuncGraphPtr &func_graph, const PrimitiveVarMapPtr &primitive_vars,
                                          const AnfNodePtr &pattern) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(pattern != nullptr);
  auto return_node = func_graph->get_return();
  auto visitor = std::make_shared<Visitor>();
  MS_CHECK_TRUE_RET(visitor != nullptr, nullptr);
  PatternEngine pattern_engine(visitor);
  auto empty_equiv = std::make_shared<Equiv>();
  MS_CHECK_TRUE_RET(empty_equiv != nullptr, nullptr);
  EquivPtr equiv = pattern_engine.Match(pattern, return_node, *primitive_vars, empty_equiv);
  return equiv;
}

// make sure that only 3,4,5 output of while is referenced
bool TfliteLstmCellFusion::CheckReferencedOutputs(const FuncGraphPtr &func_graph, const CNodePtr &while_cnode) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(while_cnode != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return false;
  }
  auto while_node_users = manager->node_users()[while_cnode];
  std::vector<size_t> valid_indexes{3, 4, 5};
  for (auto &node_user : while_node_users) {
    if (!utils::isa<CNodePtr>(node_user.first)) {
      return false;
    }
    auto cnode = utils::cast<CNodePtr>(node_user.first);
    if (IsMarkedTrainOp(cnode)) {
      return false;
    }
    if (!CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
      return false;
    }
    auto index = GetTupleGetItemOutIndex(cnode);
    if (!lite::IsContain(valid_indexes, index)) {
      return false;
    }
  }
  return true;
}

EquivPtr TfliteLstmCellFusion::CheckSubGraph(const AnfNodePtr &pattern, const PrimitiveVarMapPtr &primitive_vars,
                                             const AnfNodePtr &anf_sub_graph, const size_t cnode_num,
                                             const size_t all_node_num) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(pattern != nullptr);
  MS_ASSERT(primitive_vars != nullptr);
  MS_ASSERT(anf_sub_graph != nullptr);
  auto sub_graph = GetValueNode<FuncGraphPtr>(anf_sub_graph);
  MS_CHECK_TRUE_RET(sub_graph != nullptr, nullptr);
  auto nodes = TopoSort(sub_graph->get_return());
  auto cnodes = sub_graph->GetOrderedCnodes();
  if (cnodes.size() != cnode_num || nodes.size() != all_node_num) {
    MS_LOG(DEBUG) << "sub graph nodes num not match";
    return nullptr;
  }
  return MatchGraph(sub_graph, primitive_vars, pattern);
}

bool TfliteLstmCellFusion::CheckBodyGraph(const EquivPtr &equiv, float *zoneout_cell, float *zoneout_hidden) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(while_cnode != nullptr);
  MS_ASSERT(zoneout_cell != nullptr);
  MS_ASSERT(zoneout_hidden != nullptr);

  auto cell_zoneout_old_node = utils::cast<AnfNodePtr>((*equiv)[cell_zoneout_old_]);
  MS_ASSERT(cell_zoneout_old_node != nullptr);
  auto cell_zoneout_new_node = utils::cast<AnfNodePtr>((*equiv)[cell_zoneout_new_]);
  MS_ASSERT(cell_zoneout_new_node != nullptr);
  auto hidden_zoneout_old_node = utils::cast<AnfNodePtr>((*equiv)[hidden_zoneout_old_]);
  MS_ASSERT(hidden_zoneout_old_node != nullptr);
  auto hidden_zoneout_new_node = utils::cast<AnfNodePtr>((*equiv)[hidden_zoneout_new_]);
  MS_ASSERT(hidden_zoneout_new_node != nullptr);

  float cell_old, cell_new, hidden_old, hidden_new;
  if (GetFloatScalarFromTensorInfo(cell_zoneout_old_node, &cell_old) != RET_OK) {
    return false;
  }
  if (GetFloatScalarFromTensorInfo(cell_zoneout_new_node, &cell_new) != RET_OK) {
    return false;
  }
  if (GetFloatScalarFromTensorInfo(hidden_zoneout_old_node, &hidden_old) != RET_OK) {
    return false;
  }
  if (GetFloatScalarFromTensorInfo(hidden_zoneout_new_node, &hidden_new) != RET_OK) {
    return false;
  }
  if (cell_old < 0.0f || cell_old > 1.0f || cell_new < 0.0f || cell_new > 1.0f) {
    MS_LOG(DEBUG) << "cell zoneout value illegal";
    return false;
  }
  if (hidden_old < 0.0f || hidden_old > 1.0f || hidden_new < 0.0f || hidden_new > 1.0f) {
    MS_LOG(DEBUG) << "hidden zoneout value illegal";
    return false;
  }
  if (std::abs(cell_old + cell_new - 1.0f) > EPSILON || std::abs(hidden_old + hidden_new - 1.0f) > EPSILON ||
      std::abs(cell_old - hidden_old) > EPSILON) {
    MS_LOG(DEBUG) << "zoneout value illegal";
    return false;
  }
  *zoneout_cell = cell_old;
  *zoneout_hidden = hidden_old;
  return true;
}

STATUS TfliteLstmCellFusion::GetConcatedParam(const std::vector<AnfNodePtr> &params, const ParameterPtr &new_param,
                                              bool is_bias) {
  MS_ASSERT(new_param != nullptr);
  MS_ASSERT(params.size() == 4);
  std::vector<float *> data_ptrs;
  std::vector<std::vector<int64_t>> data_shapes;
  for (auto &param : params) {
    if (!utils::isa<ParameterPtr>(param)) {
      MS_LOG(DEBUG) << "param is not Parameter node";
      return RET_FAILED;
    }
    auto param_t = utils::cast<ParameterPtr>(param);
    if (!param_t->has_default() || param_t->default_param() == nullptr) {
      MS_LOG(DEBUG) << "param not have default value";
      return RET_FAILED;
    }
    if (!utils::isa<tensor::TensorPtr>(param_t->default_param())) {
      MS_LOG(DEBUG) << "default value is not tensor::Tensor";
      return RET_FAILED;
    }
    auto origin_tensor = std::dynamic_pointer_cast<tensor::Tensor>(param_t->default_param());
    if (origin_tensor->data_type() != kNumberTypeFloat32 && origin_tensor->data_type() != kNumberTypeFloat) {
      MS_LOG(DEBUG) << "origin_tensor is not float32 type";
      return RET_FAILED;
    }
    auto data_ptr = reinterpret_cast<float *>(origin_tensor->data_c());
    auto data_shape = origin_tensor->shape();
    data_ptrs.push_back(data_ptr);
    data_shapes.push_back(data_shape);
  }

  for (size_t i = 1; i < data_shapes.size(); ++i) {
    if (data_shapes[i] != data_shapes[0]) {
      MS_LOG(DEBUG) << "data shape not same";
      return RET_FAILED;
    }
  }
  std::vector<int64_t> new_shape;
  int step = 0;
  int data_size = 0;
  MS_ASSERT(!data_shapes.empty());
  if (is_bias) {
    if (data_shapes[0].size() != 1) {
      MS_LOG(ERROR) << "bias data shape error";
      return RET_ERROR;
    }
    step = static_cast<int>(data_shapes[0][0]);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(C8NUM, step, RET_ERROR);
    data_size = C8NUM * step;
    new_shape = std::vector<int64_t>({1, data_size});

  } else {
    if (data_shapes[0].size() != 2) {
      MS_LOG(ERROR) << "weight data shape error";
      return RET_ERROR;
    }
    new_shape = std::vector<int64_t>({1, data_shapes[0][0] * kUnidirectionalGateNum, data_shapes[0][1]});
    MS_CHECK_INT_MUL_NOT_OVERFLOW(data_shapes[0][0], data_shapes[0][1], RET_ERROR);
    step = static_cast<int>(data_shapes[0][0] * data_shapes[0][1]);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(C4NUM, step, RET_ERROR);
    data_size = C4NUM * step;
  }

  auto tensor_info = lite::CreateTensorInfo(nullptr, 0, new_shape, kNumberTypeFloat32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }

  auto tensor_data = static_cast<float *>(tensor_info->data_c());
  for (int i = 0; i < data_size; ++i) {  // bias are stored into first 4*hidden_size buffer, the rest is all 0
    tensor_data[i] = 0.0f;
  }

  for (size_t i = 0; i < data_ptrs.size(); ++i) {
    auto source_len = std::accumulate(data_shapes[i].begin(), data_shapes[i].end(), 1, std::multiplies<int>());
    auto ret = memcpy_s(tensor_data + i * step, step * sizeof(float), data_ptrs[i], source_len * sizeof(float));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error";
      return RET_ERROR;
    }
  }

  auto status = lite::InitParameterFromTensorInfo(new_param, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }

  return RET_OK;
}

CNodePtr TfliteLstmCellFusion::CreateLSTMNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const EquivPtr &body_equiv, const std::string &base_name,
                                              const float zoneout_cell, const float zoneout_hidden) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(body_equiv != nullptr);
  /*
   * input vars for while node
   * 0:cond_ 1:body_ 2:time_ 3:limit1_ 4:output_ 5:cell_ 6:hidden_ 7:limit2_ 8:input_
   * 9:i2i_  10:i2f_ 11:i2c_ 12:i2o_   13:c2i_   14:c2f_ 15:c2c_   16:c2o_   17:i_bias_ 18:f_bias_ 19:c_bias_ 20:o_bias_
   */
  auto lstm_prim = std::make_shared<ops::LSTM>();
  MS_CHECK_TRUE_RET(lstm_prim != nullptr, nullptr);
  auto lstm_prim_c = lstm_prim->GetPrim();
  MS_CHECK_TRUE_RET(lstm_prim_c != nullptr, nullptr);
  lstm_prim->set_bidirectional(false);
  lstm_prim->set_zoneout_cell(zoneout_cell);
  lstm_prim->set_zoneout_hidden(zoneout_hidden);
  auto value_node = NewValueNode(lstm_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);

  auto &vars = while_input_vars_;
  auto i2i_weight = utils::cast<AnfNodePtr>((*equiv)[vars[9]]);
  MS_ASSERT(i2i_weight);
  auto i2f_weight = utils::cast<AnfNodePtr>((*equiv)[vars[10]]);
  MS_ASSERT(i2f_weight);
  auto i2c_weight = utils::cast<AnfNodePtr>((*equiv)[vars[11]]);
  MS_ASSERT(i2c_weight);
  auto i2o_weight = utils::cast<AnfNodePtr>((*equiv)[vars[12]]);
  MS_ASSERT(i2o_weight);

  auto c2i_weight = utils::cast<AnfNodePtr>((*equiv)[vars[13]]);
  MS_ASSERT(c2i_weight);
  auto c2f_weight = utils::cast<AnfNodePtr>((*equiv)[vars[14]]);
  MS_ASSERT(c2f_weight);
  auto c2c_weight = utils::cast<AnfNodePtr>((*equiv)[vars[15]]);
  MS_ASSERT(c2c_weight);
  auto c2o_weight = utils::cast<AnfNodePtr>((*equiv)[vars[16]]);
  MS_ASSERT(c2o_weight);

  auto i_bias = utils::cast<AnfNodePtr>((*equiv)[vars[17]]);
  MS_ASSERT(i_bias);
  auto f_bias = utils::cast<AnfNodePtr>((*equiv)[vars[18]]);
  MS_ASSERT(f_bias);
  auto c_bias = utils::cast<AnfNodePtr>((*equiv)[vars[19]]);
  MS_ASSERT(c_bias);
  auto o_bias = utils::cast<AnfNodePtr>((*equiv)[vars[20]]);
  MS_ASSERT(o_bias);

  auto input = utils::cast<AnfNodePtr>((*equiv)[vars[8]]);
  MS_ASSERT(input);
  auto cell = utils::cast<AnfNodePtr>((*equiv)[vars[5]]);
  MS_ASSERT(cell);
  auto hidden = utils::cast<AnfNodePtr>((*equiv)[vars[6]]);
  MS_ASSERT(hidden);

  std::vector<AnfNodePtr> i_weights{i2i_weight, i2o_weight, i2f_weight, i2c_weight};
  auto i_weight = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(i_weight != nullptr, nullptr);
  auto status = GetConcatedParam(i_weights, i_weight, false);
  if (status != RET_OK) {
    return nullptr;
  }
  i_weight->set_name(base_name + "_weight_i");

  std::vector<AnfNodePtr> c_weights{c2i_weight, c2o_weight, c2f_weight, c2c_weight};
  auto c_weight = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(c_weight != nullptr, nullptr);
  status = GetConcatedParam(c_weights, c_weight, false);
  if (status != RET_OK) {
    return nullptr;
  }
  c_weight->set_name(base_name + "_weight_c");

  std::vector<AnfNodePtr> biases{i_bias, o_bias, f_bias, c_bias};
  auto bias = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(bias != nullptr, nullptr);
  status = GetConcatedParam(biases, bias, true);
  if (status != RET_OK) {
    return nullptr;
  }
  bias->set_name(base_name + "_bias");

  if (!utils::isa<CNodePtr>(input) || !CheckPrimitiveType(input, prim::kPrimTensorListFromTensor)) {
    MS_LOG(DEBUG) << "input is not tensorlistfromtensor op";
    return nullptr;
  }
  auto tensor_list_cnode = utils::cast<CNodePtr>(input);
  auto input_tensor_node = tensor_list_cnode->input(1);

  std::vector<AnfNodePtr> new_node_inputs = {value_node, input_tensor_node, i_weight, c_weight, bias, hidden, cell};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}

CNodePtr TfliteLstmCellFusion::CreateOutputGetItem(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                                   const int item_index) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto tuple_get_item_prim = std::make_shared<ops::TupleGetItem>();
  auto get_item_value = NewValueNode(MakeValue<int>(item_index));
  if (tuple_get_item_prim == nullptr || get_item_value == nullptr) {
    MS_LOG(ERROR) << "NewValueNode is nullptr";
    return nullptr;
  }
  auto tuple_get_item_prim_c = tuple_get_item_prim->GetPrim();
  MS_ASSERT(tuple_get_item_prim_c != nullptr);
  CNodePtr get_item_cnode = func_graph->NewCNode(tuple_get_item_prim_c, {node, get_item_value});
  MS_CHECK_TRUE_RET(get_item_cnode != nullptr, nullptr);
  auto abstract = lite::CreateTensorAbstract({}, kNumberTypeFloat32);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return nullptr;
  }
  get_item_cnode->set_abstract(abstract);
  get_item_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_output_getitem_" +
                                          std::to_string(item_index));
  return get_item_cnode;
}

STATUS TfliteLstmCellFusion::AdjustOtherGetItems(const FuncGraphPtr &func_graph, const CNodePtr &while_cnode,
                                                 const CNodePtr &lstm_cnode, const CNodePtr &output_get_item) {
  MS_ASSERT(func_graph != nullptr && while_cnode != nullptr);
  MS_ASSERT(lstm_cnode != nullptr && output_get_item != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return RET_ERROR;
  }
  auto while_node_users = manager->node_users()[while_cnode];
  for (auto &node_user : while_node_users) {
    if (node_user.first == output_get_item) {
      continue;
    }
    if (!utils::isa<CNodePtr>(node_user.first)) {
      return RET_ERROR;
    }
    auto get_item = utils::cast<CNodePtr>(node_user.first);
    if (!CheckPrimitiveType(get_item, prim::kPrimTupleGetItem)) {
      return RET_ERROR;
    }
    auto new_inputs = get_item->inputs();
    if (new_inputs.size() != 3) {
      return RET_ERROR;
    }
    new_inputs[1] = lstm_cnode;
    auto index_vnode = get_item->input(2);
    if (!utils::isa<ValueNode>(index_vnode)) {
      MS_LOG(ERROR) << "TupleGetItem's input 2 is not value node";
      return RET_ERROR;
    }
    auto value_node = utils::cast<ValueNodePtr>(index_vnode);
    if (value_node == nullptr) {
      MS_LOG(ERROR) << "cast to ValueNode failed";
      return RET_ERROR;
    }
    auto origin_index = value_node->value()->type()->number_type() == kNumberTypeInt64
                          ? GetValue<int64_t>(value_node->value())
                          : GetValue<int>(value_node->value());
    int new_index = origin_index == 4 ? 2 : 1;
    auto new_index_vnode = NewValueNode(MakeValue<int>(new_index));
    MS_CHECK_TRUE_RET(new_index_vnode != nullptr, RET_ERROR);
    new_inputs[2] = new_index_vnode;
    get_item->set_inputs(new_inputs);
    get_item->set_fullname_with_scope(lstm_cnode->fullname_with_scope() + "_getitem_" + std::to_string(new_index));
    if (get_item->abstract() == nullptr) {
      MS_LOG(ERROR) << "get_item's abstract is nullptr";
      return RET_ERROR;
    }

    std::vector<int> squeeze_axis{0};
    auto squeeze_node = CreateSqueezeNode(func_graph, get_item, squeeze_axis);
    if (squeeze_node == nullptr) {
      return RET_ERROR;
    }

    auto get_item_users = manager->node_users()[get_item];
    for (auto &get_item_user : get_item_users) {
      manager->SetEdge(get_item_user.first, get_item_user.second, squeeze_node);
    }
  }
  return RET_OK;
}

STATUS TfliteLstmCellFusion::SetAbstractTuple(const CNodePtr &cnode, const int output_num) {
  MS_ASSERT(cnode != nullptr);
  AbstractBasePtrList abstract_list;
  for (int i = 0; i < output_num; ++i) {
    auto abstract = lite::CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    abstract_list.emplace_back(abstract);
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  if (abstract_tuple == nullptr) {
    MS_LOG(ERROR) << "create abstract_tuple failed";
    return RET_ERROR;
  }
  cnode->set_abstract(abstract_tuple);
  return RET_OK;
}

CNodePtr TfliteLstmCellFusion::CreateSqueezeNode(const FuncGraphPtr &func_graph, const CNodePtr &input_node,
                                                 const std::vector<int> &axis) {
  MS_ASSERT(func_graph != nullptr && input_node != nullptr);
  auto squeeze_prim = std::make_shared<ops::Squeeze>();
  MS_CHECK_TRUE_RET(squeeze_prim != nullptr, nullptr);
  auto squeeze_prim_c = squeeze_prim->GetPrim();
  MS_CHECK_TRUE_RET(squeeze_prim_c != nullptr, nullptr);
  std::vector<int64_t> axis_vec;
  std::transform(axis.begin(), axis.end(), std::back_inserter(axis_vec),
                 [](int val) { return static_cast<int64_t>(val); });
  squeeze_prim->set_axis(axis_vec);
  auto squeeze_cnode = func_graph->NewCNode(squeeze_prim_c, {input_node});
  MS_CHECK_TRUE_RET(squeeze_cnode != nullptr, nullptr);
  if (input_node->abstract() != nullptr) {
    squeeze_cnode->set_abstract(input_node->abstract()->Clone());
  }
  squeeze_cnode->set_fullname_with_scope("squeeze_" + input_node->fullname_with_scope());
  return squeeze_cnode;
}

const AnfNodePtr TfliteLstmCellFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto tensor_list_stack_cnode = utils::cast<CNodePtr>(node);
  auto tuple_get_item_node = tensor_list_stack_cnode->input(1);
  if (!utils::isa<CNodePtr>(tuple_get_item_node)) {
    return nullptr;
  }
  auto tuple_get_item_cnode = utils::cast<CNodePtr>(tuple_get_item_node);
  auto while_node = tuple_get_item_cnode->input(1);
  if (!utils::isa<CNodePtr>(while_node)) {
    return nullptr;
  }
  auto while_cnode = utils::cast<CNodePtr>(while_node);

  if (while_cnode == nullptr || while_cnode->size() != while_inputs_num_) {
    return nullptr;
  }
  if (!CheckReferencedOutputs(func_graph, while_cnode)) {
    return nullptr;
  }
  auto primitive_vars_cond = std::make_shared<PrimitiveVarMap>();
  MS_CHECK_TRUE_RET(primitive_vars_cond != nullptr, nullptr);
  auto cond_graph_pattern = GetCondGraphPattern(primitive_vars_cond);
  MS_CHECK_TRUE_RET(cond_graph_pattern != nullptr, nullptr);
  auto cond_equiv =
    CheckSubGraph(cond_graph_pattern, primitive_vars_cond, while_cnode->input(1), cond_cnodes_num_, cond_nodes_num_);
  if (cond_equiv == nullptr || cond_equiv->empty()) {
    return nullptr;
  }
  auto primitive_vars_body = std::make_shared<PrimitiveVarMap>();
  MS_CHECK_TRUE_RET(primitive_vars_body != nullptr, nullptr);
  auto body_graph_pattern = GetBodyGraphPattern(primitive_vars_body);
  MS_CHECK_TRUE_RET(body_graph_pattern != nullptr, nullptr);
  auto body_equiv =
    CheckSubGraph(body_graph_pattern, primitive_vars_body, while_cnode->input(2), body_cnodes_num_, body_nodes_num_);
  if (body_equiv == nullptr || body_equiv->empty()) {
    return nullptr;
  }
  float zoneout_cell = 0.0f;
  float zoneout_hidden = 0.0f;
  if (!CheckBodyGraph(body_equiv, &zoneout_cell, &zoneout_hidden)) {
    return nullptr;
  }
  const std::string lstm_name = "lstm_" + while_cnode->fullname_with_scope();
  auto lstm_node = CreateLSTMNode(func_graph, equiv, body_equiv, lstm_name, zoneout_cell, zoneout_hidden);
  if (lstm_node == nullptr) {
    return nullptr;
  }
  auto status = SetAbstractTuple(lstm_node, kLSTMOutputNum);
  if (status != RET_OK) {
    return nullptr;
  }

  auto get_item_node = CreateOutputGetItem(func_graph, lstm_node, 0);
  if (get_item_node == nullptr) {
    MS_LOG(DEBUG) << "create lstm output get_item node failed";
    return nullptr;
  }

  status = AdjustOtherGetItems(func_graph, while_cnode, lstm_node, tuple_get_item_cnode);
  if (status != RET_OK) {
    return nullptr;
  }

  std::vector<int> squeeze_axis{1};  // our lstm output:0 have an extra axis that tflite not have, it must be squeezed
  auto squeeze_node = CreateSqueezeNode(func_graph, get_item_node, squeeze_axis);
  MS_CHECK_TRUE_MSG(squeeze_node != nullptr, nullptr, "create a squeeze node failed.");

  auto cond_cnode_index_pair = std::make_shared<CNodeIndexPair>(while_cnode, 1);
  MS_CHECK_TRUE_RET(cond_cnode_index_pair != nullptr, nullptr);
  func_graph->DropFuncGraphCNodeIndex(cond_cnode_index_pair);
  auto body_cnode_index_pair = std::make_shared<CNodeIndexPair>(while_cnode, 2);
  MS_CHECK_TRUE_RET(body_cnode_index_pair != nullptr, nullptr);
  func_graph->DropFuncGraphCNodeIndex(body_cnode_index_pair);
  MS_LOG(INFO) << "lstm node:" << lstm_node->fullname_with_scope() << " fusion success";
  return squeeze_node;
}
}  // namespace opt
}  // namespace mindspore
