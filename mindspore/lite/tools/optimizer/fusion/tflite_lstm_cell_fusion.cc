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
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include <algorithm>
#include <memory>
#include <functional>
#include "ops/lstm.h"
#include "ops/squeeze.h"
#include "ops/tuple_get_item.h"
#include "src/common/utils.h"
#include "src/param_value_lite.h"
#include "schema/inner/model_generated.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

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
const auto &p1 = std::placeholders::_1;
constexpr float EPSILON = 1e-5;

bool IsParameterNode(const BaseRef &n) { return utils::isa<ParameterPtr>(n); }

bool IsOpType(const BaseRef &n, const PrimitivePtr &prim) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim);
  }
  return false;
}
}  // namespace

STATUS TfliteLstmCellFusion::GetFloatScalarFromParamValueLite(const AnfNodePtr &param_value, float *v) const {
  if (param_value == nullptr || v == nullptr) {
    MS_LOG(ERROR) << "param_value or v is nullptr";
    return RET_ERROR;
  }
  if (!utils::isa<ParameterPtr>(param_value)) {
    MS_LOG(DEBUG) << "param_value is not ParamValueLitePtr";
    return RET_ERROR;
  }
  auto param_ptr = utils::cast<ParameterPtr>(param_value);
  if (!param_ptr->has_default()) {
    MS_LOG(DEBUG) << "param not have default";
    return RET_ERROR;
  }
  auto default_param = param_ptr->default_param();
  if (!utils::isa<ParamValueLitePtr>(default_param)) {
    MS_LOG(DEBUG) << "param_value is not ParamValueLitePtr";
    return RET_ERROR;
  }
  auto default_param_ptr = utils::cast<ParamValueLitePtr>(default_param);
  auto tensor_shape = default_param_ptr->tensor_shape();
  if (!(tensor_shape.size() == 0 || (tensor_shape.size() == 1 && tensor_shape[0] == 1))) {
    MS_LOG(DEBUG) << "default param is not scalar";
    return RET_ERROR;
  }
  if (default_param_ptr->tensor_type() != kNumberTypeFloat32 && default_param_ptr->tensor_type() != kNumberTypeFloat) {
    MS_LOG(DEBUG) << "default param is not float";
    return RET_ERROR;
  }
  *v = *(reinterpret_cast<float *>(default_param_ptr->tensor_addr()));
  return RET_OK;
}

TfliteLstmCellFusion::TfliteLstmCellFusion(const std::string &name, bool multigraph, int input_length, int var_num,
                                           int cond_nodes_num, int cond_cnodes_num, int body_nodes_num,
                                           int body_cnodes_num)
    : PatternProcessPass(name, multigraph) {
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
  for (size_t i = 0; i < this->while_input_var_num_; ++i) {
    while_input_vars_.emplace_back(std::make_shared<Var>());
  }
  cell_zoneout_old_ = std::make_shared<Var>();
  cell_zoneout_new_ = std::make_shared<Var>();
  hidden_zoneout_old_ = std::make_shared<Var>();
  hidden_zoneout_new_ = std::make_shared<Var>();
}

AnfNodePtr TfliteLstmCellFusion::GetCondGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  auto is_parameter1 = std::make_shared<CondVar>(IsParameterNode);
  auto is_parameter2 = std::make_shared<CondVar>(IsParameterNode);
  auto is_parameter3 = std::make_shared<CondVar>(IsParameterNode);
  auto is_parameter4 = std::make_shared<CondVar>(IsParameterNode);
  auto is_less1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess));
  auto is_less2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess));
  auto is_logical_and = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLogicalAnd));
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, kPrimReturn));
  VectorRef less1_ref = VectorRef({is_less1, is_parameter1, is_parameter2});
  VectorRef less2_ref = VectorRef({is_less2, is_parameter3, is_parameter4});
  VectorRef logicaland_ref = VectorRef({is_logical_and, less1_ref, less2_ref});
  VectorRef return_ref = VectorRef({is_return, logicaland_ref});
  VarPtr fg = std::make_shared<Var>("RootG");
  auto pattern = SexpToNode(return_ref, fg, primitive_vars.get(), true);
  return pattern;
}

AnfNodePtr TfliteLstmCellFusion::GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  std::vector<CondVarPtr> placeholders;
  for (int i = 0; i < 20; ++i) {
    placeholders.emplace_back(std::make_shared<CondVar>(IsParameterNode));
  }
  VectorRef add2 = VectorRef({std::make_shared<Var>(), placeholders[2], std::make_shared<CondVar>(IsParameterNode)});
  VectorRef add3 = VectorRef({std::make_shared<Var>(), placeholders[0], std::make_shared<CondVar>(IsParameterNode)});

  VectorRef concat_i_w = VectorRef({std::make_shared<Var>(), placeholders[8], placeholders[12]});
  VectorRef concat_f_w = VectorRef({std::make_shared<Var>(), placeholders[9], placeholders[13]});
  VectorRef concat_c_w = VectorRef({std::make_shared<Var>(), placeholders[10], placeholders[14]});
  VectorRef concat_o_w = VectorRef({std::make_shared<Var>(), placeholders[11], placeholders[15]});

  VectorRef get_item = VectorRef(
    {std::make_shared<Var>("GetItem"), placeholders[7], placeholders[2], std::make_shared<CondVar>(IsParameterNode)});
  VectorRef concat_input_h = VectorRef({std::make_shared<Var>(), get_item, placeholders[5]});

  VectorRef matmul_input = VectorRef({std::make_shared<Var>(), concat_input_h, concat_i_w});
  VectorRef matmul_forget = VectorRef({std::make_shared<Var>(), concat_input_h, concat_f_w});
  VectorRef matmul_cell = VectorRef({std::make_shared<Var>(), concat_input_h, concat_c_w});
  VectorRef matmul_output = VectorRef({std::make_shared<Var>(), concat_input_h, concat_o_w});

  VectorRef bias_input = VectorRef({std::make_shared<Var>(), matmul_input, placeholders[16]});
  VectorRef bias_forget = VectorRef({std::make_shared<Var>(), matmul_forget, placeholders[17]});
  VectorRef bias_cell = VectorRef({std::make_shared<Var>(), matmul_cell, placeholders[18]});
  VectorRef bias_output = VectorRef({std::make_shared<Var>(), matmul_output, placeholders[19]});

  VectorRef cell = VectorRef({std::make_shared<Var>("Tanh"), bias_cell});
  VectorRef input_gate = VectorRef({std::make_shared<Var>("Sigmoid"), bias_input});
  VectorRef cell_input = VectorRef({std::make_shared<Var>("Mul"), input_gate, cell});
  VectorRef forget_gate = VectorRef({std::make_shared<Var>("Sigmoid"), bias_forget});
  VectorRef cell_forgeted = VectorRef({std::make_shared<Var>("Mul"), forget_gate, placeholders[4]});
  VectorRef cell_new = VectorRef({std::make_shared<Var>("Add"), cell_forgeted, cell_input});

  VectorRef zoneout_cell_old = VectorRef({std::make_shared<Var>("Mul"), cell_zoneout_old_, placeholders[4]});
  VectorRef zoneout_cell_new = VectorRef({std::make_shared<Var>("Mul"), cell_zoneout_new_, cell_new});
  VectorRef cell_output = VectorRef({std::make_shared<Var>("Add"), zoneout_cell_new, zoneout_cell_old});

  VectorRef output_gate = VectorRef({std::make_shared<Var>("Sigmoid"), bias_output});
  VectorRef cell_to_output = VectorRef({std::make_shared<Var>("Tanh"), cell_new});
  VectorRef output = VectorRef({std::make_shared<Var>("Mul"), output_gate, cell_to_output});

  VectorRef zoneout_hidden_old = VectorRef({std::make_shared<Var>("Mul"), hidden_zoneout_old_, placeholders[5]});
  VectorRef zoneout_hidden_new = VectorRef({std::make_shared<Var>("Mul"), hidden_zoneout_new_, output});
  VectorRef hidden_output = VectorRef({std::make_shared<Var>("Add"), zoneout_hidden_new, zoneout_hidden_old});

  VectorRef set_item = VectorRef({std::make_shared<Var>("SetItem"), placeholders[3], placeholders[2], output});

  auto is_make_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, kPrimMakeTuple));
  std::vector<BaseRef> outputs = {is_make_tuple, add3, placeholders[1], add2, set_item, cell_output, hidden_output};
  outputs.insert(outputs.end(), placeholders.begin() + 6, placeholders.end());
  VectorRef make_tuple_node = VectorRef(outputs);
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, kPrimReturn));
  VectorRef return_node = VectorRef({is_return, make_tuple_node});

  VarPtr fg = std::make_shared<Var>("RootG");
  auto pattern = SexpToNode(return_node, fg, primitive_vars.get(), true);
  return pattern;
}

const BaseRef TfliteLstmCellFusion::DefinePattern() const {
  auto is_while_node = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimWhile));
  VectorRef while_node = VectorRef({is_while_node});
  auto while_inputs = while_input_vars_;
  while_inputs.insert(while_inputs.begin() + 4, while_input_vars_[2]);
  while_node.insert(while_node.end(), while_inputs.begin(), while_inputs.end());

  auto is_tuple_get_item = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem));
  VectorRef while_output = VectorRef({is_tuple_get_item, while_node, std::make_shared<Var>()});

  auto is_tensor_list_stack = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListStack));
  auto is_parameter = std::make_shared<CondVar>(IsParameterNode);
  VectorRef tensor_list_stack_node = VectorRef({is_tensor_list_stack, while_output, is_parameter});

  return tensor_list_stack_node;
}

EquivPtr TfliteLstmCellFusion::MatchGraph(const FuncGraphPtr &func_graph, const PrimitiveVarMapPtr &primitive_vars,
                                          const AnfNodePtr &pattern) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(pattern != nullptr);
  auto return_node = func_graph->get_return();
  PatternEngine pattern_engine(PatternEngine(std::make_shared<DefaultVisitor>(),
                                             std::function<bool(const BaseRef &, const BaseRef &)>(AnfEqual),
                                             std::function<bool(const BaseRef &, const BaseRef &)>(CNodeTypeEqual)));
  auto empty_equiv = std::make_shared<Equiv>();
  EquivPtr equiv = pattern_engine.Match(pattern, return_node, *primitive_vars, empty_equiv);
  return equiv;
}

// make sure that only 3,4,5 output of while is referenced
bool TfliteLstmCellFusion::CheckReferencedOutputs(const FuncGraphPtr &func_graph, const CNodePtr &while_cnode) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(while_cnode != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return RET_ERROR;
  }
  auto while_node_users = manager->node_users()[while_cnode];
  std::vector<size_t> valid_indexes{3, 4, 5};
  for (auto &node_user : while_node_users) {
    if (!utils::isa<CNodePtr>(node_user.first)) {
      return false;
    }
    auto cnode = utils::cast<CNodePtr>(node_user.first);
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

EquivPtr TfliteLstmCellFusion::CheckSubGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &pattern,
                                             const PrimitiveVarMapPtr &primitive_vars, const AnfNodePtr &anf_sub_graph,
                                             const size_t cnode_num, const size_t all_node_num) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(pattern != nullptr);
  MS_ASSERT(anf_sub_graph != nullptr);
  auto sub_graph = GetValueNode<FuncGraphPtr>(anf_sub_graph);
  auto nodes = TopoSort(sub_graph->get_return());
  auto cnodes = sub_graph->GetOrderedCnodes();
  if (cnodes.size() != cnode_num || nodes.size() != all_node_num) {
    MS_LOG(DEBUG) << "sub graph nodes num not match";
    return nullptr;
  }
  return MatchGraph(sub_graph, primitive_vars, pattern);
}

bool TfliteLstmCellFusion::CheckBodyGraph(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                          const CNodePtr &while_cnode, float *zoneout_cell,
                                          float *zoneout_hidden) const {
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
  if (GetFloatScalarFromParamValueLite(cell_zoneout_old_node, &cell_old) != RET_OK) {
    return false;
  }
  if (GetFloatScalarFromParamValueLite(cell_zoneout_new_node, &cell_new) != RET_OK) {
    return false;
  }
  if (GetFloatScalarFromParamValueLite(hidden_zoneout_old_node, &hidden_old) != RET_OK) {
    return false;
  }
  if (GetFloatScalarFromParamValueLite(hidden_zoneout_new_node, &hidden_new) != RET_OK) {
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
                                              bool is_bias) const {
  MS_ASSERT(new_param != nullptr);
  MS_ASSERT(params.size() == 4);
  std::vector<float *> data_ptrs;
  std::vector<std::vector<int>> data_shapes;
  for (auto &param : params) {
    if (!utils::isa<ParameterPtr>(param)) {
      MS_LOG(DEBUG) << "param is not Parameter node";
      return RET_FAILED;
    }
    auto param_t = utils::cast<ParameterPtr>(param);
    if (!param_t->has_default()) {
      MS_LOG(DEBUG) << "param not have default value";
      return RET_FAILED;
    }
    if (!utils::isa<ParamValueLitePtr>(param_t->default_param())) {
      MS_LOG(DEBUG) << "default value is not ParamValueLite";
      return RET_FAILED;
    }
    auto origin_tensor = std::dynamic_pointer_cast<ParamValueLite>(param_t->default_param());
    if (origin_tensor->tensor_type() != kNumberTypeFloat32 && origin_tensor->tensor_type() != kNumberTypeFloat) {
      MS_LOG(DEBUG) << "origin_tensor is not float32 type";
      return RET_FAILED;
    }
    auto data_ptr = reinterpret_cast<float *>(origin_tensor->tensor_addr());
    auto data_shape = origin_tensor->tensor_shape();
    data_ptrs.push_back(data_ptr);
    data_shapes.push_back(data_shape);
  }

  for (size_t i = 1; i < data_shapes.size(); ++i) {
    if (data_shapes[i] != data_shapes[0]) {
      MS_LOG(DEBUG) << "data shape not same";
      return RET_FAILED;
    }
  }
  auto new_default = std::make_shared<ParamValueLite>();
  if (new_default == nullptr) {
    MS_LOG(ERROR) << "new_default is nullptr";
    return RET_ERROR;
  }
  std::vector<int> new_shape;
  float *tensor_data = nullptr;
  int step = 0;
  int data_size = 0;
  if (is_bias) {
    if (data_shapes[0].size() != 1) {
      MS_LOG(ERROR) << "bias data shape error";
      return RET_ERROR;
    }
    step = data_shapes[0][0];
    data_size = 8 * step;
    new_shape = std::vector<int>({1, data_size});

  } else {
    if (data_shapes[0].size() != 2) {
      MS_LOG(ERROR) << "weight data shape error";
      return RET_ERROR;
    }
    new_shape = std::vector<int>({1, data_shapes[0][0] * 4, data_shapes[0][1]});
    step = data_shapes[0][0] * data_shapes[0][1];
    data_size = 4 * step;
  }

  tensor_data = new (std::nothrow) float[data_size];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    return RET_ERROR;
  }
  for (int i = 0; i < data_size; ++i) {  // bias are stored into first 4*hidden_size buffer, the rest is all 0
    tensor_data[i] = 0.0f;
  }

  for (size_t i = 0; i < data_ptrs.size(); ++i) {
    auto source_len = std::accumulate(data_shapes[i].begin(), data_shapes[i].end(), 1, std::multiplies<int>());
    auto ret = memcpy_s(tensor_data + i * step, step * sizeof(float), data_ptrs[i], source_len * sizeof(float));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error";
      delete[] tensor_data;
      return RET_ERROR;
    }
  }
  new_default->set_tensor_shape(new_shape);
  new_default->set_tensor_type(kNumberTypeFloat32);
  new_default->set_format(schema::Format_NHWC);
  new_default->SetTensorData(tensor_data, data_size * sizeof(float));
  new_param->set_default_param(new_default);

  std::vector<int64_t> shape_vector(new_shape.begin(), new_shape.end());
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor is nullptr";
    return RET_ERROR;
  }
  new_param->set_abstract(abstract_tensor);
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
  lstm_prim->set_bidirectional(false);
  lstm_prim->set_zoneout_cell(zoneout_cell);
  lstm_prim->set_zoneout_hidden(zoneout_hidden);
  auto value_node = NewValueNode(lstm_prim);

  auto &vars = while_input_vars_;

  auto limit1 = utils::cast<AnfNodePtr>((*equiv)[vars[3]]);
  MS_ASSERT(limit1);
  auto limit2 = utils::cast<AnfNodePtr>((*equiv)[vars[7]]);
  MS_ASSERT(limit2);

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
  auto status = GetConcatedParam(i_weights, i_weight, false);
  if (status != RET_OK) {
    return nullptr;
  }
  i_weight->set_name(base_name + "_weight_i");

  std::vector<AnfNodePtr> c_weights{c2i_weight, c2o_weight, c2f_weight, c2c_weight};
  auto c_weight = func_graph->add_parameter();
  status = GetConcatedParam(c_weights, c_weight, false);
  if (status != RET_OK) {
    return nullptr;
  }
  c_weight->set_name(base_name + "_weight_c");

  std::vector<AnfNodePtr> biases{i_bias, o_bias, f_bias, c_bias};
  auto bias = func_graph->add_parameter();
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
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}

CNodePtr TfliteLstmCellFusion::CreateOutputGetItem(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                                   const int item_index) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(get_items != nullptr);
  auto tuple_get_item_prim = std::make_shared<ops::TupleGetItem>();
  auto get_item_value = NewValueNode(MakeValue<int>(item_index));
  if (tuple_get_item_prim == nullptr || get_item_value == nullptr) {
    MS_LOG(ERROR) << "NewValueNode is nullptr";
    return nullptr;
  }
  CNodePtr get_item_cnode = func_graph->NewCNode(tuple_get_item_prim, {node, get_item_value});
  std::vector<int64_t> shape_vector;
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "create abstract_tensor failed";
    return nullptr;
  }
  get_item_cnode->set_abstract(abstract_tensor);
  get_item_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_output_getitem_" +
                                          std::to_string(item_index));
  return get_item_cnode;
}

STATUS TfliteLstmCellFusion::AdjustOtherGetItems(const FuncGraphPtr &func_graph, const CNodePtr &while_cnode,
                                                 const CNodePtr &lstm_cnode, const CNodePtr &output_get_item) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(while_cnode != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return RET_ERROR;
  }
  auto tr = manager->Transact();
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
    auto origin_index = GetValue<int>(value_node->value());
    int new_index = origin_index == 4 ? 2 : 1;
    auto new_index_vnode = NewValueNode(MakeValue<int>(new_index));
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
      tr.SetEdge(get_item_user.first, get_item_user.second, squeeze_node);
    }
  }
  tr.Commit();
  return RET_OK;
}

STATUS TfliteLstmCellFusion::SetAbstractTuple(const CNodePtr &cnode, const int output_num) {
  MS_ASSERT(cnode != nullptr);
  AbstractBasePtrList abstract_list;
  for (int i = 0; i < output_num; ++i) {
    std::vector<int64_t> shape_vector;
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "create abstract_tensor failed";
      return RET_ERROR;
    }
    abstract_list.emplace_back(abstract_tensor);
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
                                                 const std::vector<int> &axis) const {
  MS_ASSERT(func_graph != nullptr);
  auto squeeze_prim = std::make_shared<ops::Squeeze>();
  std::vector<int64_t> axis_vec;
  std::transform(axis.begin(), axis.end(), std::back_inserter(axis_vec),
                 [](int val) { return static_cast<int64_t>(val); });
  squeeze_prim->set_axis(axis_vec);
  auto squeeze_cnode = func_graph->NewCNode(squeeze_prim, {input_node});
  squeeze_cnode->set_abstract(input_node->abstract()->Clone());
  squeeze_cnode->set_fullname_with_scope("squeeze_" + input_node->fullname_with_scope());
  return squeeze_cnode;
}

const AnfNodePtr TfliteLstmCellFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_LOG(DEBUG) << "lstm fusion pass";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
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

  if (CheckIfCNodeIsNull(while_cnode) != RET_OK || CheckInputSize(while_cnode, while_inputs_num_) != RET_OK) {
    return nullptr;
  }
  if (!CheckReferencedOutputs(func_graph, while_cnode)) {
    return nullptr;
  }
  PrimitiveVarMapPtr primitive_vars_cond = std::make_shared<PrimitiveVarMap>();
  auto cond_graph_pattern = GetCondGraphPattern(primitive_vars_cond);
  auto cond_equiv = CheckSubGraph(func_graph, cond_graph_pattern, primitive_vars_cond, while_cnode->input(1),
                                  cond_cnodes_num_, cond_nodes_num_);
  if (cond_equiv == nullptr || cond_equiv->empty()) {
    return nullptr;
  }
  PrimitiveVarMapPtr primitive_vars_body = std::make_shared<PrimitiveVarMap>();
  auto body_graph_pattern = GetBodyGraphPattern(primitive_vars_body);
  auto body_equiv = CheckSubGraph(func_graph, body_graph_pattern, primitive_vars_body, while_cnode->input(2),
                                  body_cnodes_num_, body_nodes_num_);
  if (body_equiv == nullptr || body_equiv->empty()) {
    return nullptr;
  }
  float zoneout_cell = 0.0f;
  float zoneout_hidden = 0.0f;
  if (!CheckBodyGraph(func_graph, body_equiv, while_cnode, &zoneout_cell, &zoneout_hidden)) {
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
  if (squeeze_node == nullptr) {
    return nullptr;
  }

  auto cond_cnode_index_pair = std::make_shared<CNodeIndexPair>(while_cnode, 1);
  func_graph->DropFuncGraphCNodeIndex(cond_cnode_index_pair);
  auto body_cnode_index_pair = std::make_shared<CNodeIndexPair>(while_cnode, 2);
  func_graph->DropFuncGraphCNodeIndex(body_cnode_index_pair);
  MS_LOG(INFO) << "lstm node:" << lstm_node->fullname_with_scope() << " fusion success";
  return squeeze_node;
}
}  // namespace opt
}  // namespace mindspore
