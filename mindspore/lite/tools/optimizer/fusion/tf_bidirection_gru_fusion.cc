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
#include "tools/optimizer/fusion/tf_bidirection_gru_fusion.h"
#include <memory>
#include <functional>
#include "ops/concat.h"
#include "ops/gru.h"
#include "ops/split.h"
#include "ops/squeeze.h"
#include "ops/stack.h"
#include "ops/transpose.h"
#include "src/common/utils.h"
#include "utils/utils.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCondNodesNum = 12;
constexpr size_t kCondCNodesNum = 4;
constexpr size_t kBodyNodesNum = 69;
constexpr size_t kBodyCNodesNum = 25;
const auto &p1 = std::placeholders::_1;
}  // namespace

TfBidirectionGruFusion::TfBidirectionGruFusion(int num_fw_vars, int num_bw_vars, const std::string &name,
                                               bool multi_graph)
    : PatternProcessPass(name, multi_graph) {
  /*
   * vars for while input
   * fw_while_inputs:
   * 0:cond 1:body 2:kernel_gate 3:bias_gate 4:cand_kernel 5:cand_bias
   * bw_while_inputs:
   * 0:cond 1:body 2:kernel_gate 3:bias_gate 4:cand_kernel 5:cand_bias
   */
  for (int i = 0; i < num_fw_vars; ++i) {
    fw_vars_.emplace_back(std::make_shared<Var>());
  }
  for (int i = 0; i < num_bw_vars; ++i) {
    bw_vars_.emplace_back(std::make_shared<Var>());
  }
  input_ = std::make_shared<Var>();
  input_length_ = std::make_shared<Var>();
  transpose_input_ = std::make_shared<Var>();
  fw_init_state_ = std::make_shared<Var>();
  bw_init_state_ = std::make_shared<Var>();
}

const BaseRef TfBidirectionGruFusion::DefinePattern() const {
  // forward
  auto fw_reduce = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion)),
                              input_length_, std::make_shared<CondVar>(IsParameterNode)});
  auto fw_max = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMaximum)),
                           std::make_shared<CondVar>(IsParameterNode), fw_reduce});

  auto fw_shape = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimShape)), transpose_input_});
  auto fw_stride = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice)), fw_shape,
                              std::make_shared<SeqVar>()});
  auto fw_min = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMinimum)), fw_stride, fw_max});

  auto fw_reserve = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListReserve)),
                               std::make_shared<CondVar>(IsParameterNode), fw_stride});
  auto fw_from_tensor = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListFromTensor)),
                                   transpose_input_, std::make_shared<CondVar>(IsParameterNode)});
  auto is_fw_while = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimWhile));
  auto fw_while = VectorRef({is_fw_while, fw_vars_[0], fw_vars_[1], std::make_shared<CondVar>(IsParameterNode),
                             fw_stride, std::make_shared<CondVar>(IsParameterNode), fw_reserve, fw_init_state_, fw_min,
                             fw_from_tensor, input_length_});
  fw_while.insert(fw_while.end(), fw_vars_.begin() + 2, fw_vars_.end());
  fw_while.emplace_back(std::make_shared<Var>());
  auto fw_get_item = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem)), fw_while, std::make_shared<Var>()});
  auto fw_stack = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListStack)),
                             fw_get_item, std::make_shared<CondVar>(IsParameterNode)});
  auto fw_out_trans = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), fw_stack, std::make_shared<Var>()});

  // backward
  auto bw_reverse_seq =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReverseSequence)), input_, input_length_});
  auto bw_max1 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion)), input_length_,
                            std::make_shared<CondVar>(IsParameterNode)});
  auto bw_max2 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMaximum)),
                            std::make_shared<CondVar>(IsParameterNode), bw_max1});
  auto bw_trans = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), bw_reverse_seq,
                             std::make_shared<Var>()});
  auto bw_shape = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimShape)), bw_trans});
  auto bw_stride = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice)), bw_shape,
                              std::make_shared<SeqVar>()});
  auto bw_min = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMinimum)), bw_stride, bw_max2});
  auto bw_reserve = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListReserve)),
                               std::make_shared<CondVar>(IsParameterNode), bw_stride});
  auto bw_from_tensor = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListFromTensor)),
                                   bw_trans, std::make_shared<CondVar>(IsParameterNode)});
  auto is_bw_while = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimWhile));
  auto bw_while = VectorRef({is_bw_while, bw_vars_[0], bw_vars_[1], std::make_shared<CondVar>(IsParameterNode),
                             bw_stride, std::make_shared<CondVar>(IsParameterNode), bw_reserve, bw_init_state_, bw_min,
                             bw_from_tensor, input_length_});
  bw_while.insert(bw_while.end(), bw_vars_.begin() + 2, bw_vars_.end());
  bw_while.emplace_back(std::make_shared<Var>());
  auto bw_get_item = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem)), bw_while, std::make_shared<Var>()});
  auto bw_stack = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListStack)),
                             bw_get_item, std::make_shared<CondVar>(IsParameterNode)});
  auto bw_out_trans = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose)), bw_stack, std::make_shared<Var>()});
  auto bw_reverse1 = VectorRef(
    {std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReverseSequence)), bw_out_trans, input_length_});

  auto concat =
    VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimConcat)), fw_out_trans, bw_reverse1});
  return concat;
}

AnfNodePtr TfBidirectionGruFusion::GetCondGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
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

AnfNodePtr TfBidirectionGruFusion::GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  std::vector<CondVarPtr> placeholders;
  for (int i = 0; i < 13; ++i) {
    placeholders.emplace_back(std::make_shared<CondVar>(IsParameterNode));
  }
  VectorRef add = VectorRef({std::make_shared<Var>(), placeholders[2], std::make_shared<CondVar>(IsParameterNode)});
  VectorRef add1 = VectorRef({std::make_shared<Var>(), placeholders[0], std::make_shared<CondVar>(IsParameterNode)});

  VectorRef get_item = VectorRef(
    {std::make_shared<Var>("GetItem"), placeholders[6], placeholders[2], std::make_shared<CondVar>(IsParameterNode)});
  VectorRef concat_input_h = VectorRef({std::make_shared<Var>(), get_item, placeholders[4]});

  VectorRef matmul1 = VectorRef({std::make_shared<Var>("Matmul"), concat_input_h, placeholders[8]});
  VectorRef biasadd1 = VectorRef({std::make_shared<Var>("BiasAdd"), matmul1, placeholders[9]});
  VectorRef sigmoid1 = VectorRef({std::make_shared<Var>("Sigmoid"), biasadd1});

  VectorRef split = VectorRef({std::make_shared<Var>("Split"), sigmoid1});
  VectorRef get_item1 = VectorRef({std::make_shared<Var>("TupleGetItem"), split, std::make_shared<Var>()});
  VectorRef get_item2 = VectorRef({std::make_shared<Var>("TupleGetItem"), split, std::make_shared<Var>()});

  VectorRef pre_reset = VectorRef({std::make_shared<Var>("Mul"), get_item1, placeholders[4]});
  VectorRef concat2 = VectorRef({std::make_shared<Var>("Concat"), get_item, pre_reset});
  VectorRef matmul2 = VectorRef({std::make_shared<Var>("Matmul"), concat2, placeholders[10]});
  VectorRef biasadd2 = VectorRef({std::make_shared<Var>("BiasAdd"), matmul2, placeholders[11]});
  VectorRef tanh = VectorRef({std::make_shared<Var>("Tanh"), biasadd2});

  VectorRef update_hidden = VectorRef({std::make_shared<Var>("Mul"), get_item2, placeholders[4]});
  VectorRef minus_update =
    VectorRef({std::make_shared<Var>("Sub"), std::make_shared<CondVar>(IsParameterNode), get_item2});
  VectorRef updated = VectorRef({std::make_shared<Var>("Mul"), minus_update, tanh});

  VectorRef new_hidden = VectorRef({std::make_shared<Var>("Add"), update_hidden, updated});

  VectorRef greater_equal = VectorRef({std::make_shared<Var>("GreaterEqual"), placeholders[2], placeholders[7]});

  VectorRef select_output = VectorRef({std::make_shared<Var>("Switch"), greater_equal, placeholders[12], new_hidden});
  VectorRef output = VectorRef({std::make_shared<Var>("SetItem"), placeholders[3], placeholders[2], select_output});

  VectorRef select_hidden = VectorRef({std::make_shared<Var>("Switch"), greater_equal, placeholders[4], new_hidden});

  auto is_make_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, kPrimMakeTuple));
  std::vector<BaseRef> outputs = {is_make_tuple,  add1,          placeholders[1], add,
                                  output,         select_hidden, placeholders[5], placeholders[6],
                                  placeholders[7]};
  outputs.insert(outputs.end(), placeholders.begin() + 8, placeholders.end());
  VectorRef make_tuple_node = VectorRef(outputs);
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, kPrimReturn));
  VectorRef return_node = VectorRef({is_return, make_tuple_node});

  VarPtr fg = std::make_shared<Var>("RootG");
  auto pattern = SexpToNode(return_node, fg, primitive_vars.get(), true);
  return pattern;
}

ParamValueLitePtr TfBidirectionGruFusion::GetDefaultParamValue(const AnfNodePtr &parameter_anf) const {
  MS_ASSERT(parameter_anf != nullptr);
  if (!utils::isa<ParameterPtr>(parameter_anf)) {
    MS_LOG(DEBUG) << "parameter_anf is not ParameterPtr";
    return nullptr;
  }
  auto parameter = utils::cast<ParameterPtr>(parameter_anf);
  if (!parameter->has_default()) {
    MS_LOG(DEBUG) << "parameter not have default value";
    return nullptr;
  }
  auto param_value = std::dynamic_pointer_cast<ParamValueLite>(parameter->default_param());
  return param_value;
}

STATUS TfBidirectionGruFusion::GetInputAndHiddenSize(const AnfNodePtr &fw_cand_kernel_anf,
                                                     const AnfNodePtr &bw_cand_kernel_anf, int *input_size,
                                                     int *hidden_size) const {
  MS_ASSERT(fw_cand_kernel != nullptr);
  MS_ASSERT(bw_cand_kernel != nullptr);
  MS_ASSERT(input_size != nullptr);
  MS_ASSERT(hidden_size != nullptr);
  auto fw_cand_kernel_value = GetDefaultParamValue(fw_cand_kernel_anf);
  if (fw_cand_kernel_value == nullptr) {
    return RET_ERROR;
  }
  auto fw_cand_kernel_shape = fw_cand_kernel_value->tensor_shape();
  if (fw_cand_kernel_shape.size() != 2) {
    return RET_ERROR;
  }
  auto bw_cand_kernel_value = GetDefaultParamValue(bw_cand_kernel_anf);
  if (bw_cand_kernel_value == nullptr) {
    return RET_ERROR;
  }
  auto bw_cand_kernel_shape = bw_cand_kernel_value->tensor_shape();
  if (bw_cand_kernel_shape.size() != 2) {
    return RET_ERROR;
  }
  if (fw_cand_kernel_shape != bw_cand_kernel_shape) {
    return RET_ERROR;
  }
  if (fw_cand_kernel_shape[1] <= 0 || fw_cand_kernel_shape[0] - fw_cand_kernel_shape[1] <= 0) {
    MS_LOG(DEBUG) << "gru input size or hidden size illegal";
    return RET_ERROR;
  }
  *hidden_size = fw_cand_kernel_shape[1];
  *input_size = fw_cand_kernel_shape[0] - fw_cand_kernel_shape[1];
  return RET_OK;
}

ParameterPtr TfBidirectionGruFusion::AddDefaultParameter(const FuncGraphPtr &func_graph, const std::string &name,
                                                         const std::vector<int> &shape, const TypeId type,
                                                         void **tensor_data) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(tensor_data != nullptr);
  auto parameter = func_graph->add_parameter();
  parameter->set_name(name);
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type), shape_vector);
  if (abstract_tensor == nullptr) {
    return nullptr;
  }
  parameter->set_abstract(abstract_tensor);

  auto gate_weight_default = std::make_shared<ParamValueLite>();
  if (gate_weight_default == nullptr) {
    MS_LOG(ERROR) << "gate_weight_default is nullptr";
    return nullptr;
  }
  gate_weight_default->set_tensor_shape(shape);
  gate_weight_default->set_tensor_type(type);
  gate_weight_default->set_format(schema::Format_NHWC);
  int data_len = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  int data_size = 0;
  if (type == kNumberTypeFloat32 || type == kNumberTypeFloat) {
    data_size = data_len * sizeof(float);
    *tensor_data = new (std::nothrow) float[data_len];
  } else if (type == kNumberTypeInt || type == kNumberTypeInt32) {
    data_size = data_len * sizeof(int);
    *tensor_data = new (std::nothrow) int[data_len];
  } else {
    MS_LOG(DEBUG) << "unsupported data type";
    return nullptr;
  }
  if (*tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    return nullptr;
  }

  gate_weight_default->SetTensorData(*tensor_data, data_size);
  parameter->set_default_param(gate_weight_default);
  return parameter;
}

void TfBidirectionGruFusion::CopyFlattenMatData(const float *mat, const int R, const int C, const int r0, const int r1,
                                                const int c0, const int c1, float *data, bool t) const {
  MS_ASSERT(mat != nullptr);
  MS_ASSERT(data != nullptr);
  MS_ASSERT(0 <= r0 && r0 < r1 && r1 <= R);
  MS_ASSERT(0 <= c0 && c0 < c1 && c1 <= C);
  const int RT = r1 - r0;
  const int CT = c1 - c0;
  for (int i = r0; i < r1; ++i) {
    for (int j = c0; j < c1; ++j) {
      if (t) {
        data[(j - c0) * RT + (i - r0)] = mat[i * C + j];
      } else {
        data[(i - r0) * CT + (j - c0)] = mat[i * C + j];
      }
    }
  }
}

STATUS TfBidirectionGruFusion::ConvertWeightData(const AnfNodePtr &gate_weight, const AnfNodePtr &cand_weight,
                                                 const int input_size, const int hidden_size, float *gate_tensor_data,
                                                 float *recu_tensor_data) const {
  MS_ASSERT(gate_weight != nullptr);
  MS_ASSERT(cand_weight != nullptr);
  MS_ASSERT(gate_tensor_data != nullptr);
  MS_ASSERT(recu_tensor_data != nullptr);
  const std::vector<int> gate_shape{input_size + hidden_size, hidden_size * 2};
  const std::vector<int> cand_shape{hidden_size * 2, hidden_size};
  auto gate_weight_value = GetDefaultParamValue(gate_weight);
  if (gate_weight_value == nullptr) {
    return RET_ERROR;
  }
  auto gate_weight_data = reinterpret_cast<float *>(gate_weight_value->tensor_addr());
  if (gate_weight_data == nullptr) {
    return RET_ERROR;
  }
  auto gate_weight_shape = gate_weight_value->tensor_shape();

  auto cand_weight_value = GetDefaultParamValue(cand_weight);
  if (cand_weight_value == nullptr) {
    return RET_ERROR;
  }
  auto cand_weight_data = reinterpret_cast<float *>(cand_weight_value->tensor_addr());
  if (cand_weight_data == nullptr) {
    return RET_ERROR;
  }
  auto cand_weight_shape = cand_weight_value->tensor_shape();

  if (gate_weight_shape != gate_shape || cand_weight_shape != cand_shape) {
    return RET_ERROR;
  }

  // input_update_weight
  CopyFlattenMatData(gate_weight_data, input_size + hidden_size, hidden_size * 2, 0, input_size, hidden_size,
                     hidden_size * 2, gate_tensor_data, true);
  // input_reset_weight
  CopyFlattenMatData(gate_weight_data, input_size + hidden_size, hidden_size * 2, 0, input_size, 0, hidden_size,
                     gate_tensor_data + input_size * hidden_size, true);
  // input_hidden_weight
  CopyFlattenMatData(cand_weight_data, input_size + hidden_size, hidden_size, 0, input_size, 0, hidden_size,
                     gate_tensor_data + input_size * hidden_size * 2, true);

  // state_update_weight
  CopyFlattenMatData(gate_weight_data, input_size + hidden_size, hidden_size * 2, input_size, input_size + hidden_size,
                     hidden_size, hidden_size * 2, recu_tensor_data, true);
  // state_reset_weight
  CopyFlattenMatData(gate_weight_data, input_size + hidden_size, hidden_size * 2, input_size, input_size + hidden_size,
                     0, hidden_size, recu_tensor_data + hidden_size * hidden_size, true);
  // state_hidden_weight
  CopyFlattenMatData(cand_weight_data, input_size + hidden_size, hidden_size, input_size, input_size + hidden_size, 0,
                     hidden_size, recu_tensor_data + hidden_size * hidden_size * 2, true);
  return RET_OK;
}

STATUS TfBidirectionGruFusion::ConvertBiasData(const AnfNodePtr &gate_bias, const AnfNodePtr &cand_bias,
                                               const int hidden_size, float *tensor_data) const {
  MS_ASSERT(bias != nullptr);
  MS_ASSERT(tensor_data != nullptr);
  std::vector<int> gate_shape{hidden_size * 2};
  std::vector<int> cand_shape{hidden_size};
  auto gate_bias_value = GetDefaultParamValue(gate_bias);
  if (gate_bias_value == nullptr) {
    return RET_ERROR;
  }
  auto gate_bias_data = reinterpret_cast<float *>(gate_bias_value->tensor_addr());
  auto gate_bias_shape = gate_bias_value->tensor_shape();
  auto cand_bias_value = GetDefaultParamValue(cand_bias);
  if (cand_bias_value == nullptr) {
    return RET_ERROR;
  }
  auto cand_bias_data = reinterpret_cast<float *>(cand_bias_value->tensor_addr());
  auto cand_bias_shape = cand_bias_value->tensor_shape();
  if (gate_bias_shape != gate_shape || cand_bias_shape != cand_shape) {
    return RET_ERROR;
  }

  // update_gate bias
  CopyFlattenMatData(gate_bias_data, 1, hidden_size * 2, 0, 1, hidden_size, hidden_size * 2, tensor_data, false);
  // reset_gate bias
  CopyFlattenMatData(gate_bias_data, 1, hidden_size * 2, 0, 1, 0, hidden_size, tensor_data + hidden_size, false);
  // hidden_gate bias
  CopyFlattenMatData(cand_bias_data, 1, hidden_size, 0, 1, 0, hidden_size, tensor_data + hidden_size * 2, false);

  return RET_OK;
}

CNodePtr TfBidirectionGruFusion::GetStackedHiddenState(const FuncGraphPtr &func_graph, const AnfNodePtr &fw_init_state,
                                                       const AnfNodePtr &bw_init_state,
                                                       const std::string base_name) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(fw_init_state != nullptr);
  MS_ASSERT(bw_init_state != nullptr);
  auto stack_prim = std::make_shared<ops::Stack>();
  stack_prim->set_axis(0);
  auto value_node = NewValueNode(stack_prim);
  std::vector<AnfNodePtr> new_node_inputs = {value_node, fw_init_state, bw_init_state};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  new_node->set_abstract(fw_init_state->abstract()->Clone());
  new_node->set_fullname_with_scope("stack_hidden_" + base_name);
  return new_node;
}

CNodePtr TfBidirectionGruFusion::CreateBiDirectionGruNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                          const EquivPtr &equiv, const std::string &base_name,
                                                          int var_offset) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(input != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto gru_prim = std::make_shared<ops::GRU>();
  gru_prim->set_bidirectional(true);
  auto value_node = NewValueNode(gru_prim);

  auto fw_gate_kernel = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[var_offset]]);
  MS_ASSERT(fw_gate_kernel != nullptr);
  auto fw_gate_bias = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[var_offset + 1]]);
  MS_ASSERT(fw_gate_bias != nullptr);
  auto fw_cand_kernel = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[var_offset + 2]]);
  MS_ASSERT(fw_cand_kernel != nullptr);
  auto fw_cand_bias = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[var_offset + 3]]);
  MS_ASSERT(fw_cand_bias != nullptr);

  auto bw_gate_kernel = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[var_offset]]);
  MS_ASSERT(bw_gate_kernel != nullptr);
  auto bw_gate_bias = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[var_offset + 1]]);
  MS_ASSERT(bw_gate_bias != nullptr);
  auto bw_cand_kernel = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[var_offset + 2]]);
  MS_ASSERT(bw_cand_kernel != nullptr);
  auto bw_cand_bias = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[var_offset + 3]]);
  MS_ASSERT(bw_cand_bias != nullptr);

  auto fw_init_state = utils::cast<AnfNodePtr>((*equiv)[fw_init_state_]);
  MS_ASSERT(fw_init_state != nullptr);
  auto bw_init_state = utils::cast<AnfNodePtr>((*equiv)[bw_init_state_]);
  MS_ASSERT(bw_init_state != nullptr);
  auto stacked_hidden = GetStackedHiddenState(func_graph, fw_init_state, bw_init_state, base_name);
  if (stacked_hidden == nullptr) {
    return nullptr;
  }
  auto input_length = utils::cast<AnfNodePtr>((*equiv)[input_length_]);
  MS_ASSERT(hidden != nullptr);

  int input_size = 0;
  int hidden_size = 0;
  auto status = GetInputAndHiddenSize(fw_cand_kernel, bw_cand_kernel, &input_size, &hidden_size);
  if (status != RET_OK) {
    return nullptr;
  }
  std::vector<int> gate_weight_shape{2, hidden_size * 3, input_size};
  float *gate_tensor_data = nullptr;
  auto gate_weight = AddDefaultParameter(func_graph, base_name + "_gate_weight", gate_weight_shape, kNumberTypeFloat32,
                                         reinterpret_cast<void **>(&gate_tensor_data));
  if (gate_weight == nullptr) {
    return nullptr;
  }
  std::vector<int> recu_weight_shape{2, hidden_size * 3, hidden_size};
  float *recu_tensor_data = nullptr;
  auto recu_weight = AddDefaultParameter(func_graph, base_name + "_cand_weight", recu_weight_shape, kNumberTypeFloat32,
                                         reinterpret_cast<void **>(&recu_tensor_data));
  if (recu_weight == nullptr) {
    return nullptr;
  }
  std::vector<int> bias_shape{2, hidden_size * 6};
  float *bias_tensor_data = nullptr;
  auto bias = AddDefaultParameter(func_graph, base_name + "_bias", bias_shape, kNumberTypeFloat32,
                                  reinterpret_cast<void **>(&bias_tensor_data));
  if (bias == nullptr) {
    return nullptr;
  }
  for (int i = 0; i < 2 * hidden_size * 6; ++i) {
    bias_tensor_data[i] = 0.0f;
  }

  if (ConvertWeightData(fw_gate_kernel, fw_cand_kernel, input_size, hidden_size, gate_tensor_data, recu_tensor_data) !=
      RET_OK) {
    return nullptr;
  }
  auto gate_data_diff = hidden_size * input_size * 3;
  auto recu_data_diff = hidden_size * hidden_size * 3;
  if (ConvertWeightData(bw_gate_kernel, bw_cand_kernel, input_size, hidden_size, gate_tensor_data + gate_data_diff,
                        recu_tensor_data + recu_data_diff) != RET_OK) {
    return nullptr;
  }

  if (ConvertBiasData(fw_gate_bias, fw_cand_bias, hidden_size, bias_tensor_data) != RET_OK) {
    return nullptr;
  }
  auto bias_data_diff = hidden_size * 6;
  if (ConvertBiasData(bw_gate_bias, bw_cand_bias, hidden_size, bias_tensor_data + bias_data_diff) != RET_OK) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_node_inputs = {value_node, input,          gate_weight, recu_weight,
                                             bias,       stacked_hidden, input_length};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}

CNodePtr TfBidirectionGruFusion::GetPostProcessNode(const FuncGraphPtr &func_graph, const CNodePtr &gru_output,
                                                    const std::string base_name) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(gru_output != nullptr);
  auto split_prim = std::make_shared<ops::Split>();
  split_prim->set_output_num(2);
  split_prim->set_axis(1);
  auto split_value_node = NewValueNode(split_prim);
  std::vector<AnfNodePtr> new_node_inputs = {split_value_node, gru_output};
  auto split_new_node = func_graph->NewCNode(new_node_inputs);
  split_new_node->set_fullname_with_scope("split_" + base_name);
  if (TfliteLstmCellFusion::SetAbstractTuple(split_new_node, 2) != RET_OK) {
    return nullptr;
  }

  auto split_out1 = TfliteLstmCellFusion::CreateOutputGetItem(func_graph, split_new_node, 0);
  if (split_out1 == nullptr) {
    return nullptr;
  }
  auto split_out2 = TfliteLstmCellFusion::CreateOutputGetItem(func_graph, split_new_node, 1);
  if (split_out2 == nullptr) {
    return nullptr;
  }

  auto concat_prim = std::make_shared<ops::Concat>();
  concat_prim->set_axis(3);
  auto concat_value_node = NewValueNode(concat_prim);
  std::vector<AnfNodePtr> concat_new_node_inputs = {concat_value_node, split_out1, split_out2};
  auto concat_new_node = func_graph->NewCNode(concat_new_node_inputs);
  concat_new_node->set_fullname_with_scope("concat_" + base_name);
  concat_new_node->set_abstract(gru_output->abstract()->Clone());

  auto squeeze_prim = std::make_shared<ops::Squeeze>();
  squeeze_prim->set_axis(std::vector<int64_t>{1});
  auto squeeze_value_node = NewValueNode(squeeze_prim);
  std::vector<AnfNodePtr> squeeze_new_node_inputs = {squeeze_value_node, concat_new_node};
  auto squeeze_new_node = func_graph->NewCNode(squeeze_new_node_inputs);
  squeeze_new_node->set_fullname_with_scope("squeeze_" + base_name);
  squeeze_new_node->set_abstract(gru_output->abstract()->Clone());

  auto transpose_prim = std::make_shared<ops::Transpose>();
  auto transpose_perm = BuildIntVecParameterNode(func_graph, {1, 0, 2}, "transpose_" + base_name + "_perm");
  auto transpose_new_node = func_graph->NewCNode(transpose_prim, {squeeze_new_node, transpose_perm});
  transpose_new_node->set_fullname_with_scope("transpose_" + base_name);
  transpose_new_node->set_abstract(gru_output->abstract()->Clone());

  return transpose_new_node;
}

const AnfNodePtr TfBidirectionGruFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &concat_node,
                                                 const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(concat_node != nullptr);
  MS_LOG(DEBUG) << "bidirection tf gru fusion pass";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(concat_node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  auto transpose_input = utils::cast<AnfNodePtr>((*equiv)[transpose_input_]);
  MS_ASSERT(transpose_input != nullptr);
  if (!utils::isa<CNodePtr>(transpose_input) || !CheckPrimitiveType(transpose_input, prim::kPrimTranspose)) {
    return nullptr;
  }

  PrimitiveVarMapPtr fw_cond_primitive_vars = std::make_shared<PrimitiveVarMap>();
  auto fw_cond_graph_pattern = GetCondGraphPattern(fw_cond_primitive_vars);
  auto fw_cond = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[0]]);
  MS_ASSERT(fw_cond != nullptr);
  auto fw_cond_equiv = TfliteLstmCellFusion::CheckSubGraph(func_graph, fw_cond_graph_pattern, fw_cond_primitive_vars,
                                                           fw_cond, kCondCNodesNum, kCondNodesNum);
  if (fw_cond_equiv == nullptr || fw_cond_equiv->empty()) {
    return nullptr;
  }

  PrimitiveVarMapPtr bw_cond_primitive_vars = std::make_shared<PrimitiveVarMap>();
  auto bw_cond_graph_pattern = GetCondGraphPattern(bw_cond_primitive_vars);
  auto bw_cond = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[0]]);
  MS_ASSERT(bw_cond != nullptr);
  auto bw_cond_equiv = TfliteLstmCellFusion::CheckSubGraph(func_graph, bw_cond_graph_pattern, bw_cond_primitive_vars,
                                                           bw_cond, kCondCNodesNum, kCondNodesNum);
  if (bw_cond_equiv == nullptr || bw_cond_equiv->empty()) {
    return nullptr;
  }

  PrimitiveVarMapPtr fw_primitive_vars_body = std::make_shared<PrimitiveVarMap>();
  auto fw_body_graph_pattern = GetBodyGraphPattern(fw_primitive_vars_body);
  auto fw_body = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[1]]);
  MS_ASSERT(fw_body != nullptr);
  auto fw_body_equiv = TfliteLstmCellFusion::CheckSubGraph(func_graph, fw_body_graph_pattern, fw_primitive_vars_body,
                                                           fw_body, kBodyCNodesNum, kBodyNodesNum);
  if (fw_body_equiv == nullptr || fw_body_equiv->empty()) {
    return nullptr;
  }

  PrimitiveVarMapPtr bw_primitive_vars_body = std::make_shared<PrimitiveVarMap>();
  auto bw_body_graph_pattern = GetBodyGraphPattern(bw_primitive_vars_body);
  auto bw_body = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[1]]);
  MS_ASSERT(bw_body != nullptr);
  auto bw_body_equiv = TfliteLstmCellFusion::CheckSubGraph(func_graph, bw_body_graph_pattern, bw_primitive_vars_body,
                                                           bw_body, kBodyCNodesNum, kBodyNodesNum);
  if (bw_body_equiv == nullptr || bw_body_equiv->empty()) {
    return nullptr;
  }

  const std::string gru_name = "gru_" + concat_node->fullname_with_scope();
  auto gru_node = CreateBiDirectionGruNode(func_graph, transpose_input, equiv, gru_name, 2);
  if (gru_node == nullptr) {
    return nullptr;
  }
  if (TfliteLstmCellFusion::SetAbstractTuple(gru_node, 2) != RET_OK) {
    return nullptr;
  }

  auto get_item_node = TfliteLstmCellFusion::CreateOutputGetItem(func_graph, gru_node, 0);
  if (get_item_node == nullptr) {
    return nullptr;
  }

  auto output_node = GetPostProcessNode(func_graph, get_item_node, gru_node->fullname_with_scope());
  MS_LOG(INFO) << "gru node:" << gru_node->fullname_with_scope() << " fusion success";
  return output_node;
}
}  // namespace opt
}  // namespace mindspore
