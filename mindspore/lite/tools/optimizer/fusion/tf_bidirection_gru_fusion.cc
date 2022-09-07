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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/tf_bidirection_gru_fusion.h"
#include <memory>
#include <functional>
#include "tools/optimizer/common/helper.h"
#include "ops/concat.h"
#include "ops/gru.h"
#include "ops/split.h"
#include "ops/squeeze.h"
#include "ops/stack.h"
#include "ops/transpose.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "include/common/utils/utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kOffsetTwo = 2;
constexpr int kReservedParamNodesNum = 13;
constexpr size_t kCondNodesNum = 12;
constexpr size_t kCondCNodesNum = 4;
constexpr size_t kBodyNodesNum = 69;
constexpr size_t kBodyCNodesNum = 25;
constexpr auto kGateNum = 2;
const auto &p1 = std::placeholders::_1;
VectorRef GenerateBodyGraphHiddenPattern(const BaseRef &sigmoid1, const BaseRef &get_item,
                                         const std::vector<CondVarPtr> &placeholders) {
  MS_CHECK_TRUE_RET(placeholders.size() >= kCondCNodesNum, {});
  auto is_var_split = std::make_shared<Var>("Split");
  MS_CHECK_TRUE_RET(is_var_split != nullptr, {});
  VectorRef split = VectorRef({is_var_split, sigmoid1});
  auto is_var_tuple_getitem1 = std::make_shared<Var>("TupleGetItem");
  MS_CHECK_TRUE_RET(is_var_tuple_getitem1 != nullptr, {});
  auto is_var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var4 != nullptr, {});
  VectorRef get_item1 = VectorRef({is_var_tuple_getitem1, split, is_var4});
  auto is_var_tuple_getitem2 = std::make_shared<Var>("TupleGetItem");
  MS_CHECK_TRUE_RET(is_var_tuple_getitem2 != nullptr, {});
  auto is_var5 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var5 != nullptr, {});
  VectorRef get_item2 = VectorRef({is_var_tuple_getitem2, split, is_var5});

  auto is_var_mul1 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul1 != nullptr, {});
  VectorRef pre_reset = VectorRef({is_var_mul1, get_item1, placeholders[4]});
  auto is_var_concat = std::make_shared<Var>("Concat");
  MS_CHECK_TRUE_RET(is_var_concat != nullptr, {});
  VectorRef concat2 = VectorRef({is_var_concat, get_item, pre_reset});
  auto is_var_matmul2 = std::make_shared<Var>("Matmul");
  MS_CHECK_TRUE_RET(is_var_matmul2 != nullptr, {});
  VectorRef matmul2 = VectorRef({is_var_matmul2, concat2, placeholders[10]});
  auto is_var_biasadd2 = std::make_shared<Var>("BiasAdd");
  MS_CHECK_TRUE_RET(is_var_biasadd2 != nullptr, {});
  VectorRef biasadd2 = VectorRef({is_var_biasadd2, matmul2, placeholders[11]});
  auto is_var_tanh = std::make_shared<Var>("Tanh");
  MS_CHECK_TRUE_RET(is_var_tanh != nullptr, {});
  VectorRef tanh = VectorRef({is_var_tanh, biasadd2});

  auto is_var_mul2 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul2 != nullptr, {});
  VectorRef update_hidden = VectorRef({is_var_mul2, get_item2, placeholders[4]});
  auto is_var_sub = std::make_shared<Var>("Sub");
  MS_CHECK_TRUE_RET(is_var_sub != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  VectorRef minus_update = VectorRef({is_var_sub, is_param, get_item2});
  auto is_var_mul3 = std::make_shared<Var>("Mul");
  MS_CHECK_TRUE_RET(is_var_mul3 != nullptr, {});
  VectorRef updated = VectorRef({is_var_mul3, minus_update, tanh});

  auto is_var_add = std::make_shared<Var>("Add");
  MS_CHECK_TRUE_RET(is_var_add != nullptr, {});
  VectorRef new_hidden = VectorRef({is_var_add, update_hidden, updated});

  return new_hidden;
}
}  // namespace

bool TfBidirectionGruFusion::Init() const {
  /*
   * vars for while input
   * fw_while_inputs:
   * 0:cond 1:body 2:kernel_gate 3:bias_gate 4:cand_kernel 5:cand_bias
   * bw_while_inputs:
   * 0:cond 1:body 2:kernel_gate 3:bias_gate 4:cand_kernel 5:cand_bias
   */
  for (int i = 0; i < num_fw_vars_; ++i) {
    auto is_var = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var != nullptr, false);
    fw_vars_.emplace_back(is_var);
  }
  for (int i = 0; i < num_bw_vars_; ++i) {
    auto is_var = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(is_var != nullptr, false);
    bw_vars_.emplace_back(is_var);
  }
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, false);
  input_length_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_length_ != nullptr, false);
  transpose_input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(transpose_input_ != nullptr, false);
  fw_init_state_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(fw_init_state_ != nullptr, false);
  bw_init_state_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(bw_init_state_ != nullptr, false);
  return true;
}

const VectorRef TfBidirectionGruFusion::DefineFowardPattern() const {
  auto is_reduce = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion));
  MS_CHECK_TRUE_RET(is_reduce != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto fw_reduce = VectorRef({is_reduce, input_length_, is_param1});
  auto is_maximum = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMaximum));
  MS_CHECK_TRUE_RET(is_maximum != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto fw_max = VectorRef({is_maximum, is_param2, fw_reduce});

  auto is_shape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimShape));
  MS_CHECK_TRUE_RET(is_shape != nullptr, {});
  auto fw_shape = VectorRef({is_shape, transpose_input_});
  auto is_strided_slice = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice));
  MS_CHECK_TRUE_RET(is_strided_slice != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  auto fw_stride = VectorRef({is_strided_slice, fw_shape, is_seq_var});
  auto is_minimum = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMinimum));
  MS_CHECK_TRUE_RET(is_minimum != nullptr, {});
  auto fw_min = VectorRef({is_minimum, fw_stride, fw_max});
  auto is_tensor_list_reserve = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListReserve));
  MS_CHECK_TRUE_RET(is_tensor_list_reserve != nullptr, {});
  auto is_param3 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, {});
  auto fw_reserve = VectorRef({is_tensor_list_reserve, is_param3, fw_stride});
  auto is_tensor_list_from_tensor = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListFromTensor));
  MS_CHECK_TRUE_RET(is_tensor_list_from_tensor != nullptr, {});
  auto is_param4 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param4 != nullptr, {});
  auto fw_from_tensor = VectorRef({is_tensor_list_from_tensor, transpose_input_, is_param4});
  auto is_while = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimWhile));
  MS_CHECK_TRUE_RET(is_while != nullptr, {});
  auto is_param5 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param5 != nullptr, {});
  auto is_param6 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param6 != nullptr, {});
  auto fw_while = VectorRef({is_while, fw_vars_[0], fw_vars_[1], is_param5, fw_stride, is_param6, fw_reserve,
                             fw_init_state_, fw_min, fw_from_tensor, input_length_});
  fw_while.insert(fw_while.end(), fw_vars_.begin() + kOffsetTwo, fw_vars_.end());
  auto is_var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var1 != nullptr, {});
  fw_while.emplace_back(is_var1);
  auto is_tuple_getitem = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem));
  MS_CHECK_TRUE_RET(is_tuple_getitem != nullptr, {});
  auto is_var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var2 != nullptr, {});
  auto fw_get_item = VectorRef({is_tuple_getitem, fw_while, is_var2});
  auto is_tensor_list_stack = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListStack));
  MS_CHECK_TRUE_RET(is_tensor_list_stack != nullptr, {});
  auto is_param7 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param7 != nullptr, {});
  auto fw_stack = VectorRef({is_tensor_list_stack, fw_get_item, is_param7});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var3 != nullptr, {});
  auto fw_out_trans = VectorRef({is_transpose, fw_stack, is_var3});
  return fw_out_trans;
}

const VectorRef TfBidirectionGruFusion::DefinebackwardPattern() const {
  auto is_reverse_sequence = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReverseSequence));
  MS_CHECK_TRUE_RET(is_reverse_sequence != nullptr, {});
  auto bw_reverse_seq = VectorRef({is_reverse_sequence, input_, input_length_});
  auto is_reduce = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReduceFusion));
  MS_CHECK_TRUE_RET(is_reduce != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto bw_max1 = VectorRef({is_reduce, input_length_, is_param1});
  auto is_maximum = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMaximum));
  MS_CHECK_TRUE_RET(is_maximum != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto bw_max2 = VectorRef({is_maximum, is_param2, bw_max1});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var1 != nullptr, {});
  auto bw_trans = VectorRef({is_transpose, bw_reverse_seq, is_var1});
  auto is_shape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimShape));
  MS_CHECK_TRUE_RET(is_shape != nullptr, {});
  auto bw_shape = VectorRef({is_shape, bw_trans});
  auto is_strided_slice = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimStridedSlice));
  MS_CHECK_TRUE_RET(is_strided_slice != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  auto bw_stride = VectorRef({is_strided_slice, bw_shape, is_seq_var});
  auto is_minimum = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMinimum));
  MS_CHECK_TRUE_RET(is_minimum != nullptr, {});
  auto bw_min = VectorRef({is_minimum, bw_stride, bw_max2});
  auto is_tensor_list_reserve = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListReserve));
  MS_CHECK_TRUE_RET(is_tensor_list_reserve != nullptr, {});
  auto is_param3 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, {});
  auto bw_reserve = VectorRef({is_tensor_list_reserve, is_param3, bw_stride});
  auto is_tensor_list_from_tensor = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListFromTensor));
  MS_CHECK_TRUE_RET(is_tensor_list_from_tensor != nullptr, {});
  auto is_param4 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param4 != nullptr, {});
  auto bw_from_tensor = VectorRef({is_tensor_list_from_tensor, bw_trans, is_param4});
  auto is_while = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimWhile));
  MS_CHECK_TRUE_RET(is_while != nullptr, {});
  auto is_param5 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param5 != nullptr, {});
  auto is_param6 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param6 != nullptr, {});
  auto bw_while = VectorRef({is_while, bw_vars_[0], bw_vars_[1], is_param5, bw_stride, is_param6, bw_reserve,
                             bw_init_state_, bw_min, bw_from_tensor, input_length_});
  bw_while.insert(bw_while.end(), bw_vars_.begin() + kOffsetTwo, bw_vars_.end());
  auto is_var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var2 != nullptr, {});
  bw_while.emplace_back(is_var2);
  auto is_tuple_getitem = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTupleGetItem));
  MS_CHECK_TRUE_RET(is_tuple_getitem != nullptr, {});
  auto is_var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var3 != nullptr, {});
  auto bw_get_item = VectorRef({is_tuple_getitem, bw_while, is_var3});
  auto is_tensor_list_stack = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTensorListStack));
  MS_CHECK_TRUE_RET(is_tensor_list_stack != nullptr, {});
  auto is_param7 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param7 != nullptr, {});
  auto bw_stack = VectorRef({is_tensor_list_stack, bw_get_item, is_param7});
  auto is_var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var4 != nullptr, {});
  auto bw_out_trans = VectorRef({is_transpose, bw_stack, is_var4});
  auto bw_reverse1 = VectorRef({is_reverse_sequence, bw_out_trans, input_length_});
  return bw_reverse1;
}

const BaseRef TfBidirectionGruFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }

  // forward
  auto fw_out_trans = DefineFowardPattern();
  MS_CHECK_TRUE_RET(!fw_out_trans.empty(), {});

  // backward
  auto bw_reverse1 = DefinebackwardPattern();
  MS_CHECK_TRUE_RET(!bw_reverse1.empty(), {});

  auto is_concat = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimConcat));
  MS_CHECK_TRUE_RET(is_concat != nullptr, {});
  auto concat = VectorRef({is_concat, fw_out_trans, bw_reverse1});
  return concat;
}

AnfNodePtr TfBidirectionGruFusion::GetCondGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  MS_ASSERT(primitive_vars != nullptr);
  auto is_less1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess));
  MS_CHECK_TRUE_RET(is_less1 != nullptr, nullptr);
  auto is_param1 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, nullptr);
  auto is_param2 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, nullptr);
  VectorRef less1_ref = VectorRef({is_less1, is_param1, is_param2});
  auto is_less2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLess));
  MS_CHECK_TRUE_RET(is_less2 != nullptr, nullptr);
  auto is_param3 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, nullptr);
  auto is_param4 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param4 != nullptr, nullptr);
  VectorRef less2_ref = VectorRef({is_less2, is_param3, is_param4});
  auto is_logical_and = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimLogicalAnd));
  MS_CHECK_TRUE_RET(is_logical_and != nullptr, nullptr);
  VectorRef logicaland_ref = VectorRef({is_logical_and, less1_ref, less2_ref});
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReturn));
  MS_CHECK_TRUE_RET(is_return != nullptr, nullptr);
  VectorRef return_ref = VectorRef({is_return, logicaland_ref});
  VarPtr is_fg = std::make_shared<Var>("RootG");
  auto pattern = Helper::SexpToNode(return_ref, is_fg, primitive_vars.get(), true);
  return pattern;
}

AnfNodePtr TfBidirectionGruFusion::GetBodyGraphPattern(const PrimitiveVarMapPtr &primitive_vars) const {
  MS_ASSERT(primitive_vars != nullptr);
  std::vector<CondVarPtr> placeholders;
  for (int i = 0; i < kReservedParamNodesNum; ++i) {
    auto is_param_placeholder = std::make_shared<CondVar>(IsParameterNode);
    MS_CHECK_TRUE_RET(is_param_placeholder != nullptr, nullptr);
    placeholders.emplace_back(is_param_placeholder);
  }
  auto is_var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var1 != nullptr, nullptr);
  auto is_param1 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, nullptr);
  VectorRef add = VectorRef({is_var1, placeholders[2], is_param1});
  auto is_var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var2 != nullptr, nullptr);
  auto is_param2 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, nullptr);
  VectorRef add1 = VectorRef({is_var2, placeholders[0], is_param2});

  auto is_getitem = std::make_shared<Var>("GetItem");
  MS_CHECK_TRUE_RET(is_getitem != nullptr, nullptr);
  auto is_param3 = std::make_shared<CondVar>(IsParameterNode);
  MS_CHECK_TRUE_RET(is_param3 != nullptr, nullptr);
  VectorRef get_item = VectorRef({is_getitem, placeholders[6], placeholders[2], is_param3});
  auto is_var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var3 != nullptr, nullptr);
  VectorRef concat_input_h = VectorRef({is_var3, get_item, placeholders[4]});

  auto is_var_matmul1 = std::make_shared<Var>("Matmul");
  MS_CHECK_TRUE_RET(is_var_matmul1 != nullptr, nullptr);
  VectorRef matmul1 = VectorRef({is_var_matmul1, concat_input_h, placeholders[8]});
  auto is_var_biasadd1 = std::make_shared<Var>("BiasAdd");
  MS_CHECK_TRUE_RET(is_var_biasadd1 != nullptr, nullptr);
  VectorRef biasadd1 = VectorRef({is_var_biasadd1, matmul1, placeholders[9]});
  auto is_var_sigmoid = std::make_shared<Var>("Sigmoid");
  MS_CHECK_TRUE_RET(is_var_sigmoid != nullptr, nullptr);
  VectorRef sigmoid1 = VectorRef({is_var_sigmoid, biasadd1});

  auto new_hidden = GenerateBodyGraphHiddenPattern(sigmoid1, get_item, placeholders);
  MS_CHECK_TRUE_RET(!new_hidden.empty(), nullptr);

  auto is_var_ge = std::make_shared<Var>("GreaterEqual");
  MS_CHECK_TRUE_RET(is_var_ge != nullptr, nullptr);
  VectorRef greater_equal = VectorRef({is_var_ge, placeholders[2], placeholders[7]});

  auto is_var_switch1 = std::make_shared<Var>("Switch");
  MS_CHECK_TRUE_RET(is_var_switch1 != nullptr, {});
  VectorRef select_output = VectorRef({is_var_switch1, greater_equal, placeholders[12], new_hidden});
  auto is_var_setitem = std::make_shared<Var>("SetItem");
  MS_CHECK_TRUE_RET(is_var_setitem != nullptr, {});
  VectorRef output = VectorRef({is_var_setitem, placeholders[3], placeholders[2], select_output});

  auto is_var_switch2 = std::make_shared<Var>("Switch");
  MS_CHECK_TRUE_RET(is_var_switch2 != nullptr, {});
  VectorRef select_hidden = VectorRef({is_var_switch2, greater_equal, placeholders[4], new_hidden});

  auto is_make_tuple = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMakeTuple));
  MS_CHECK_TRUE_RET(is_make_tuple != nullptr, nullptr);
  std::vector<BaseRef> outputs = {is_make_tuple,  add1,          placeholders[1], add,
                                  output,         select_hidden, placeholders[5], placeholders[6],
                                  placeholders[7]};
  outputs.insert(outputs.end(), placeholders.begin() + 8, placeholders.end());
  VectorRef make_tuple_node = VectorRef(outputs);
  auto is_return = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReturn));
  MS_CHECK_TRUE_RET(is_return != nullptr, nullptr);
  VectorRef return_node = VectorRef({is_return, make_tuple_node});

  VarPtr is_fg = std::make_shared<Var>("RootG");
  MS_CHECK_TRUE_RET(is_fg != nullptr, nullptr);
  auto pattern = Helper::SexpToNode(return_node, is_fg, primitive_vars.get(), true);
  return pattern;
}

tensor::TensorPtr TfBidirectionGruFusion::GetDefaultTensorInfo(const AnfNodePtr &parameter_anf) {
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
  auto tensor_info = std::dynamic_pointer_cast<tensor::Tensor>(parameter->default_param());
  return tensor_info;
}

STATUS TfBidirectionGruFusion::GetInputAndHiddenSize(const AnfNodePtr &fw_cand_kernel_anf,
                                                     const AnfNodePtr &bw_cand_kernel_anf, int *input_size,
                                                     int *hidden_size) {
  MS_ASSERT(fw_cand_kernel_anf != nullptr);
  MS_ASSERT(bw_cand_kernel_anf != nullptr);
  MS_ASSERT(input_size != nullptr);
  MS_ASSERT(hidden_size != nullptr);
  auto fw_cand_kernel_value = GetDefaultTensorInfo(fw_cand_kernel_anf);
  if (fw_cand_kernel_value == nullptr) {
    return RET_ERROR;
  }
  auto fw_cand_kernel_shape = fw_cand_kernel_value->shape();
  if (fw_cand_kernel_shape.size() != kInputSizeTwo) {
    return RET_ERROR;
  }
  auto bw_cand_kernel_value = GetDefaultTensorInfo(bw_cand_kernel_anf);
  if (bw_cand_kernel_value == nullptr) {
    return RET_ERROR;
  }
  auto bw_cand_kernel_shape = bw_cand_kernel_value->shape();
  if (bw_cand_kernel_shape.size() != kInputSizeTwo) {
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
                                                         void **tensor_data) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(tensor_data != nullptr);
  auto parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(parameter != nullptr, nullptr);
  parameter->set_name(name);
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto abstract = lite::CreateTensorAbstract(shape_vector, type);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return nullptr;
  }
  parameter->set_abstract(abstract);

  auto gate_weight_default = std::make_shared<tensor::Tensor>(type, shape_vector);
  if (gate_weight_default == nullptr) {
    MS_LOG(ERROR) << "gate_weight_default is nullptr";
    return nullptr;
  }

  *tensor_data = gate_weight_default->data_c();
  parameter->set_default_param(gate_weight_default);
  return parameter;
}

void TfBidirectionGruFusion::CopyFlattenMatData(const float *mat, const int C, const int r0, const int r1, const int c0,
                                                const int c1, float *data, bool t) {
  MS_ASSERT(mat != nullptr);
  MS_ASSERT(data != nullptr);
  MS_ASSERT(r0 >= 0 && r0 < r1);
  MS_ASSERT(c0 >= 0 && c0 < c1 && c1 <= C);
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
                                                 float *recu_tensor_data) {
  MS_ASSERT(gate_weight != nullptr);
  MS_ASSERT(cand_weight != nullptr);
  MS_ASSERT(gate_tensor_data != nullptr);
  MS_ASSERT(recu_tensor_data != nullptr);
  const std::vector<int64_t> gate_shape{input_size + hidden_size, hidden_size * kGateNum};
  const std::vector<int64_t> cand_shape{hidden_size * kGateNum, hidden_size};
  auto gate_weight_value = GetDefaultTensorInfo(gate_weight);
  if (gate_weight_value == nullptr) {
    return RET_ERROR;
  }
  auto gate_weight_data = reinterpret_cast<float *>(gate_weight_value->data_c());
  if (gate_weight_data == nullptr) {
    return RET_ERROR;
  }
  auto gate_weight_shape = gate_weight_value->shape();

  auto cand_weight_value = GetDefaultTensorInfo(cand_weight);
  if (cand_weight_value == nullptr) {
    return RET_ERROR;
  }
  auto cand_weight_data = reinterpret_cast<float *>(cand_weight_value->data_c());
  if (cand_weight_data == nullptr) {
    return RET_ERROR;
  }
  auto cand_weight_shape = cand_weight_value->shape();

  if (gate_weight_shape != gate_shape || cand_weight_shape != cand_shape) {
    return RET_ERROR;
  }

  // input_update_weight
  CopyFlattenMatData(gate_weight_data, hidden_size * kGateNum, 0, input_size, hidden_size, hidden_size * kGateNum,
                     gate_tensor_data, true);
  // input_reset_weight
  CopyFlattenMatData(gate_weight_data, hidden_size * kGateNum, 0, input_size, 0, hidden_size,
                     gate_tensor_data + input_size * hidden_size, true);
  // input_hidden_weight
  CopyFlattenMatData(cand_weight_data, hidden_size, 0, input_size, 0, hidden_size,
                     gate_tensor_data + input_size * hidden_size * kGateNum, true);

  // state_update_weight
  CopyFlattenMatData(gate_weight_data, hidden_size * kGateNum, input_size, input_size + hidden_size, hidden_size,
                     hidden_size * kGateNum, recu_tensor_data, true);
  // state_reset_weight
  CopyFlattenMatData(gate_weight_data, hidden_size * kGateNum, input_size, input_size + hidden_size, 0, hidden_size,
                     recu_tensor_data + hidden_size * hidden_size, true);
  // state_hidden_weight
  CopyFlattenMatData(cand_weight_data, hidden_size, input_size, input_size + hidden_size, 0, hidden_size,
                     recu_tensor_data + hidden_size * hidden_size * kGateNum, true);
  return RET_OK;
}

STATUS TfBidirectionGruFusion::ConvertBiasData(const AnfNodePtr &gate_bias, const AnfNodePtr &cand_bias,
                                               const int hidden_size, float *tensor_data) {
  MS_ASSERT(gate_bias != nullptr && cand_bias != nullptr);
  MS_ASSERT(tensor_data != nullptr);
  std::vector<int64_t> gate_shape{hidden_size * kGateNum};
  std::vector<int64_t> cand_shape{hidden_size};
  auto gate_bias_value = GetDefaultTensorInfo(gate_bias);
  if (gate_bias_value == nullptr) {
    return RET_ERROR;
  }
  auto gate_bias_data = reinterpret_cast<float *>(gate_bias_value->data_c());
  auto gate_bias_shape = gate_bias_value->shape();
  auto cand_bias_value = GetDefaultTensorInfo(cand_bias);
  if (cand_bias_value == nullptr) {
    return RET_ERROR;
  }
  auto cand_bias_data = reinterpret_cast<float *>(cand_bias_value->data_c());
  auto cand_bias_shape = cand_bias_value->shape();
  if (gate_bias_shape != gate_shape || cand_bias_shape != cand_shape) {
    return RET_ERROR;
  }

  // update_gate bias
  CopyFlattenMatData(gate_bias_data, hidden_size * kGateNum, 0, 1, hidden_size, hidden_size * kGateNum, tensor_data,
                     false);
  // reset_gate bias
  CopyFlattenMatData(gate_bias_data, hidden_size * kGateNum, 0, 1, 0, hidden_size, tensor_data + hidden_size, false);
  // hidden_gate bias
  CopyFlattenMatData(cand_bias_data, hidden_size, 0, 1, 0, hidden_size, tensor_data + hidden_size * kGateNum, false);

  return RET_OK;
}

CNodePtr TfBidirectionGruFusion::GetStackedHiddenState(const FuncGraphPtr &func_graph, const AnfNodePtr &fw_init_state,
                                                       const AnfNodePtr &bw_init_state, const std::string &base_name) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(fw_init_state != nullptr);
  MS_ASSERT(bw_init_state != nullptr);
  auto stack_prim = std::make_shared<ops::Stack>();
  MS_CHECK_TRUE_RET(stack_prim != nullptr, nullptr);
  auto stack_prim_c = stack_prim->GetPrim();
  MS_CHECK_TRUE_RET(stack_prim_c != nullptr, nullptr);
  stack_prim->set_axis(0);
  auto value_node = NewValueNode(stack_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> new_node_inputs = {value_node, fw_init_state, bw_init_state};
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  if (fw_init_state->abstract() != nullptr) {
    new_node->set_abstract(fw_init_state->abstract()->Clone());
  }
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
  MS_CHECK_TRUE_RET(gru_prim != nullptr, nullptr);
  auto gru_prim_c = gru_prim->GetPrim();
  MS_CHECK_TRUE_RET(gru_prim_c != nullptr, nullptr);
  gru_prim->set_bidirectional(true);
  auto value_node = NewValueNode(gru_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);

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
  MS_ASSERT(input_length != nullptr);

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
  if (recu_weight == nullptr || recu_tensor_data == nullptr) {
    return nullptr;
  }
  std::vector<int> bias_shape{2, hidden_size * 6};
  float *bias_tensor_data = nullptr;
  auto bias = AddDefaultParameter(func_graph, base_name + "_bias", bias_shape, kNumberTypeFloat32,
                                  reinterpret_cast<void **>(&bias_tensor_data));
  if (bias == nullptr || bias_tensor_data == nullptr) {
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
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  auto prim = GetValueNode<PrimitivePtr>(new_node->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->AddAttr(ops::kFormat, MakeValue<int64_t>(Format::NHWC));
  new_node->set_fullname_with_scope(base_name);
  return new_node;
}

CNodePtr TfBidirectionGruFusion::GetPostProcessNode(const FuncGraphPtr &func_graph, const CNodePtr &gru_output,
                                                    const std::string &base_name) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(gru_output != nullptr);
  auto split_prim = std::make_shared<ops::Split>();
  MS_CHECK_TRUE_RET(split_prim != nullptr, nullptr);
  auto split_prim_c = split_prim->GetPrim();
  MS_CHECK_TRUE_RET(split_prim_c != nullptr, nullptr);
  split_prim->set_output_num(2);
  split_prim->set_axis(1);
  auto split_value_node = NewValueNode(split_prim_c);
  MS_CHECK_TRUE_RET(split_value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> new_node_inputs = {split_value_node, gru_output};
  auto split_new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(split_new_node != nullptr, nullptr);
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
  MS_CHECK_TRUE_RET(concat_prim != nullptr, nullptr);
  auto concat_prim_c = concat_prim->GetPrim();
  MS_CHECK_TRUE_RET(concat_prim_c != nullptr, nullptr);
  concat_prim->set_axis(3);
  auto concat_value_node = NewValueNode(concat_prim_c);
  MS_CHECK_TRUE_RET(concat_value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> concat_new_node_inputs = {concat_value_node, split_out1, split_out2};
  auto concat_new_node = func_graph->NewCNode(concat_new_node_inputs);
  MS_CHECK_TRUE_RET(concat_new_node != nullptr, nullptr);
  concat_new_node->set_fullname_with_scope("concat_" + base_name);
  if (gru_output->abstract() != nullptr) {
    concat_new_node->set_abstract(gru_output->abstract()->Clone());
  }

  auto squeeze_prim = std::make_shared<ops::Squeeze>();
  MS_CHECK_TRUE_RET(squeeze_prim != nullptr, nullptr);
  auto squeeze_prim_c = squeeze_prim->GetPrim();
  MS_CHECK_TRUE_RET(squeeze_prim_c != nullptr, nullptr);
  squeeze_prim->set_axis(std::vector<int64_t>{1});
  auto squeeze_value_node = NewValueNode(squeeze_prim_c);
  MS_CHECK_TRUE_RET(squeeze_value_node != nullptr, nullptr);
  std::vector<AnfNodePtr> squeeze_new_node_inputs = {squeeze_value_node, concat_new_node};
  auto squeeze_new_node = func_graph->NewCNode(squeeze_new_node_inputs);
  MS_CHECK_TRUE_RET(squeeze_new_node != nullptr, nullptr);
  squeeze_new_node->set_fullname_with_scope("squeeze_" + base_name);
  if (gru_output->abstract() != nullptr) {
    squeeze_new_node->set_abstract(gru_output->abstract()->Clone());
  }

  auto transpose_prim = std::make_shared<ops::Transpose>();
  MS_CHECK_TRUE_RET(transpose_prim != nullptr, nullptr);
  auto transpose_perm = BuildIntVecParameterNode(func_graph, {1, 0, 2}, "transpose_" + base_name + "_perm");
  MS_CHECK_TRUE_RET(transpose_perm != nullptr, nullptr);
  auto transpose_prim_c = transpose_prim->GetPrim();
  MS_CHECK_TRUE_RET(transpose_prim_c != nullptr, nullptr);
  auto transpose_new_node = func_graph->NewCNode(transpose_prim_c, {squeeze_new_node, transpose_perm});
  MS_CHECK_TRUE_RET(transpose_new_node != nullptr, nullptr);
  transpose_new_node->set_fullname_with_scope("transpose_" + base_name);
  if (gru_output->abstract() != nullptr) {
    transpose_new_node->set_abstract(gru_output->abstract()->Clone());
  }

  return transpose_new_node;
}

const AnfNodePtr TfBidirectionGruFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &concat_node,
                                                 const EquivPtr &equiv) const {
  if (func_graph == nullptr || concat_node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  auto transpose_input = utils::cast<AnfNodePtr>((*equiv)[transpose_input_]);
  MS_ASSERT(transpose_input != nullptr);
  if (!utils::isa<CNodePtr>(transpose_input) || !CheckPrimitiveType(transpose_input, prim::kPrimTranspose)) {
    return nullptr;
  }

  auto fw_cond_primitive_vars = std::make_shared<PrimitiveVarMap>();
  MS_CHECK_TRUE_RET(fw_cond_primitive_vars != nullptr, nullptr);
  auto fw_cond_graph_pattern = GetCondGraphPattern(fw_cond_primitive_vars);
  MS_CHECK_TRUE_RET(fw_cond_graph_pattern != nullptr, nullptr);
  auto fw_cond = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[0]]);
  MS_ASSERT(fw_cond != nullptr);
  auto fw_cond_equiv = TfliteLstmCellFusion::CheckSubGraph(fw_cond_graph_pattern, fw_cond_primitive_vars, fw_cond,
                                                           kCondCNodesNum, kCondNodesNum);
  if (fw_cond_equiv == nullptr || fw_cond_equiv->empty()) {
    return nullptr;
  }

  auto bw_cond_primitive_vars = std::make_shared<PrimitiveVarMap>();
  MS_CHECK_TRUE_RET(bw_cond_primitive_vars != nullptr, nullptr);
  auto bw_cond_graph_pattern = GetCondGraphPattern(bw_cond_primitive_vars);
  MS_CHECK_TRUE_RET(bw_cond_graph_pattern != nullptr, nullptr);
  auto bw_cond = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[0]]);
  MS_ASSERT(bw_cond != nullptr);
  auto bw_cond_equiv = TfliteLstmCellFusion::CheckSubGraph(bw_cond_graph_pattern, bw_cond_primitive_vars, bw_cond,
                                                           kCondCNodesNum, kCondNodesNum);
  if (bw_cond_equiv == nullptr || bw_cond_equiv->empty()) {
    return nullptr;
  }

  auto fw_primitive_vars_body = std::make_shared<PrimitiveVarMap>();
  MS_CHECK_TRUE_RET(fw_primitive_vars_body != nullptr, nullptr);
  auto fw_body_graph_pattern = GetBodyGraphPattern(fw_primitive_vars_body);
  MS_CHECK_TRUE_RET(fw_body_graph_pattern != nullptr, nullptr);
  auto fw_body = utils::cast<AnfNodePtr>((*equiv)[fw_vars_[1]]);
  MS_ASSERT(fw_body != nullptr);
  auto fw_body_equiv = TfliteLstmCellFusion::CheckSubGraph(fw_body_graph_pattern, fw_primitive_vars_body, fw_body,
                                                           kBodyCNodesNum, kBodyNodesNum);
  if (fw_body_equiv == nullptr || fw_body_equiv->empty()) {
    return nullptr;
  }

  auto bw_primitive_vars_body = std::make_shared<PrimitiveVarMap>();
  MS_CHECK_TRUE_RET(bw_primitive_vars_body != nullptr, nullptr);
  auto bw_body_graph_pattern = GetBodyGraphPattern(bw_primitive_vars_body);
  MS_CHECK_TRUE_RET(bw_body_graph_pattern != nullptr, nullptr);
  auto bw_body = utils::cast<AnfNodePtr>((*equiv)[bw_vars_[1]]);
  MS_ASSERT(bw_body != nullptr);
  auto bw_body_equiv = TfliteLstmCellFusion::CheckSubGraph(bw_body_graph_pattern, bw_primitive_vars_body, bw_body,
                                                           kBodyCNodesNum, kBodyNodesNum);
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
