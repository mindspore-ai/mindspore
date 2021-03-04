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
#include "tools/optimizer/fusion/tf_bidirection_gru_cf_fusion.h"
#include <memory>
#include <set>
#include <functional>
#include "src/common/utils.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "tools/converter/ops/ops_def.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kNumFwVars = 4;
constexpr size_t kNumBwVars = 4;
const auto &p1 = std::placeholders::_1;
BaseRef GetPrim(const PrimitivePtr &prim) { return std::make_shared<CondVar>(std::bind(IsOpType, p1, prim)); }

BaseRef GetPrim(const std::string &prim_name) {
  auto prim = std::make_shared<Primitive>(prim_name);
  return GetPrim(prim);
}
}  // namespace

TfBidirectionGruCfFusion::TfBidirectionGruCfFusion(const std::string &name, bool multi_graph)
    : TfBidirectionGruFusion(kNumFwVars, kNumBwVars, name, multi_graph) {
  /*
   * vars for fw/bw input
   * fw:
   * 0:kernel_gate 1:bias_gate 2:cand_kernel 3:cand_bias
   * bw:
   * 0:kernel_gate 1:bias_gate 2:cand_kernel 3:cand_bias
   */
}

BaseRef TfBidirectionGruCfFusion::DefineGruCellPattern(const BaseRef &in_ta_read, const BaseRef &switch3_true,
                                                       const std::vector<VarPtr> &vars) const {
  auto concat = VectorRef({GetPrim(prim::kPrimConcat), in_ta_read, switch3_true});
  auto matmul_enter = VectorRef({GetPrim(lite::kNameEnter), vars[0]});  // gate_kernel
  auto matmul = VectorRef({GetPrim(prim::kPrimMatMul), concat, matmul_enter});
  auto bias_enter = VectorRef({GetPrim(lite::kNameEnter), vars[1]});  // cand_bias
  auto bias = VectorRef({GetPrim(prim::kPrimBiasAdd), matmul, bias_enter});
  auto sigmoid = VectorRef({GetPrim(prim::kPrimActivation), bias});
  auto split = VectorRef({GetPrim(prim::kPrimSplit), sigmoid});
  auto rt = VectorRef({GetPrim(prim::kPrimTupleGetItem), split, std::make_shared<Var>()});
  auto zt = VectorRef({GetPrim(prim::kPrimTupleGetItem), split, std::make_shared<Var>()});
  auto mul = VectorRef({GetPrim(prim::kPrimMulFusion), rt, switch3_true});
  auto concat1 = VectorRef({GetPrim(prim::kPrimConcat), in_ta_read, mul});
  auto matmul1_enter = VectorRef({GetPrim(lite::kNameEnter), vars[2]});  // cand_kernel
  auto matmul1 = VectorRef({GetPrim(prim::kPrimMatMul), concat1, matmul1_enter});
  auto bias1_enter = VectorRef({GetPrim(lite::kNameEnter), vars[3]});  // cand_bias
  auto bias1 = VectorRef({GetPrim(prim::kPrimBiasAdd), matmul1, bias1_enter});
  auto tanh = VectorRef({GetPrim(prim::kPrimActivation), bias1});
  auto sub = VectorRef({GetPrim(prim::kPrimSubFusion), std::make_shared<CondVar>(IsParameterNode), zt});
  auto mul2 = VectorRef({GetPrim(prim::kPrimMulFusion), sub, tanh});
  auto mul1 = VectorRef({GetPrim(prim::kPrimMulFusion), zt, switch3_true});
  auto add = VectorRef({GetPrim(prim::kPrimAddFusion), mul1, mul2});
  return add;
}

const BaseRef TfBidirectionGruCfFusion::DefineBidirectionRnnPattern(const BaseRef &input,
                                                                    const std::vector<VarPtr> &vars,
                                                                    const VarPtr &init_state) const {
  // in order to match cyclic graph, some node in cycle is represented by SeqVar
  auto fw_shape1 = VectorRef({GetPrim(prim::kPrimShape), input});
  auto strided_slice = VectorRef({GetPrim(prim::kPrimStridedSlice), fw_shape1, std::make_shared<SeqVar>()});
  auto fw_max = VectorRef({GetPrim(prim::kPrimReduceFusion), input_length_, std::make_shared<Var>()});
  auto fw_maximum = VectorRef({GetPrim(prim::kPrimMaximum), std::make_shared<CondVar>(IsParameterNode), fw_max});
  auto fw_minimum = VectorRef({GetPrim(prim::kPrimMinimum), strided_slice, fw_maximum});
  auto fw_less1_enter = VectorRef({GetPrim(lite::kNameEnter), fw_minimum});
  // SeqVar:counter_merge1
  auto fw_less1 = VectorRef({GetPrim(prim::kPrimLess), std::make_shared<SeqVar>(), fw_less1_enter});

  // SeqVar:fw_merge,loop_cond
  auto fw_switch = VectorRef({GetPrim(prim::kPrimSwitch), std::make_shared<SeqVar>()});
  auto fw_switch_true = VectorRef({GetPrim(prim::kPrimTupleGetItem), fw_switch, std::make_shared<Var>()});  // identity
  auto fw_add = VectorRef({GetPrim(prim::kPrimAddFusion), fw_switch_true, std::make_shared<CondVar>(IsParameterNode)});
  auto fw_next_iter = VectorRef({GetPrim(lite::kNameNextIteration), fw_add});
  auto fw_merge_enter = VectorRef({GetPrim(lite::kNameEnter), std::make_shared<CondVar>(IsParameterNode)});
  auto fw_merge = VectorRef({GetPrim(prim::kPrimMerge), fw_merge_enter, fw_next_iter});
  auto fw_less_enter = VectorRef({GetPrim(lite::kNameEnter), strided_slice});
  auto fw_less = VectorRef({GetPrim(prim::kPrimLess), fw_merge, fw_less_enter});

  auto fw_logical_and = VectorRef({GetPrim(prim::kPrimLogicalAnd), fw_less, fw_less1});
  // SeqVar:fw_logical_and
  auto loop_cond = VectorRef({GetPrim(lite::kNameLoopCond), fw_logical_and});

  auto fw_shape = VectorRef({GetPrim(prim::kPrimShape), input});
  auto fw_unstack_strided_slice = VectorRef({GetPrim(prim::kPrimStridedSlice), fw_shape, std::make_shared<SeqVar>()});
  auto fw_unstack_range = VectorRef({GetPrim(prim::kPrimRange), std::make_shared<CondVar>(IsParameterNode),
                                     fw_unstack_strided_slice, std::make_shared<CondVar>(IsParameterNode)});

  // SeqVar:switch1_true
  auto counter_add =
    VectorRef({GetPrim(prim::kPrimAddFusion), std::make_shared<SeqVar>(), std::make_shared<CondVar>(IsParameterNode)});
  auto counter_zero = VectorRef({GetPrim(lite::kNameEnter), std::make_shared<CondVar>(IsParameterNode)});
  auto counter_next_iter = VectorRef({GetPrim(lite::kNameNextIteration), counter_add});
  auto counter_merge1 = VectorRef({GetPrim(prim::kPrimMerge), counter_zero, counter_next_iter});
  auto counter_switch1 = VectorRef({GetPrim(prim::kPrimSwitch), counter_merge1, loop_cond});
  auto switch1_true =
    VectorRef({GetPrim(prim::kPrimTupleGetItem), counter_switch1, std::make_shared<Var>()});  // identity1

  auto in_ta = VectorRef({GetPrim(lite::kNameTensorArrayV3), strided_slice});
  auto in_ta_handle = VectorRef({GetPrim(prim::kPrimTupleGetItem), in_ta, std::make_shared<Var>()});
  auto in_ta_flow = VectorRef({GetPrim(prim::kPrimTupleGetItem), in_ta, std::make_shared<Var>()});
  auto fw_unstack_ta_scatter =
    VectorRef({GetPrim(lite::kNameTensorArrayScatterV3), in_ta_handle, fw_unstack_range, input, in_ta_flow});
  auto in_ta_enter1 = VectorRef({GetPrim(lite::kNameEnter), fw_unstack_ta_scatter});
  auto in_ta_enter = VectorRef({GetPrim(lite::kNameEnter), in_ta_handle});
  auto in_ta_read = VectorRef({GetPrim(lite::kNameTensorArrayReadV3), in_ta_enter, switch1_true, in_ta_enter1});

  auto greater_equal_enter = VectorRef({GetPrim(lite::kNameEnter), input_length_});
  auto greater_equal = VectorRef({GetPrim(prim::kPrimGreaterEqual), switch1_true, greater_equal_enter});
  auto select1 = VectorRef({GetPrim(prim::kPrimSelect), greater_equal, std::make_shared<SeqVar>()});  // select h

  auto next_iteration3 = VectorRef({GetPrim(lite::kNameNextIteration), select1});
  auto enter3 = VectorRef({GetPrim(lite::kNameEnter), init_state});
  auto merge3 = VectorRef({GetPrim(prim::kPrimMerge), enter3, next_iteration3});
  auto switch3 = VectorRef({GetPrim(prim::kPrimSwitch), merge3, loop_cond});
  auto switch3_true = VectorRef({GetPrim(prim::kPrimTupleGetItem), switch3, std::make_shared<Var>()});  // identity3

  auto rnn_cell_out = DefineGruCellPattern(in_ta_read, switch3_true, vars);

  auto out_ta = VectorRef({GetPrim(lite::kNameTensorArrayV3), strided_slice});
  auto out_ta_handle = VectorRef({GetPrim(prim::kPrimTupleGetItem), out_ta, std::make_shared<Var>()});
  auto out_ta_flow = VectorRef({GetPrim(prim::kPrimTupleGetItem), out_ta, std::make_shared<Var>()});
  auto out_ta_enter = VectorRef({GetPrim(lite::kNameEnter), out_ta_handle});

  auto switch2_true = VectorRef({GetPrim(prim::kPrimTupleGetItem), std::make_shared<SeqVar>()});  // cycle

  auto concat1 = VectorRef({GetPrim(prim::kPrimConcat), std::make_shared<SeqVar>()});
  auto zeros1 = VectorRef({GetPrim(prim::kPrimFill), std::make_shared<CondVar>(IsParameterNode), concat1});
  auto select_enter = VectorRef({GetPrim(lite::kNameEnter), zeros1});
  auto select = VectorRef({GetPrim(prim::kPrimSelect), greater_equal, select_enter, rnn_cell_out});  // select x
  auto ta_write = VectorRef({GetPrim(lite::kNameTensorArrayWriteV3), out_ta_enter, switch1_true, select, switch2_true});

  auto enter2 = VectorRef({GetPrim(lite::kNameEnter), out_ta_flow});
  auto next_iter2 = VectorRef({GetPrim(lite::kNameNextIteration), ta_write});
  auto merge2 = VectorRef({GetPrim(prim::kPrimMerge), enter2, next_iter2});
  auto switch2 = VectorRef({GetPrim(prim::kPrimSwitch), merge2, loop_cond});
  auto switch2_false = VectorRef({GetPrim(prim::kPrimTupleGetItem), switch2, std::make_shared<Var>()});

  auto exit2 = VectorRef({GetPrim(lite::kNameExit), switch2_false});
  auto ta_size = VectorRef({GetPrim(lite::kNameTensorArraySizeV3), out_ta_handle, exit2});
  auto range = VectorRef({GetPrim(prim::kPrimRange), std::make_shared<Var>(), ta_size, std::make_shared<Var>()});
  auto tensor_array_gather = VectorRef({GetPrim(lite::kNameTensorArrayGatherV3), out_ta_handle, range, exit2});
  auto range1 = VectorRef({GetPrim(prim::kPrimRange), std::make_shared<SeqVar>()});
  auto concat2 = VectorRef({GetPrim(prim::kPrimConcat), std::make_shared<CondVar>(IsParameterNode), range1});
  auto fw_out_trans = VectorRef({GetPrim(prim::kPrimTranspose), tensor_array_gather, concat2});
  return fw_out_trans;
}

const BaseRef TfBidirectionGruCfFusion::DefinePattern() const {
  const auto fw_out_trans = DefineBidirectionRnnPattern(transpose_input_, fw_vars_, fw_init_state_);

  auto bw_reverse_in = VectorRef({GetPrim(prim::kPrimReverseSequence), input_, input_length_});
  auto bw_range = VectorRef({GetPrim(prim::kPrimRange), std::make_shared<SeqVar>()});
  auto bw_concat = VectorRef({GetPrim(prim::kPrimConcat), std::make_shared<CondVar>(IsParameterNode), bw_range});
  auto bw_transpose = VectorRef({GetPrim(prim::kPrimTranspose), bw_reverse_in, bw_concat});
  auto bw_out_trans = DefineBidirectionRnnPattern(bw_transpose, bw_vars_, bw_init_state_);
  auto bw_reverse_out = VectorRef({GetPrim(prim::kPrimReverseSequence), bw_out_trans, input_length_});
  auto concat = VectorRef({GetPrim(prim::kPrimConcat), fw_out_trans, bw_reverse_out});
  return concat;
}

const AnfNodePtr TfBidirectionGruCfFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &concat_node,
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

  const std::string gru_name = "gru_" + concat_node->fullname_with_scope();
  auto gru_node = CreateBiDirectionGruNode(func_graph, transpose_input, equiv, gru_name, 0);
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
