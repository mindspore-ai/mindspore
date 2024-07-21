/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <cmath>
#include <memory>
#include <queue>
#include <utility>
#include <list>
#include <vector>
#include <string>
#include <algorithm>

#include "ops/other_ops.h"
#include "ops/array_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/make_tuple.h"
#include "utils/anf_utils.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/comm_manager.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/device_matrix.h"
#include "pipeline/jit/ps/action.h"
#include "mindspore/ccsrc/include/backend/optimizer/helper.h"
#include "mindspore/ccsrc/frontend/parallel/graph_util/generate_graph.h"
#include "mindspore/core/ops/op_enum.h"
#include "mindspore/core/ops/ops_func_impl/flash_attention_score.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/flash_attention_score_info.h"
#include "frontend/optimizer/flash_sp.h"
#include "frontend/parallel/graph_util/graph_info.h"

namespace mindspore {
using mindspore::ops::FASInputLayoutMode;
namespace parallel {
FlashSPInfo::FlashSPInfo(CNodePtr fa_score_node) {
  MS_EXCEPTION_IF_NULL(fa_score_node);
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  MS_EXCEPTION_IF_NULL(flash_score_info_ptr);

  flashsp_num_ = flash_score_info_ptr->s1_split_num();
  is_input_passed_ = flash_score_info_ptr->is_input_passed();
  dev_rank_id_ = g_device_manager->global_rank();

  auto rankList = flash_score_info_ptr->GetSPRankList();
  size_t pos = -1;
  for (size_t i = 0; i < rankList.size(); ++i) {
    if (dev_rank_id_ == rankList[i]) {
      pos = i;
    }
  }
  send_rank_id_ = rankList[(pos + 1) % rankList.size()];
  recv_rank_id_ = rankList[(pos + rankList.size() - 1) % rankList.size()];
}
namespace {
using CNodePtrPair = std::pair<CNodePtr, CNodePtr>;
using FSPInfo = FlashSPInfo;

std::vector<CNodePtr> FindFWFlashAttentionScore(const FuncGraphManagerPtr &manager,
                                                const std::vector<AnfNodePtr> &origin_nodes_topological) {
  std::vector<CNodePtr> result;
  for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
    auto node = origin_nodes_topological[i];
    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
      result.push_back(node->cast<CNodePtr>());
    }
  }
  return result;
}

CNodePtr NewReshapeNode(const AnfNodePtr &input_node, const ShapeVector &output_shape, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                            input_node, NewValueNode(MakeValue(output_shape))};
  auto reshape = input_node->func_graph()->NewCNode(reshape_inputs);
  MS_EXCEPTION_IF_NULL(reshape);

  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(output_shape), reshape);
  reshape->set_scope(input_node->scope());
  return reshape;
}

CNodePtr NewConcatNode(const AnfNodePtr &input_node, size_t concat_dim) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           input_node, NewValueNode(MakeValue(static_cast<int64_t>(concat_dim)))};
  auto concat = input_node->func_graph()->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat);
  concat->set_scope(input_node->scope());
  return concat;
}

CNodePtr NewMakeTupleNode(const std::vector<AnfNodePtr> &input_nodes) {
  // input_nodes are getitem nodes
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    make_tuple_inputs.push_back(input_nodes[i]);
  }
  auto make_tuple = input_nodes[0]->func_graph()->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_scope(input_nodes[0]->scope());
  return make_tuple;
}

CNodePtr NewSplitNode(const AnfNodePtr &input_node, size_t split_dim, size_t split_num) {
  if (split_num == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                          input_node, NewValueNode<int64_t>(split_dim),
                                          NewValueNode<int64_t>(split_num)};
  auto split = input_node->func_graph()->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);
  split->set_scope(input_node->scope());
  return split;
}

CNodePtr NewTupleGetItemNode(const AnfNodePtr &input_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto idx = NewValueNode(SizeToLong(output_index));
  MS_EXCEPTION_IF_NULL(idx);
  auto getitem = input_node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input_node, idx});
  MS_EXCEPTION_IF_NULL(getitem);
  getitem->set_scope(input_node->scope());
  return getitem;
}

CNodePtr NewNeighborExchangeNode(const AnfNodePtr &input_node, const std::vector<int64_t> &send_rank_ids,
                                 const std::vector<int64_t> &recv_rank_ids, int fa_index, int ne_index,
                                 parallel::Shape neigh_shape) {
  MS_EXCEPTION_IF_NULL(input_node);
  // input_node is maketuple node
  std::vector<AnfNodePtr> ne_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimNeighborExchange->name())),
                                       input_node};
  auto neighbor_exchange = input_node->func_graph()->NewCNode(ne_inputs);
  MS_EXCEPTION_IF_NULL(neighbor_exchange);

  // RECV_TYPE
  auto dtype = TypeId::kNumberTypeFloat16;
  common::AnfAlgo::SetNodeAttr(parallel::RECV_TYPE, TypeIdToType(dtype), neighbor_exchange);

  std::stringstream ss;
  ss << fa_index << "_" << ne_index;
  std::string ss_result = ss.str();
  common::AnfAlgo::SetNodeAttr("FLASH_INDEX", MakeValue<std::string>(ss_result), neighbor_exchange);

  // GROUP
  std::string group = g_device_manager->world_group();
  common::AnfAlgo::SetNodeAttr(parallel::GROUP, MakeValue<std::string>(group), neighbor_exchange);

  // SEND_RANK_IDS, RECV_RANK_IDS
  common::AnfAlgo::SetNodeAttr(parallel::SEND_RANK_IDS, parallel::MakeListValue(send_rank_ids), neighbor_exchange);
  common::AnfAlgo::SetNodeAttr(parallel::RECV_RANK_IDS, parallel::MakeListValue(recv_rank_ids), neighbor_exchange);

  // SEND_SHAPES, RECV_SHAPES
  parallel::Shape shape = neigh_shape;
  parallel::Shapes send_shapes;
  parallel::Shapes recv_shapes;
  for (size_t i = 0; i < send_rank_ids.size(); ++i) {
    send_shapes.push_back(shape);
    recv_shapes.push_back(shape);
  }
  common::AnfAlgo::SetNodeAttr(parallel::SEND_SHAPES, parallel::MakeTupleListValue(send_shapes), neighbor_exchange);
  common::AnfAlgo::SetNodeAttr(parallel::RECV_SHAPES, parallel::MakeTupleListValue(recv_shapes), neighbor_exchange);

  common::AnfAlgo::SetNodeAttr(parallel::COMM_REUSE, MakeValue(true), neighbor_exchange);

  neighbor_exchange->set_scope(input_node->scope());
  return neighbor_exchange;
}

CNodePtr NewFlashAttentionScoreNode(const std::vector<AnfNodePtr> &input_nodes, int fa_index, int ne_index) {
  std::vector<AnfNodePtr> fa_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimFlashAttentionScore->name()))};

  for (size_t i = 0; i < input_nodes.size(); ++i) {
    fa_inputs.push_back(input_nodes[i]);
  }
  auto fa_score = input_nodes[0]->func_graph()->NewCNode(fa_inputs);
  MS_EXCEPTION_IF_NULL(fa_score);

  std::stringstream ss;
  ss << fa_index << "_" << ne_index;
  std::string ss_result = ss.str();
  common::AnfAlgo::SetNodeAttr(FLASH_INDEX, MakeValue<std::string>(ss_result), fa_score);
  fa_score->set_scope(input_nodes[0]->scope());
  return fa_score;
}

CNodePtr NewAddNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  std::vector<AnfNodePtr> add_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimAdd->name())), left_node,
                                        right_node};
  auto add_node = left_node->func_graph()->NewCNode(add_inputs);
  MS_EXCEPTION_IF_NULL(add_node);
  add_node->set_scope(left_node->scope());
  return add_node;
}

CNodePtr NewSubNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> sub_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSub->name())), left_node,
                                        right_node};
  auto sub_node = left_node->func_graph()->NewCNode(sub_inputs);
  MS_EXCEPTION_IF_NULL(sub_node);
  sub_node->set_scope(left_node->scope());
  return sub_node;
}

CNodePtr NewMulNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> mul_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimMul->name())), left_node,
                                        right_node};
  auto mul_node = left_node->func_graph()->NewCNode(mul_inputs);
  MS_EXCEPTION_IF_NULL(mul_node);
  mul_node->set_scope(left_node->scope());
  return mul_node;
}

CNodePtr NewDivNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> div_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimRealDiv->name())),
                                        left_node, right_node};
  auto div_node = left_node->func_graph()->NewCNode(div_inputs);
  MS_EXCEPTION_IF_NULL(div_node);
  div_node->set_scope(left_node->scope());
  return div_node;
}

CNodePtr NewExpNode(const AnfNodePtr &left_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  std::vector<AnfNodePtr> exp_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimExp->name())), left_node};
  auto exp_node = left_node->func_graph()->NewCNode(exp_inputs);
  MS_EXCEPTION_IF_NULL(exp_node);
  exp_node->set_scope(left_node->scope());
  return exp_node;
}

CNodePtr NewMaxNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> max_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimMaximum->name())),
                                        left_node, right_node};
  auto max_node = left_node->func_graph()->NewCNode(max_inputs);
  MS_EXCEPTION_IF_NULL(max_node);
  max_node->set_scope(left_node->scope());
  return max_node;
}

CNodePtr NewCastNode(const AnfNodePtr &tensor_node, const TypeId &dtype) {
  MS_EXCEPTION_IF_NULL(tensor_node);
  auto type_node = NewValueNode(static_cast<int64_t>(dtype));
  std::vector<AnfNodePtr> cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())),
                                         tensor_node, type_node};
  auto cast_node = tensor_node->func_graph()->NewCNode(cast_inputs);

  MS_EXCEPTION_IF_NULL(cast_node);
  common::AnfAlgo::SetNodeAttrSafely(kAttrDstType, TypeIdToType(dtype), cast_node);
  cast_node->set_scope(tensor_node->scope());
  return cast_node;
}

CNodePtr NewTransposeNode(const AnfNodePtr &tensor_node, const AnfNodePtr &tuple, ShapeVector output_shape) {
  MS_EXCEPTION_IF_NULL(tensor_node);
  MS_EXCEPTION_IF_NULL(tuple);
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimTranspose->name())),
                                              tensor_node, tuple};
  auto transpose_node = tensor_node->func_graph()->NewCNode(transpose_inputs);
  MS_EXCEPTION_IF_NULL(transpose_node);
  transpose_node->set_scope(tensor_node->scope());
  return transpose_node;
}

CNodePtr NewTileNode(const AnfNodePtr &tensor_node, const AnfNodePtr &tuple) {
  MS_EXCEPTION_IF_NULL(tensor_node);
  MS_EXCEPTION_IF_NULL(tuple);
  std::vector<AnfNodePtr> tile_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimTile->name())),
                                         tensor_node, tuple};
  auto tile_node = tensor_node->func_graph()->NewCNode(tile_inputs);
  MS_EXCEPTION_IF_NULL(tile_node);
  tile_node->set_scope(tensor_node->scope());
  return tile_node;
}

tensor::TensorPtr make_mask_tensor(TypeId type_id, ShapeVector shape, uint8_t value, bool is_causle) {
  tensor::TensorPtr mask_tensor = std::make_shared<mindspore::tensor::Tensor>(type_id, shape);
  int tensor_size = SizeToInt(mask_tensor->data().size());
  uint8_t *uint8_data = reinterpret_cast<uint8_t *>(mask_tensor->data_c());
  if (!is_causle) {
    for (int i = 0; i < tensor_size; ++i) {
      uint8_data[i] = value;
    }
  } else {
    for (int i = 0; i < shape[kIndex0]; ++i) {
      for (int j = 0; j < shape[kIndex1]; ++j) {
        if (i >= j) {
          uint8_data[i * shape[kIndex0] + j] = 0;
        } else {
          uint8_data[i * shape[kIndex0] + j] = 1;
        }
      }
    }
  }
  return mask_tensor;
}

AnfNodePtr GetActualMask(int index, int64_t rank_id, TypeId mask_dtype, ShapeVector mask_shape) {
  // index: the epoch
  AnfNodePtr actual_mask;
  if (index == 0) {
    auto mask_tensor = make_mask_tensor(mask_dtype, mask_shape, 0, true);
    actual_mask = NewValueNode(MakeValue(mask_tensor));
  } else if (index <= rank_id) {
    auto mask_tensor = make_mask_tensor(mask_dtype, mask_shape, 0, false);
    actual_mask = NewValueNode(MakeValue(mask_tensor));
  } else {
    auto mask_tensor = make_mask_tensor(mask_dtype, mask_shape, 1, false);
    actual_mask = NewValueNode(MakeValue(mask_tensor));
  }
  return actual_mask;
}

int64_t GetUDMaskIndex(int index, int64_t pos, int64_t split_num) {
  int64_t step_index = pos - index;
  return step_index >= 0 ? step_index : split_num + step_index;
}

int64_t GetPosInSpDevice(std::shared_ptr<FlashAttentionScoreInfo> flash_score_info_ptr, int64_t rank_id) {
  auto rankList = flash_score_info_ptr->GetSPRankList();
  int64_t pos = -1;
  for (size_t rank_list_idx = 0; rank_list_idx < rankList.size(); ++rank_list_idx) {
    if (rank_id == rankList[rank_list_idx]) {
      pos = rank_list_idx;
    }
  }
  return pos;
}

void GetBSHFromShape(int64_t input_layout, Shape q_shape, Shape kv_shape, int64_t *fa_b, int64_t *fa_s1, int64_t *fa_h1,
                     int64_t *fa_s2) {
  if (input_layout == FASInputLayoutMode::BSH) {
    *fa_b = q_shape[kIndex0];
    *fa_s1 = q_shape[kIndex1];
    *fa_h1 = q_shape[kIndex2];
    *fa_s2 = kv_shape[kIndex1];
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    *fa_b = q_shape[kIndex0];
    *fa_s1 = q_shape[kIndex2];
    *fa_h1 = q_shape[kIndex1] * q_shape[kIndex3];
    *fa_s2 = kv_shape[kIndex2];
  }
}

ValuePtr GetFlashIndexString(int fa_index, int index) {
  std::stringstream ss;
  ss << fa_index << "_" << index;
  std::string ss_result = ss.str();
  return MakeValue<std::string>(ss_result);
}

void UpdateAttentionOutput(CNodePtr *history_max, CNodePtr *history_sum, CNodePtr *acc_attention,
                           const CNodePtr &softmax_max, const CNodePtr &softmax_sum, CNodePtr attention_output,
                           int64_t fa_b, int64_t fa_s1, int64_t fa_n1, int64_t fa_h1, int64_t input_layout,
                           int fa_index, int index) {
  auto temp_max = NewMaxNode(*history_max, softmax_max);
  auto m_h_sub_temp = NewSubNode(*history_max, temp_max);
  auto m_i_sub_temp = NewSubNode(softmax_max, temp_max);
  auto e_m_h_temp = NewExpNode(m_h_sub_temp);
  auto e_m_i_temp = NewExpNode(m_i_sub_temp);
  auto e_l_h = NewMulNode(e_m_h_temp, *history_sum);
  auto e_l_i = NewMulNode(e_m_i_temp, softmax_sum);
  auto l = NewAddNode(e_l_h, e_l_i);
  auto e_m_h_div = NewDivNode(e_l_h, l);
  auto e_m_i_div = NewDivNode(e_l_i, l);
  auto e_m_h_div_split = NewSplitNode(e_m_h_div, 3, 8);
  auto e_m_h_div_item = NewTupleGetItemNode(e_m_h_div_split, 0);
  auto e_m_h_div_concat = NewTileNode(e_m_h_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  auto e_m_i_div_split = NewSplitNode(e_m_i_div, 3, 8);
  auto e_m_i_div_item = NewTupleGetItemNode(e_m_i_div_split, 0);
  auto e_m_i_div_concat = NewTileNode(e_m_i_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  if (input_layout == FASInputLayoutMode::BSH) {
    (*acc_attention) = NewReshapeNode(*acc_attention, {fa_b, fa_s1, fa_n1, fa_h1 / fa_n1}, TypeId::kNumberTypeFloat16);
    attention_output =
      NewReshapeNode(attention_output, {fa_b, fa_s1, fa_n1, fa_h1 / fa_n1}, TypeId::kNumberTypeFloat16);
    AnfNodePtr tmp_tup = parallel::CreateTuple({0, 2, 1, 3});
    (*acc_attention) = NewTransposeNode(*acc_attention, tmp_tup, {fa_b, fa_n1, fa_s1, fa_h1 / fa_n1});
    attention_output = NewTransposeNode(attention_output, tmp_tup, {fa_b, fa_n1, fa_s1, fa_h1 / fa_n1});
  }
  (*acc_attention) = NewCastNode(*acc_attention, TypeId::kNumberTypeFloat32);
  attention_output = NewCastNode(attention_output, TypeId::kNumberTypeFloat32);
  auto weighted_history = NewMulNode(e_m_h_div_concat, *acc_attention);
  auto weighted_attention = NewMulNode(e_m_i_div_concat, attention_output);
  (*acc_attention) = NewAddNode(weighted_history, weighted_attention);
  common::AnfAlgo::SetNodeAttr(kAttrAccumulatedAttention, MakeValue(1), *acc_attention);
  common::AnfAlgo::SetNodeAttr("FLASH_INDEX", GetFlashIndexString(fa_index, index), *acc_attention);
  if (input_layout == FASInputLayoutMode::BSH) {
    auto tmp_tup1 = parallel::CreateTuple({0, 2, 1, 3});
    (*acc_attention) = NewTransposeNode(*acc_attention, tmp_tup1, {fa_b, fa_s1, fa_n1, fa_h1 / fa_n1});
    (*acc_attention) = NewReshapeNode(*acc_attention, {fa_b, fa_s1, fa_h1}, TypeId::kNumberTypeFloat32);
  }
  (*history_max) = temp_max;
  (*history_sum) = l;
}

CNodePtr CreateReplaceFSPGraph(const FuncGraphManagerPtr &manager,
                               const std::vector<CNodePtr> &origin_nodes_topological, const CNodePtr &fa_score_node,
                               FSPInfo *fsp_info, int fa_index) {
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }

  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);

  int64_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  int64_t send_rank_id = fsp_info->GetSendRankId(), recv_rank_id = fsp_info->GetRecvRankId();
  std::vector<bool> is_input_passed = fsp_info->GetIsInputPassed();

  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto input_layout = flash_score_info_ptr->input_layout();
  if (input_layout != FASInputLayoutMode::BSH && input_layout != FASInputLayoutMode::BNSD) {
    return nullptr;
  }

  CNodePtr attn_mask_split_node;
  if (is_input_passed[ops::kFlashAttentionScoreInputAttnMaskIndex]) {
    auto attn_mask_shape = operator_info->inputs_tensor_info()[kIndex6].tensor_layout().base_slice_shape().array();
    auto attn_mask_node =
      fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex + 1);
    attn_mask_split_node = NewSplitNode(attn_mask_node, attn_mask_shape.size() - kIndex1,
                                        sp_num);  // mask has been split in the last dim by sp_num
  }

  int64_t fa_n1 = GetValue<int64_t>(
    fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputHeadNumIndex + 1)
      ->cast<ValueNodePtr>()
      ->value());
  int64_t fa_b, fa_s1, fa_h1, fa_s2;
  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2);

  CNodePtr local_fa_node, kv_received_tuple, softmax_max, softmax_sum, softmax_out, attention_output;
  CNodePtr history_max, history_sum, acc_attention;
  AnfNodePtr actual_mask;
  for (int i = 0; i < sp_num; ++i) {
    // i: the epoch of flash_sp; rank_id:
    // fa_s1:shape[0]  fa_s2:shape[1]
    std::vector<AnfNodePtr> kv_nodes = {key_node, value_node};
    auto kv_tuple = NewMakeTupleNode(kv_nodes);
    auto kv_concat = NewConcatNode(kv_tuple, 0);
    std::vector<AnfNodePtr> concat_tuple = {kv_concat};
    auto kv_concat_tuple = NewMakeTupleNode(concat_tuple);
    if (i != sp_num - 1) {
      auto neigh_shape = kv_shape;
      neigh_shape[0] = neigh_shape[0] * kIndex2;
      kv_received_tuple =
        NewNeighborExchangeNode(kv_concat_tuple, {send_rank_id}, {recv_rank_id}, fa_index, i, neigh_shape);
    }

    auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
    // the index of flash_score_info_ptr->GetSPRankList() where the rank_id locates
    if (is_input_passed[ops::kFlashAttentionScoreInputAttnMaskIndex] && attn_mask_split_node != nullptr) {
      auto pos_index = GetUDMaskIndex(i, pos, sp_num);
      actual_mask = NewTupleGetItemNode(attn_mask_split_node, pos_index);
    } else {
      actual_mask = GetActualMask(i, pos, TypeId::kNumberTypeUInt8, Shape{fa_s1, fa_s2});
    }
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = key_node;
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = value_node;
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = actual_mask;

    local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, i);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);

    if (i != sp_num - 1) {
      auto kv_exchanged_item = NewTupleGetItemNode(kv_received_tuple, kIndex0);
      auto kv_split = NewSplitNode(kv_exchanged_item, kIndex0, kIndex2);
      key_node = NewTupleGetItemNode(kv_split, kIndex0);
      value_node = NewTupleGetItemNode(kv_split, kIndex1);
    }

    softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
    softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);
    attention_output = NewTupleGetItemNode(local_fa_node, kIndex3);

    if (i == 0) {
      acc_attention = attention_output->cast<CNodePtr>();
      history_max = softmax_max->cast<CNodePtr>();
      history_sum = softmax_sum->cast<CNodePtr>();
    } else {
      UpdateAttentionOutput(&history_max, &history_sum, &acc_attention, softmax_max, softmax_sum, attention_output,
                            fa_b, fa_s1, fa_n1, fa_h1, input_layout, fa_index, i);
    }
  }
  acc_attention = NewCastNode(acc_attention, TypeId::kNumberTypeFloat16);
  softmax_out = NewTupleGetItemNode(local_fa_node, kIndex2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

void CreateAndReplaceFAScore(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                             const CNodePtr &fa_score_node, FSPInfo *fsp_info, int i) {
  auto cnode = CreateReplaceFSPGraph(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
  MS_EXCEPTION_IF_NULL(cnode);
  (void)manager->Replace(fa_score_node, cnode);
}

bool CheckUserSettings(const FuncGraphPtr &fg, FSPInfo *fsp_info) {
  fsp_info->DisplayInfo();

  int64_t sp_num = fsp_info->GetSPNum();
  if (sp_num <= 1) {
    MS_LOG(WARNING) << "FSP: To activate the pass, sp num " << sp_num << " should between larger than 1";
    return false;
  }
  return true;
}
}  // namespace

bool SetFlashSP(const FuncGraphPtr &func_graph) {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto ret = func_graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);

  std::vector<CNodePtr> fa_score_nodes = FindFWFlashAttentionScore(manager, origin_nodes_topological);
  if (fa_score_nodes.size() == 0) {
    return false;
  }

  for (size_t i = 0; i < fa_score_nodes.size(); ++i) {
    auto fa_score_node = fa_score_nodes[i];
    auto fa_score_node_prim = GetCNodePrimitive(fa_score_node);
    MS_EXCEPTION_IF_NULL(fa_score_node_prim);
    if (!fa_score_node_prim->HasAttr(parallel::ENABLE_RING_ATTENTION) ||
        !GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RING_ATTENTION)))) {
      continue;
    }

    auto fsp_info = FSPInfo(fa_score_node);
    if (!CheckUserSettings(func_graph, &fsp_info)) {
      return false;
    }

    manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto orders = func_graph->GetOrderedCnodes();
    std::vector<CNodePtr> nodes_topological(orders.cbegin(), orders.cend());
    CreateAndReplaceFAScore(manager, nodes_topological, fa_score_node, &fsp_info, i);
  }
  return true;
}
}  // namespace parallel
}  // namespace mindspore
