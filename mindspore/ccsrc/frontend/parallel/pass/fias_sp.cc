/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "primitive/other_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/nn_ops.h"
#include "infer/make_tuple.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"
#include "include/utils/parallel_context.h"
#include "include/utils/comm_manager.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/jit/ps/action.h"
#include "mindspore/ccsrc/frontend/parallel/graph_util/generate_graph.h"
#include "include/op_enum.h"
#include "mindspore/ops/infer/ops_func_impl/fused_infer_attention_score.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/fused_infer_attention_score_info.h"
#include "frontend/parallel/pass/fias_sp.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
using mindspore::ops::FASInputLayoutMode;
namespace parallel {
FiasSPInfo::FiasSPInfo(CNodePtr fias_node) {
  MS_EXCEPTION_IF_NULL(fias_node);
  std::shared_ptr<OperatorInfo> operator_info = fias_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto fias_info_ptr = std::dynamic_pointer_cast<FusedInferAttentionScoreInfo>(operator_info);
  MS_EXCEPTION_IF_NULL(fias_info_ptr);

  fiassp_num_ = fias_info_ptr->s1_split_num();
  dev_rank_id_ = g_device_manager->global_rank();

  auto rankList = fias_info_ptr->GetSPRankList();
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
using FSPInfo = FiasSPInfo;

std::vector<CNodePtr> FindFWFusedInferAttentionScore(const FuncGraphManagerPtr &manager,
                                                     const std::vector<CNodePtr> &origin_nodes_topological) {
  std::vector<CNodePtr> result;
  for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
    auto cnode = origin_nodes_topological[i];
    if (IsPrimitiveCNode(cnode, prim::kPrimFusedInferAttentionScore)) {
      result.push_back(cnode);
    }
  }
  return result;
}

CNodePtr NewReshapeNode(const AnfNodePtr &input_node, const ShapeVector &output_shape) {
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
  auto func_graph = input_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto concat = func_graph->NewCNode(concat_inputs);
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
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, input_node) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                          input_node, NewValueNode<int64_t>(split_dim),
                                          NewValueNode<int64_t>(split_num)};
  auto func_graph = input_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto split = func_graph->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);
  split->set_scope(input_node->scope());
  return split;
}

CNodePtr NewTupleGetItemNode(const AnfNodePtr &input_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto idx = NewValueNode(SizeToLong(output_index));
  MS_EXCEPTION_IF_NULL(idx);
  MS_EXCEPTION_IF_NULL(input_node->func_graph());
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
  common::AnfAlgo::SetNodeAttr("FIAS_INDEX", MakeValue<std::string>(ss_result), neighbor_exchange);

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

CNodePtr NewFusedInferAttentionScoreNode(const std::vector<AnfNodePtr> &input_nodes, int fa_index, int ne_index) {
  std::vector<AnfNodePtr> fa_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimFusedInferAttentionScore->name()))};

  for (size_t i = 0; i < input_nodes.size(); ++i) {
    fa_inputs.push_back(input_nodes[i]);
  }
  auto fa_score = input_nodes[0]->func_graph()->NewCNode(fa_inputs);
  MS_EXCEPTION_IF_NULL(fa_score);

  std::stringstream ss;
  ss << fa_index << "_" << ne_index;
  std::string ss_result = ss.str();
  common::AnfAlgo::SetNodeAttr(FIAS_INDEX, MakeValue<std::string>(ss_result), fa_score);
  fa_score->set_scope(input_nodes[0]->scope());
  return fa_score;
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

CNodePtr NewLogNode(const AnfNodePtr &left_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  std::vector<AnfNodePtr> log_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimLog->name())), left_node};
  auto log_node = left_node->func_graph()->NewCNode(log_inputs);
  MS_EXCEPTION_IF_NULL(log_node);
  log_node->set_scope(log_node->scope());
  return log_node;
}

CNodePtr NewSigmoidNode(const AnfNodePtr &left_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  std::vector<AnfNodePtr> sigmoid_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSigmoid->name())),
                                            left_node};
  auto sigmoid_node = left_node->func_graph()->NewCNode(sigmoid_inputs);
  MS_EXCEPTION_IF_NULL(sigmoid_node);
  sigmoid_node->set_scope(left_node->scope());
  return sigmoid_node;
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

CNodePtr NewTransposeNode(const AnfNodePtr &tensor_node, const AnfNodePtr &tuple) {
  MS_EXCEPTION_IF_NULL(tensor_node);
  MS_EXCEPTION_IF_NULL(tuple);
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimTranspose->name())),
                                              tensor_node, tuple};
  auto transpose_node = tensor_node->func_graph()->NewCNode(transpose_inputs);
  MS_EXCEPTION_IF_NULL(transpose_node);
  transpose_node->set_scope(tensor_node->scope());
  return transpose_node;
}

int64_t GetPosInSpDevice(std::shared_ptr<FusedInferAttentionScoreInfo> fias_info_ptr, int64_t rank_id) {
  auto rankList = fias_info_ptr->GetSPRankList();
  int64_t pos = -1;
  for (size_t rank_list_idx = 0; rank_list_idx < rankList.size(); ++rank_list_idx) {
    if (rank_id == rankList[rank_list_idx]) {
      pos = static_cast<int64_t>(rank_list_idx);
    }
  }
  return pos;
}

int64_t GetUDMaskIndex(int index, int64_t pos, int64_t split_num) {
  int64_t step_index = pos - index;
  return step_index >= 0 ? step_index : split_num + step_index;
}

void GetLayoutInfo(const CNodePtr &fa_score_node, Shape *q_shape, Shape *kv_shape, int64_t *fa_b, int64_t *fa_s1,
                   int64_t *fa_h1, int64_t *fa_s2, int64_t *fa_n1, int64_t *input_layout) {
  MS_EXCEPTION_IF_NULL(fa_score_node);
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto fias_info_ptr = std::dynamic_pointer_cast<FusedInferAttentionScoreInfo>(operator_info);
  MS_EXCEPTION_IF_NULL(fias_info_ptr);
  *input_layout = fias_info_ptr->input_layout();
  *q_shape = operator_info->inputs_tensor_info_new()[kIndex0]->GetValue().tensor_layout().base_slice_shape().array();
  *kv_shape = operator_info->inputs_tensor_info_new()[kIndex1]
                ->GetElement(0)
                ->GetValue()
                .tensor_layout()
                .base_slice_shape()
                .array();
  auto fa_input =
    fa_score_node->input(ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputNumHeadsIndex + 1);
  MS_EXCEPTION_IF_NULL(fa_input);
  auto fa_input_node = fa_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(fa_input_node);
  *fa_n1 = GetValue<int64_t>(fa_input_node->value());
  if (*input_layout == FASInputLayoutMode::BSH) {
    *fa_b = (*q_shape)[kIndex0];
    *fa_s1 = (*q_shape)[kIndex1];
    *fa_h1 = (*q_shape)[kIndex2];
    *fa_s2 = (*kv_shape)[kIndex1];
  } else if (*input_layout == FASInputLayoutMode::BNSD) {
    *fa_b = (*q_shape)[kIndex0];
    *fa_s1 = (*q_shape)[kIndex2];
    *fa_h1 = (*q_shape)[kIndex1] * (*q_shape)[kIndex3];
    *fa_s2 = (*kv_shape)[kIndex2];
  }
}

void UpdateAttentionOutput(const CNodePtr &softmax_lse, CNodePtr attention_output, int64_t fa_b, int64_t fa_s1,
                           int64_t fa_n1, int64_t fa_h1, int64_t input_layout, CNodePtr *acc_attention,
                           CNodePtr *history_lse) {
  auto sub_lse = NewSubNode(*history_lse, softmax_lse);
  auto sigmoid_lse = NewSigmoidNode(sub_lse);
  auto log_lse = NewLogNode(sigmoid_lse);
  auto update_lse = NewSubNode(*history_lse, log_lse);

  if (input_layout == FASInputLayoutMode::BSH) {
    (*acc_attention) = NewReshapeNode(*acc_attention, {fa_b, fa_s1, fa_n1, fa_h1 / fa_n1});
    attention_output = NewReshapeNode(attention_output, {fa_b, fa_s1, fa_n1, fa_h1 / fa_n1});
    AnfNodePtr tmp_tup = parallel::CreateTuple({0, 2, 1, 3});
    (*acc_attention) = NewTransposeNode(*acc_attention, tmp_tup);
    attention_output = NewTransposeNode(attention_output, tmp_tup);
  }
  (*acc_attention) = NewCastNode(*acc_attention, TypeId::kNumberTypeFloat32);
  attention_output = NewCastNode(attention_output, TypeId::kNumberTypeFloat32);

  auto sub_lse_1 = NewSubNode(softmax_lse, *history_lse);
  auto sigmoid_lse_1 = NewSigmoidNode(sub_lse_1);
  auto sub_out = NewSubNode(*acc_attention, attention_output);
  auto mul_out = NewMulNode(sigmoid_lse_1, sub_out);
  (*acc_attention) = NewSubNode(*acc_attention, mul_out);

  if (input_layout == FASInputLayoutMode::BSH) {
    (*acc_attention) = NewTransposeNode(*acc_attention, parallel::CreateTuple({0, 2, 1, 3}));
    (*acc_attention) = NewReshapeNode(*acc_attention, {fa_b, fa_s1, fa_h1});
  }
  (*history_lse) = update_lse;
}

void SetFusedFAInputs(int index, const CNodePtr &fa_score_node, const AnfNodePtr &key_node,
                      const AnfNodePtr &value_node, std::vector<AnfNodePtr> *fa_inputs, int64_t rank_id,
                      int64_t sp_num) {
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto fias_info_ptr = std::dynamic_pointer_cast<FusedInferAttentionScoreInfo>(operator_info);
  auto pos = GetPosInSpDevice(fias_info_ptr, rank_id);
  auto pos_index = GetUDMaskIndex(index, pos, sp_num);
  AnfNodePtr actual_mask;
  auto attn_node =
    fa_score_node->input(ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputAttnMaskIndex + 1);
  if (!IsValueNode<None>(attn_node)) {
    auto attn_shape =
      operator_info->inputs_tensor_info_new()[kIndex4]->GetValue().tensor_layout().base_slice_shape().array();
    auto attn_split_node = NewSplitNode(attn_node, attn_shape.size() - 1, sp_num);
    actual_mask = NewTupleGetItemNode(attn_split_node, pos_index);
  }
  if (index == 0) {
    (*fa_inputs)[ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputKeyIndex] = key_node;
    (*fa_inputs)[ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputValueIndex] = value_node;
  } else {
    std::vector<AnfNodePtr> key_node_v = {key_node};
    std::vector<AnfNodePtr> value_node_v = {value_node};
    auto key_node_v_tuple = NewMakeTupleNode(key_node_v);
    auto value_node_v_tuple = NewMakeTupleNode(value_node_v);
    (*fa_inputs)[ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputKeyIndex] = key_node_v_tuple;
    (*fa_inputs)[ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputValueIndex] =
      value_node_v_tuple;
  }

  if (actual_mask != nullptr) {
    (*fa_inputs)[ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputAttnMaskIndex] = actual_mask;
  }
}

void KVInputSwitch(const CNodePtr &fa_score_node, const Shape &kv_shape, int64_t sp_num, int64_t send_rank_id,
                   int64_t recv_rank_id, int64_t fa_b, int64_t fa_s1, int64_t fa_h1, int64_t fa_n1,
                   int64_t input_layout, int64_t rank_id, int fa_index, CNodePtr *local_fa_node,
                   CNodePtr *acc_attention) {
  CNodePtr kv_received_tuple;
  CNodePtr history_lse;
  CNodePtr key_node_item;
  CNodePtr value_node_item;
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }
  auto key_node =
    fa_score_node->input(ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputKeyIndex + 1);
  auto value_node =
    fa_score_node->input(ops::FusedInferAttentionScoreInputIndex::kFusedInferAttentionScoreInputValueIndex + 1);

  for (int i = 0; i < sp_num; ++i) {
    if (i == 0) {
      key_node_item = NewTupleGetItemNode(key_node, kIndex0);
      value_node_item = NewTupleGetItemNode(value_node, kIndex0);
    } else {
      key_node_item = key_node->cast<CNodePtr>();
      value_node_item = value_node->cast<CNodePtr>();
    }
    std::vector<AnfNodePtr> kv_nodes = {key_node_item, value_node_item};
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
    SetFusedFAInputs(i, fa_score_node, key_node, value_node, &fa_inputs, rank_id, sp_num);

    *local_fa_node = NewFusedInferAttentionScoreNode(fa_inputs, fa_index, i);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, *local_fa_node);
    if (i != sp_num - 1) {
      auto kv_exchanged_item = NewTupleGetItemNode(kv_received_tuple, kIndex0);
      auto kv_split = NewSplitNode(kv_exchanged_item, kIndex0, kIndex2);
      key_node = NewTupleGetItemNode(kv_split, kIndex0);
      value_node = NewTupleGetItemNode(kv_split, kIndex1);
    }
    CNodePtr attention_output = NewTupleGetItemNode(*local_fa_node, kIndex0);
    CNodePtr softmax_lse = NewTupleGetItemNode(*local_fa_node, kIndex1);
    if (i == 0) {
      *acc_attention = attention_output->cast<CNodePtr>();
      history_lse = softmax_lse->cast<CNodePtr>();
    } else {
      UpdateAttentionOutput(softmax_lse, attention_output, fa_b, fa_s1, fa_n1, fa_h1, input_layout, acc_attention,
                            &history_lse);
    }
  }
}

CNodePtr CreateReplaceFSPGraph(const FuncGraphManagerPtr &manager,
                               const std::vector<CNodePtr> &origin_nodes_topological, const CNodePtr &fa_score_node,
                               FSPInfo *fsp_info, int fa_index) {
  int64_t sp_num = fsp_info->GetSPNum();
  int64_t rank_id = fsp_info->GetRankId();
  int64_t send_rank_id = fsp_info->GetSendRankId();
  int64_t recv_rank_id = fsp_info->GetRecvRankId();
  int64_t fa_b = 0;
  int64_t fa_s1 = 0;
  int64_t fa_s2 = 0;
  int64_t fa_h1 = 0;
  int64_t fa_n1 = 0;
  int64_t input_layout = 0;
  Shape q_shape;
  Shape kv_shape;
  GetLayoutInfo(fa_score_node, &q_shape, &kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_n1, &input_layout);

  CNodePtr new_fa_node;
  CNodePtr acc_attention;
  KVInputSwitch(fa_score_node, kv_shape, sp_num, send_rank_id, recv_rank_id, fa_b, fa_s1, fa_h1, fa_n1, input_layout,
                rank_id, fa_index, &new_fa_node, &acc_attention);

  acc_attention = NewCastNode(acc_attention, TypeId::kNumberTypeFloat16);
  auto softmax_out = NewTupleGetItemNode(new_fa_node, kIndex1);
  std::vector<AnfNodePtr> output_tuple = {acc_attention, softmax_out};
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
    MS_LOG(WARNING) << "FSP: To activate the pass, sp num " << sp_num << " should larger than 1";
    return false;
  }
  return true;
}
}  // namespace

bool SetFiasSP(const FuncGraphPtr &func_graph) {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = func_graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());

  std::vector<CNodePtr> fias_nodes = FindFWFusedInferAttentionScore(manager, origin_nodes_topological);
  if (fias_nodes.size() == 0) {
    return false;
  }

  for (size_t i = 0; i < fias_nodes.size(); ++i) {
    auto fias_node = fias_nodes[i];
    auto fias_node_prim = GetCNodePrimitive(fias_node);
    MS_EXCEPTION_IF_NULL(fias_node_prim);
    if (!fias_node_prim->HasAttr(parallel::ENABLE_RING_ATTENTION) ||
        !GetValue<bool>((fias_node_prim->GetAttr(parallel::ENABLE_RING_ATTENTION)))) {
      continue;
    }

    auto fsp_info = FSPInfo(fias_node);
    if (!CheckUserSettings(func_graph, &fsp_info)) {
      return false;
    }

    manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    orders = func_graph->GetOrderedCnodes();
    std::vector<CNodePtr> nodes_topological(orders.cbegin(), orders.cend());
    CreateAndReplaceFAScore(manager, nodes_topological, fias_node, &fsp_info, i);
  }
  return true;
}
}  // namespace parallel
}  // namespace mindspore
