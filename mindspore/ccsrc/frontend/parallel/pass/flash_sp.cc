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
#include "frontend/parallel/pass/flash_sp.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/parameter_manager.h"

namespace mindspore {
using mindspore::ops::FASInputLayoutMode;
namespace parallel {
FlashSPInfo::FlashSPInfo(CNodePtr fa_score_node) {
  MS_EXCEPTION_IF_NULL(fa_score_node);
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto input_layout = flash_score_info_ptr->input_layout();
  if (input_layout != FASInputLayoutMode::BSH && input_layout != FASInputLayoutMode::BNSD) {
    MS_LOG(EXCEPTION) << "ring attention only supports BSH and BNSD layout";
  }
  MS_EXCEPTION_IF_NULL(flash_score_info_ptr);

  flashsp_num_ = flash_score_info_ptr->s1_split_num();
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
  actual_seq_length_size_ = flash_score_info_ptr->GetActualSeqLengthSize();
}
namespace {
using CNodePtrPair = std::pair<CNodePtr, CNodePtr>;
using FSPInfo = FlashSPInfo;

std::vector<CNodePtr> FindFWFlashAttentionScore(const FuncGraphManagerPtr &manager,
                                                const std::vector<AnfNodePtr> &origin_nodes_topological) {
  std::vector<CNodePtr> result;
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    return result;
  }
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

CNodePtr NewFlashAttentionScoreNode(const std::vector<AnfNodePtr> &input_nodes, int fa_index, int ne_index, bool flag) {
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
  if (flag) {
    fa_score->AddPrimalAttr("FLASH_INDEX", MakeValue<std::string>(ss_result));
  } else {
    fa_score->AddPrimalAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(ss_result));
  }
  fa_score->set_scope(input_nodes[0]->scope());
  common::AnfAlgo::SetNodeAttr(FLASH_INDEX, MakeValue<std::string>(ss_result), fa_score);
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

CNodePtr NewReluNode(const AnfNodePtr &left_node, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(left_node);
  std::vector<AnfNodePtr> relu_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReLU->name())), left_node};
  auto relu_node = node->func_graph()->NewCNode(relu_inputs);
  MS_EXCEPTION_IF_NULL(relu_node);
  relu_node->set_scope(node->scope());
  return relu_node;
}

tensor::TensorPtr make_mask_tensor(TypeId type_id, ShapeVector shape, uint8_t value, bool is_causle) {
  tensor::TensorPtr mask_tensor = std::make_shared<mindspore::tensor::Tensor>(type_id, shape);
  int64_t tensor_size = SizeToLong(mask_tensor->data().size());
  uint8_t *uint8_data = reinterpret_cast<uint8_t *>(mask_tensor->data_c());
  if (!is_causle) {
    for (int64_t i = 0; i < tensor_size; ++i) {
      uint8_data[i] = value;
    }
  } else {
    for (int64_t i = 0; i < shape[kIndex0]; ++i) {
      for (int64_t j = 0; j < shape[kIndex1]; ++j) {
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

CNodePtr NewRangeNode(const FuncGraphPtr &fg, int64_t actual_size) {
  MS_EXCEPTION_IF_NULL(fg);
  constexpr int range_op_max_value = 1000000;
  std::vector<AnfNodePtr> range_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimRange->name())),
                                          NewValueNode<int64_t>(1), NewValueNode<int64_t>(actual_size + 1),
                                          NewValueNode<int64_t>(1), NewValueNode<int64_t>(range_op_max_value)};
  auto node_range = fg->NewCNode(range_inputs);
  MS_EXCEPTION_IF_NULL(node_range);
  return node_range;
}

CNodePtr NewRollNode(const AnfNodePtr &actual_seq) {
  MS_EXCEPTION_IF_NULL(actual_seq);
  std::vector<AnfNodePtr> roll_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimRoll->name())), actual_seq,
                                         parallel::CreateTuple({1}), parallel::CreateTuple({0})};
  auto node_roll = actual_seq->func_graph()->NewCNode(roll_inputs);
  MS_EXCEPTION_IF_NULL(node_roll);
  return node_roll;
}

CNodePtr NewRepeatNode(const AnfNodePtr &range, const AnfNodePtr &repeat_nums, int64_t output_size) {
  MS_EXCEPTION_IF_NULL(range);
  MS_EXCEPTION_IF_NULL(repeat_nums);
  std::vector<AnfNodePtr> repeat_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimRepeatInterleaveTensor->name())), range, repeat_nums,
    NewValueNode<int64_t>(0), NewValueNode<int64_t>(output_size)};
  auto node_repeat = repeat_nums->func_graph()->NewCNode(repeat_inputs);
  MS_EXCEPTION_IF_NULL(node_repeat);
  return node_repeat;
}

CNodePtr NewEqualNode(const AnfNodePtr &tensor1, const AnfNodePtr &tensor2) {
  MS_EXCEPTION_IF_NULL(tensor1);
  MS_EXCEPTION_IF_NULL(tensor2);
  std::vector<AnfNodePtr> equal_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimEqual->name())), tensor1,
                                          tensor2};
  auto node_equal = tensor1->func_graph()->NewCNode(equal_inputs);
  MS_EXCEPTION_IF_NULL(node_equal);
  return node_equal;
}

CNodePtr NewTriuNode(const AnfNodePtr &tensor, const AnfNodePtr &diag) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(diag);
  std::vector<AnfNodePtr> triu_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimTriu->name())), tensor,
                                         diag};
  auto node_triu = tensor->func_graph()->NewCNode(triu_inputs);
  MS_EXCEPTION_IF_NULL(node_triu);
  return node_triu;
}

void GenerateActualMask(int index, int64_t rank_id, int64_t sp_num, int64_t actual_shape, const ShapeVector &s_shape,
                        const AnfNodePtr &actual_node, vector<AnfNodePtr> *node_masks) {
  AnfNodePtr actual_input = nullptr;
  auto actual_cnode = actual_node->cast<CNodePtr>();
  if (actual_cnode != nullptr && actual_cnode->inputs().size() > 1) {
    actual_input = actual_cnode->input(1);  // fa actual seq input is ValueToTuple, so get it first input
  }
  if (actual_input == nullptr || actual_shape == 0) {
    MS_LOG(INFO) << "Input of actual_seq_length is required when enable eod reset attention mask.";
    return;
  }

  auto node_range = NewRangeNode(actual_input->func_graph(), actual_shape);

  auto node_roll = NewRollNode(actual_input);

  auto node_sub = NewSubNode(actual_input, node_roll);

  tensor::TensorPtr const_tensor =
    std::make_shared<mindspore::tensor::Tensor>(TypeId::kNumberTypeInt64, Shape{actual_shape});
  int64_t *int_data = reinterpret_cast<int64_t *>(const_tensor->data_c());
  int_data[0] = sp_num * s_shape[0];
  for (int i = 1; i < actual_shape; ++i) {
    int_data[i] = 0;
  }
  auto const_seq = NewValueNode(MakeValue(const_tensor));

  auto node_add = NewAddNode(node_sub, const_seq);

  auto node_repeat = NewRepeatNode(node_range, node_add, sp_num * s_shape[0]);

  auto node_split = NewSplitNode(node_repeat, 0, sp_num);

  auto node_get_tuple_w = NewTupleGetItemNode(node_split, index);

  auto node_get_tuple_h = NewTupleGetItemNode(node_split, rank_id);

  auto node_w_tile = NewTileNode(node_get_tuple_w, parallel::CreateTuple({s_shape[0], 1}));

  auto node_reshape = NewReshapeNode(node_get_tuple_h, {s_shape[0], 1}, TypeId::kNumberTypeInt64);

  auto node_h_tile = NewTileNode(node_reshape, parallel::CreateTuple({1, s_shape[1]}));

  auto node_equal = NewEqualNode(node_h_tile, node_w_tile);

  std::vector<AnfNodePtr> logical_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimLogicalNot->name()))};
  if (rank_id * s_shape[0] == index * s_shape[1]) {
    auto node_tran_triu = NewTransposeNode(node_equal, parallel::CreateTuple({1, 0}));
    auto node_tril = NewTriuNode(node_tran_triu, NewValueNode<int64_t>(0));
    auto node_tran_tril = NewTransposeNode(node_tril, parallel::CreateTuple({1, 0}));
    logical_inputs.emplace_back(node_tran_tril);
  } else if (rank_id * s_shape[0] < index * s_shape[1]) {
    auto value_node0 =
      NewValueNode(MakeValue(make_mask_tensor(TypeId::kNumberTypeInt64, {s_shape[0], s_shape[1]}, 0, false)));
    logical_inputs.emplace_back(value_node0);
  } else {
    logical_inputs.emplace_back(node_equal);
  }

  auto node_logical = node_equal->func_graph()->NewCNode(logical_inputs);

  node_masks->emplace_back(node_logical);
}

tensor::TensorPtr make_start_mask_tensor(TypeId type_id, ShapeVector shape) {
  tensor::TensorPtr mask_tensor = std::make_shared<mindspore::tensor::Tensor>(type_id, shape);
  uint8_t *uint8_data = reinterpret_cast<uint8_t *>(mask_tensor->data_c());
  auto k0 = shape[kIndex0] / 2;
  auto k1 = shape[kIndex1] / 2;
  for (int i = 0; i < shape[kIndex0]; ++i) {
    for (int j = 0; j < shape[kIndex1]; ++j) {
      if (i < k0 && j < k1) {
        if (i >= j) {
          uint8_data[i * shape[kIndex0] + j] = 0;
        } else {
          uint8_data[i * shape[kIndex0] + j] = 1;
        }
      } else if (i >= k0 && j >= k1) {
        if (i >= j) {
          uint8_data[i * shape[kIndex0] + j] = 0;
        } else {
          uint8_data[i * shape[kIndex0] + j] = 1;
        }
      } else {
        uint8_data[i * shape[kIndex0] + j] = 1;
      }
    }
  }
  return mask_tensor;
}

tensor::TensorPtr make_end_mask_tensor(TypeId type_id, ShapeVector shape) {
  tensor::TensorPtr mask_tensor = std::make_shared<mindspore::tensor::Tensor>(type_id, shape);
  uint8_t *uint8_data = reinterpret_cast<uint8_t *>(mask_tensor->data_c());
  auto k0 = shape[kIndex0] / 2;
  auto k1 = shape[kIndex1] / 2;
  for (int i = 0; i < shape[kIndex0]; ++i) {
    for (int j = 0; j < shape[kIndex1]; ++j) {
      if (i >= k0 && j < k1) {
        uint8_data[i * shape[kIndex0] + j] = 0;
      } else {
        uint8_data[i * shape[kIndex0] + j] = 1;
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

int64_t GetRankIndex(int64_t rank_id, int64_t step, int64_t sp_size) {
  std::vector<int> rank_order;
  for (int i = 0; i < sp_size; i++) {
    if (i % (step + 1) == 0) {
      rank_order.push_back(i);
    }
  }
  for (int i = 1; i <= step; i++) {
    for (int j = i; j < sp_size; j += step + 1) {
      rank_order.push_back(j);
    }
  }
  int64_t pos = -1;
  for (size_t rank_list_idx = 0; rank_list_idx < rank_order.size(); ++rank_list_idx) {
    if (rank_id == rank_order[rank_list_idx]) {
      pos = rank_list_idx;
    }
  }
  return pos;
}

void GetBSHFromShape(int64_t input_layout, Shape q_shape, Shape kv_shape, int64_t *fa_b, int64_t *fa_s1, int64_t *fa_h1,
                     int64_t *fa_s2, int64_t *fa_h2) {
  if (input_layout == FASInputLayoutMode::BSH) {
    *fa_b = q_shape[kIndex0];
    *fa_s1 = q_shape[kIndex1];
    *fa_h1 = q_shape[kIndex2];
    *fa_s2 = kv_shape[kIndex1];
    *fa_h2 = kv_shape[kIndex2];
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    *fa_b = q_shape[kIndex0];
    *fa_s1 = q_shape[kIndex2];
    *fa_h1 = q_shape[kIndex1] * q_shape[kIndex3];
    *fa_s2 = kv_shape[kIndex2];
    *fa_h2 = kv_shape[kIndex1] * kv_shape[kIndex3];
  }
}

std::string GetFlashIndexString(int fa_index, int index) {
  std::stringstream ss;
  ss << fa_index << "_" << index;
  std::string ss_result = ss.str();
  return ss_result;
}

CNodePtr CreateDepend(const AnfNodePtr &latter_node, const AnfNodePtr &former_node) {
  MS_EXCEPTION_IF_NULL(latter_node);
  if (former_node == nullptr) {
    return latter_node->cast<CNodePtr>();
  }
  MS_EXCEPTION_IF_NULL(latter_node->func_graph());
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                        latter_node, former_node};
  auto depend = latter_node->func_graph()->NewCNode(depend_inputs);

  MS_EXCEPTION_IF_NULL(depend);

  depend->set_scope(latter_node->scope());
  return depend;
}

CNodePtr CreateDepends(const AnfNodePtr &latter_node, const std::vector<AnfNodePtr> &former_nodes) {
  MS_EXCEPTION_IF_NULL(latter_node);
  auto latter_cnode = latter_node->cast<CNodePtr>();
  for (size_t i = 0; i < former_nodes.size(); ++i) {
    auto former_node = former_nodes[i];
    if (former_node == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(latter_cnode->func_graph());
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          latter_cnode, former_node};
    auto depend = latter_cnode->func_graph()->NewCNode(depend_inputs);

    MS_EXCEPTION_IF_NULL(depend);

    depend->set_scope(latter_node->scope());
    latter_cnode = depend;
  }
  return latter_cnode;
}

CNodePtr NewStridedSliceNode(const AnfNodePtr &tensor_node, const Shape &begin, const Shape &end,
                             const Shape &strides) {
  MS_EXCEPTION_IF_NULL(tensor_node);
  std::vector<AnfNodePtr> stridedslice_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimStridedSlice->name())),
    tensor_node,
    NewValueNode(MakeValue(begin)),
    NewValueNode(MakeValue(end)),
    NewValueNode(MakeValue(strides)),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0)))};
  auto stridedslice_node = tensor_node->func_graph()->NewCNode(stridedslice_inputs);
  MS_EXCEPTION_IF_NULL(stridedslice_node);
  stridedslice_node->set_scope(tensor_node->scope());
  return stridedslice_node;
}

CNodePtr NewSendNode(const AnfNodePtr &send_data, int64_t tag, int64_t dest_rank, const Shape &send_shape,
                     TypeId type_id, const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(send_data);
  Attr attr_tag = std::make_pair(parallel::SR_TAG, MakeValue((tag)));
  Attr attr_rank = std::make_pair(parallel::DEST_RANK, MakeValue(dest_rank));
  Attr attr_group = std::make_pair(parallel::GROUP, MakeValue(group_name));
  Attr attr_group_back = std::make_pair(parallel::GROUP_BACK, MakeValue(group_name));
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_group, attr_group_back};
  auto send_input = ConvertToRealInputs("Send", "Send", AnfNodePtrList{send_data}, attrs);
  auto send_node = send_data->func_graph()->NewCNode(send_input);
  MS_EXCEPTION_IF_NULL(send_node);

  common::AnfAlgo::SetNodeAttr(parallel::SR_TAG, MakeValue(tag), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::DEST_RANK, MakeValue(dest_rank), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP, MakeValue(group_name), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP_BACK, MakeValue(group_name), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::SHAPE, MakeValue(send_shape), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::DTYPE, TypeIdToType(type_id), send_node);
  send_node->set_scope(send_data->scope());
  return send_node;
}

CNodePtr NewReceiveNode(const AnfNodePtr &parameter, int64_t tag, int64_t src_rank, const Shape &recv_shape,
                        TypeId type_id, const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(parameter);
  Attr attr_tag = std::make_pair(parallel::SR_TAG, MakeValue((tag)));
  Attr attr_rank = std::make_pair(parallel::SRC_RANK, MakeValue(src_rank));
  Attr attr_shape = std::make_pair(parallel::SHAPE, MakeValue(recv_shape));
  Attr attr_dtype = std::make_pair(parallel::DTYPE, TypeIdToType(type_id));
  Attr attr_group = std::make_pair(parallel::GROUP, MakeValue(group_name));
  Attr attr_group_back = std::make_pair(parallel::GROUP_BACK, MakeValue(group_name));
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_shape, attr_dtype, attr_group, attr_group_back};
  auto recv_inputs = ConvertToRealInputs("Receive", "Receive", AnfNodePtrList{parameter}, attrs);
  auto recv_node = parameter->func_graph()->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(recv_node);

  common::AnfAlgo::SetNodeAttr(parallel::SR_TAG, MakeValue(tag), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::SRC_RANK, MakeValue(src_rank), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP, MakeValue(group_name), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP_BACK, MakeValue(group_name), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::SHAPE, MakeValue(recv_shape), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::DTYPE, TypeIdToType(type_id), recv_node);
  common::AnfAlgo::SetNodeAttr("flash_tag", MakeValue("True"), recv_node);
  recv_node->set_scope(parameter->scope());
  return recv_node;
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
    (*acc_attention) = NewTransposeNode(*acc_attention, tmp_tup);
    attention_output = NewTransposeNode(attention_output, tmp_tup);
  }
  (*acc_attention) = NewCastNode(*acc_attention, TypeId::kNumberTypeFloat32);
  attention_output = NewCastNode(attention_output, TypeId::kNumberTypeFloat32);
  auto weighted_history = NewMulNode(e_m_h_div_concat, *acc_attention);
  auto weighted_attention = NewMulNode(e_m_i_div_concat, attention_output);
  (*acc_attention) = NewAddNode(weighted_history, weighted_attention);
  common::AnfAlgo::SetNodeAttr(kAttrAccumulatedAttention, MakeValue(1), *acc_attention);
  common::AnfAlgo::SetNodeAttr("FLASH_INDEX", MakeValue<std::string>(GetFlashIndexString(fa_index, index)),
                               *acc_attention);
  if (input_layout == FASInputLayoutMode::BSH) {
    auto tmp_tup1 = parallel::CreateTuple({0, 2, 1, 3});
    (*acc_attention) = NewTransposeNode(*acc_attention, tmp_tup1);
    (*acc_attention) = NewReshapeNode(*acc_attention, {fa_b, fa_s1, fa_h1}, TypeId::kNumberTypeFloat32);
  }
  (*history_max) = temp_max;
  (*history_sum) = l;
}

CNodePtr ConstructSendOMLTensor(const AnfNodePtr &send_softmax_max, const AnfNodePtr &send_softmax_sum,
                                const AnfNodePtr &send_attn_out) {
  auto send_attn_out_fp32 = NewCastNode(send_attn_out, TypeId::kNumberTypeFloat32);
  std::vector<AnfNodePtr> oml_nodes = {send_attn_out_fp32, send_softmax_max, send_softmax_sum};
  auto oml_tuple = NewMakeTupleNode(oml_nodes);
  return NewConcatNode(oml_tuple, kIndex3);
}

void DismantleRecvOMLTensor(const AnfNodePtr &recv_oml_tensor, CNodePtr *cur_attn_out, CNodePtr *cur_softmax_max,
                            CNodePtr *cur_softmax_sum, Shape q_shape) {
  (*cur_attn_out) =
    NewStridedSliceNode(recv_oml_tensor, {0, 0, 0, 0}, {q_shape[0], q_shape[1], q_shape[2], q_shape[3]}, {1, 1, 1, 1});
  (*cur_softmax_max) = NewStridedSliceNode(recv_oml_tensor, {0, 0, 0, q_shape[3]},
                                           {q_shape[0], q_shape[1], q_shape[2], q_shape[3] + 8}, {1, 1, 1, 1});
  (*cur_softmax_sum) = NewStridedSliceNode(recv_oml_tensor, {0, 0, 0, q_shape[3] + 8},
                                           {q_shape[0], q_shape[1], q_shape[2], q_shape[3] + 16}, {1, 1, 1, 1});
}

int64_t GetSendQKVDstRank(int64_t rank, int64_t step, int64_t sp_size) { return (rank + step + 1) % sp_size; }

int64_t GetRecvQKVSrcRank(int64_t rank, int64_t step, int64_t sp_size) { return (rank + sp_size - step - 1) % sp_size; }

int64_t GetSendOMLDstRank(int64_t rank, int64_t step, int64_t sp_size) { return (rank + sp_size - step) % sp_size; }

int64_t GetRecvOMLSrcRank(int64_t rank, int64_t step, int64_t sp_size) { return (rank + step) % sp_size; }

enum TagType {
  query = 0,
  kv_a = 1,
  kv_b = 2,
  oml = 3,
};

int64_t GetSendRecvTag(int64_t src, int64_t dest, TagType data_type) {
  auto src_string = std::to_string(src + 1);
  auto dest_string = std::to_string(dest + 1);
  auto data_type_string = std::to_string(data_type);

  auto res_string = src_string + dest_string + data_type_string;
  return std::stoi(res_string);
}

CNodePtr GetCurrentSendQKVNode(int64_t pos, int64_t step, int64_t inner_step, int64_t sp_num,
                               const AnfNodePtr &query_node, const std::vector<CNodePtr> &send_kv_node,
                               RankList spRankList, const std::string &qkv_group, const AnfNodePtr &pre_node,
                               Shape q_shape, Shape kv_shape) {
  if (step < (sp_num / 2)) {  // [0, sp-1]
    auto send_qkv_dst_rank = GetSendQKVDstRank(pos, step, sp_num);
    if (pos + step + 1 >= sp_num) {  // send q
      if (inner_step == 0) {
        return NewSendNode(CreateDepend(query_node, pre_node), GetSendRecvTag(pos, send_qkv_dst_rank, TagType::query),
                           spRankList[send_qkv_dst_rank], q_shape, TypeId::kNumberTypeFloat16, qkv_group);
      }
    } else {  // send kv
      auto data_type = (inner_step == 0 ? TagType::kv_b : TagType::kv_a);
      auto tmp_node = send_kv_node[(inner_step + 1) % 2];
      if (step < ((sp_num / 2) - 1)) {  // [0, sp-3]
        return NewSendNode(CreateDepend(tmp_node, pre_node), GetSendRecvTag(pos, send_qkv_dst_rank, data_type),
                           spRankList[send_qkv_dst_rank], kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
      } else if (inner_step == 0) {
        return NewSendNode(CreateDepend(tmp_node, pre_node), GetSendRecvTag(pos, send_qkv_dst_rank, data_type),
                           spRankList[send_qkv_dst_rank], kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
      }
    }
  }
  return nullptr;
}

CNodePtr GetCurrentRecvQKVNode(int64_t pos, int64_t step, int64_t inner_step, int64_t sp_num, RankList spRankList,
                               Shape q_shape, Shape kv_shape, const std::string &qkv_group, const AnfNodePtr &pre_node,
                               const AnfNodePtr &tmp_param) {
  if (step < (sp_num / 2)) {  // [0, sp-1]
    auto recv_qkv_src_rank = GetRecvQKVSrcRank(pos, step, sp_num);
    if (pos < step + 1) {
      if (inner_step == 0) {  // recv q
        if (tmp_param != nullptr) {
          return NewReceiveNode(CreateDepend(tmp_param, pre_node),
                                GetSendRecvTag(recv_qkv_src_rank, pos, TagType::query), spRankList[recv_qkv_src_rank],
                                q_shape, TypeId::kNumberTypeFloat16, qkv_group);
        } else {
          return NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, TagType::query),
                                spRankList[recv_qkv_src_rank], q_shape, TypeId::kNumberTypeFloat16, qkv_group);
        }
      }
    } else {  // recv kv
      auto data_type = inner_step == 0 ? TagType::kv_b : TagType::kv_a;
      if (step < (sp_num / 2) - 1) {
        if (tmp_param != nullptr) {
          return NewReceiveNode(CreateDepend(tmp_param, pre_node), GetSendRecvTag(recv_qkv_src_rank, pos, data_type),
                                spRankList[recv_qkv_src_rank], kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
        } else {
          return NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, data_type),
                                spRankList[recv_qkv_src_rank], kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
        }
      } else {
        if (inner_step == 0) {
          if (tmp_param != nullptr) {
            return NewReceiveNode(CreateDepend(tmp_param, pre_node), GetSendRecvTag(recv_qkv_src_rank, pos, data_type),
                                  spRankList[recv_qkv_src_rank], kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
          } else {
            return NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, data_type),
                                  spRankList[recv_qkv_src_rank], kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
          }
        }
      }
    }
  }
  return nullptr;
}

CNodePtr GetCurrentSendOMLNode(int64_t pos, int64_t step, int64_t inner_step, int64_t sp_num, RankList spRankList,
                               const AnfNodePtr &send_softmax_max, const AnfNodePtr &send_softmax_sum,
                               const AnfNodePtr &send_attn_out, const std::string &send_group,
                               const AnfNodePtr &pre_node, Shape q_shape) {
  auto recv_oml_shape = q_shape;
  recv_oml_shape[3] = recv_oml_shape[3] + 16;
  if (step > 0 && step < (sp_num / 2) + 1) {
    if (pos < (sp_num / 2)) {
      auto send_oml_dst_rank = GetSendOMLDstRank(pos, step, sp_num);
      if (pos < step) {
        if (step < (sp_num / 2)) {
          if (inner_step == 1) {
            return NewSendNode(
              CreateDepend(ConstructSendOMLTensor(send_softmax_max, send_softmax_sum, send_attn_out), pre_node),
              GetSendRecvTag(pos, send_oml_dst_rank, TagType::oml), spRankList[send_oml_dst_rank], recv_oml_shape,
              TypeId::kNumberTypeFloat32, send_group);
          }
        } else {
          if (inner_step == 0) {
            return NewSendNode(
              CreateDepend(ConstructSendOMLTensor(send_softmax_max, send_softmax_sum, send_attn_out), pre_node),
              GetSendRecvTag(pos, send_oml_dst_rank, TagType::oml), spRankList[send_oml_dst_rank], recv_oml_shape,
              TypeId::kNumberTypeFloat32, send_group);
          }
        }
      }
    }
  }
  return nullptr;
}

CNodePtr GetCurrentRecvOMLNode(int64_t pos, int64_t step, int64_t inner_step, int64_t sp_num, RankList spRankList,
                               Shape q_shape, const std::string &recv_group, const AnfNodePtr &pre_node,
                               const AnfNodePtr &tmp_param) {
  if (step > 0 && step < (sp_num / 2) + 1) {
    if (pos >= (sp_num / 2)) {
      auto recv_oml_src_rank = GetRecvOMLSrcRank(pos, step, sp_num);
      if (pos + step >= sp_num) {
        if (step < (sp_num / 2)) {
          if (inner_step == 1) {
            auto recv_oml_shape = q_shape;
            recv_oml_shape[3] = recv_oml_shape[3] + 16;
            if (tmp_param != nullptr) {
              return NewReceiveNode(CreateDepend(tmp_param, pre_node),
                                    GetSendRecvTag(recv_oml_src_rank, pos, TagType::oml), spRankList[recv_oml_src_rank],
                                    recv_oml_shape, TypeId::kNumberTypeFloat32, recv_group);
            } else {
              return NewReceiveNode(pre_node, GetSendRecvTag(recv_oml_src_rank, pos, TagType::oml),
                                    spRankList[recv_oml_src_rank], recv_oml_shape, TypeId::kNumberTypeFloat32,
                                    recv_group);
            }
          }
        } else {
          if (inner_step == 0) {
            auto recv_oml_shape = q_shape;
            recv_oml_shape[3] = recv_oml_shape[3] + 16;
            if (tmp_param != nullptr) {
              return NewReceiveNode(CreateDepend(tmp_param, pre_node),
                                    GetSendRecvTag(recv_oml_src_rank, pos, TagType::oml), spRankList[recv_oml_src_rank],
                                    recv_oml_shape, TypeId::kNumberTypeFloat32, recv_group);
            } else {
              return NewReceiveNode(pre_node, GetSendRecvTag(recv_oml_src_rank, pos, TagType::oml),
                                    spRankList[recv_oml_src_rank], recv_oml_shape, TypeId::kNumberTypeFloat32,
                                    recv_group);
            }
          }
        }
      }
    }
  }
  return nullptr;
}

void ChangeQKVToBNSD(AnfNodePtr *query_node, AnfNodePtr *key_node, AnfNodePtr *value_node, Shape *q_shape,
                     Shape *kv_shape, int64_t fa_b, int64_t fa_s1, int64_t fa_h1, int64_t fa_s2, int64_t fa_h2,
                     int64_t fa_d, int64_t fa_n1) {
  (*query_node) = NewReshapeNode((*query_node), {fa_b, fa_s1, fa_n1, fa_d}, TypeId::kNumberTypeFloat16);
  AnfNodePtr tmp_tup = parallel::CreateTuple({0, 2, 1, 3});
  (*query_node) = NewTransposeNode((*query_node), tmp_tup);

  (*key_node) = NewReshapeNode((*key_node), {fa_b, fa_s2, fa_h2 / fa_d, fa_d}, TypeId::kNumberTypeFloat16);
  (*key_node) = NewTransposeNode((*key_node), tmp_tup);

  (*value_node) = NewReshapeNode((*value_node), {fa_b, fa_s2, fa_h2 / fa_d, fa_d}, TypeId::kNumberTypeFloat16);
  (*value_node) = NewTransposeNode((*value_node), tmp_tup);
  (*q_shape) = {fa_b, fa_n1, fa_s1, fa_d};
  (*kv_shape) = {fa_b, fa_h2 / fa_d, fa_s2, fa_d};
}

void SplitKVNode(std::vector<AnfNodePtr> *kv_a_nodes, std::vector<AnfNodePtr> *kv_b_nodes, const AnfNodePtr &key_node,
                 const AnfNodePtr &value_node) {
  auto key_split_node = NewSplitNode(key_node, kIndex2, kIndex2);
  auto value_split_node = NewSplitNode(value_node, kIndex2, kIndex2);
  auto key_a_node = NewTupleGetItemNode(key_split_node, kIndex0);
  auto key_b_node = NewTupleGetItemNode(key_split_node, kIndex1);
  auto value_a_node = NewTupleGetItemNode(value_split_node, kIndex0);
  auto value_b_node = NewTupleGetItemNode(value_split_node, kIndex1);
  (*kv_a_nodes) = {key_a_node, value_a_node};
  (*kv_b_nodes) = {key_b_node, value_b_node};
}

void GetCurrentQKVMask(const AnfNodePtr &kv_a_concat, const AnfNodePtr &kv_b_concat, const AnfNodePtr &recv_qkv_tensor,
                       const AnfNodePtr &query_node, const AnfNodePtr &key_node, const AnfNodePtr &value_node,
                       AnfNodePtr *cur_attn_mask, AnfNodePtr *cur_q, AnfNodePtr *cur_k, AnfNodePtr *cur_v,
                       int64_t actual_step, int64_t fa_s1, int64_t fa_s2, const std::vector<AnfNodePtr> &kv_a_nodes,
                       const std::vector<AnfNodePtr> &kv_b_nodes, int64_t pos, int64_t sp_num, int64_t inner_step) {
  if (actual_step == 0) {
    (*cur_attn_mask) = NewValueNode(MakeValue(make_start_mask_tensor(TypeId::kNumberTypeUInt8, {fa_s1, fa_s2})));
    (*cur_q) = query_node;
    (*cur_k) = key_node;
    (*cur_v) = value_node;
    (*cur_q) = CreateDepend((*cur_q), kv_a_concat);
    (*cur_q) = CreateDepend((*cur_q), kv_b_concat);
  } else if (actual_step == sp_num) {
    (*cur_attn_mask) = NewValueNode(MakeValue(make_end_mask_tensor(TypeId::kNumberTypeUInt8, {fa_s1, fa_s2})));
    (*cur_q) = query_node;
    (*cur_k) = key_node;
    (*cur_v) = value_node;
  } else {
    (*cur_attn_mask) =
      NewValueNode(MakeValue(make_mask_tensor(TypeId::kNumberTypeUInt8, {fa_s1, fa_s2 / 2}, 0, false)));
    if (pos < (sp_num / 2)) {
      if ((2 * pos < actual_step) && (actual_step < sp_num)) {  // recv_qkv_tensor is q, compute others oml
        (*cur_q) = recv_qkv_tensor;
        if (inner_step == 0) {
          (*cur_k) = kv_b_nodes[0];
          (*cur_v) = kv_b_nodes[1];
        } else {
          (*cur_k) = kv_a_nodes[0];
          (*cur_v) = kv_a_nodes[1];
        }
      } else {  // recv_qkv_tensor is kv
        (*cur_q) = query_node;
        auto recv_kv_split_node = NewSplitNode(recv_qkv_tensor, kIndex2, kIndex2);
        (*cur_k) = NewTupleGetItemNode(recv_kv_split_node, kIndex0);
        (*cur_v) = NewTupleGetItemNode(recv_kv_split_node, kIndex1);
      }
    } else {  // // recv_qkv_tensor is always kv
      (*cur_q) = query_node;
      auto recv_kv_split_node = NewSplitNode(recv_qkv_tensor, kIndex2, kIndex2);
      (*cur_k) = NewTupleGetItemNode(recv_kv_split_node, kIndex0);
      (*cur_v) = NewTupleGetItemNode(recv_kv_split_node, kIndex1);
    }
  }
}

void SetPrimalAttr(CNodePtr *node, const std::string &flash_index, std::string flash_type,
                   std::int64_t rank_ring_index) {
  if (node == nullptr || (*node) == nullptr) {
    return;
  }
  (*node)->AddPrimalAttr(FLASH_INDEX, MakeValue<std::string>(flash_index));
  (*node)->AddPrimalAttr(FLASH_SP_COMM_TYPE, MakeValue<std::string>(flash_type));
  (*node)->AddPrimalAttr("pos", MakeValue<std::int64_t>(rank_ring_index));
}

void GetCurrentCommNode(int64_t rank_ring_index, CNodePtr *latest_send_qkv, CNodePtr *latest_recv_qkv,
                        CNodePtr *latest_send_oml, CNodePtr *latest_recv_oml, std::string *comm_order_str, int64_t pos,
                        int64_t step, int64_t inner_step, int64_t sp_num, const AnfNodePtr &query_node,
                        const std::vector<CNodePtr> &send_kv_node, RankList spRankList, const AnfNodePtr &cur_q,
                        const AnfNodePtr &send_softmax_max, const AnfNodePtr &send_softmax_sum,
                        const AnfNodePtr &send_attn_out, const AnfNodePtr &tmp_param, Shape q_shape, Shape kv_shape,
                        const std::string &flash_index_str) {
  std::stringstream comm_order;
  if (rank_ring_index % 2 == 0) {  // send first
    auto cur_send_qkv_node = GetCurrentSendQKVNode(pos, step, inner_step, sp_num, query_node, send_kv_node, spRankList,
                                                   g_device_manager->world_group(), cur_q, q_shape, kv_shape);
    auto depend_node = (cur_send_qkv_node == nullptr ? cur_q : cur_send_qkv_node);
    auto cur_recv_qkv_node = GetCurrentRecvQKVNode(pos, step, inner_step, sp_num, spRankList, q_shape, kv_shape,
                                                   g_device_manager->world_group(), depend_node, tmp_param);
    (*latest_send_qkv) = cur_send_qkv_node;
    (*latest_recv_qkv) = cur_recv_qkv_node;

    if (cur_recv_qkv_node != nullptr) {
      depend_node = cur_recv_qkv_node;
    } else if (cur_send_qkv_node != nullptr) {
      depend_node = cur_send_qkv_node;
    } else {
      depend_node = cur_q;
    }
    auto cur_send_oml_node =
      GetCurrentSendOMLNode(pos, step, inner_step, sp_num, spRankList, send_softmax_max, send_softmax_sum,
                            send_attn_out, g_device_manager->world_group(), depend_node, q_shape);
    auto cur_recv_oml_node = GetCurrentRecvOMLNode(pos, step, inner_step, sp_num, spRankList, q_shape,
                                                   g_device_manager->world_group(), depend_node, tmp_param);
    (*latest_send_oml) = cur_send_oml_node;
    (*latest_recv_oml) = cur_recv_oml_node;
    if (cur_send_qkv_node != nullptr) {
      comm_order << "0_";
    }
    if (cur_recv_qkv_node != nullptr) {
      comm_order << "1_";
    }
    if (cur_send_oml_node != nullptr) {
      comm_order << "2";
    }
    if (cur_recv_oml_node != nullptr) {
      comm_order << "3";
    }
  } else {  // recv first
    auto cur_recv_qkv_node = GetCurrentRecvQKVNode(pos, step, inner_step, sp_num, spRankList, q_shape, kv_shape,
                                                   g_device_manager->world_group(), cur_q, tmp_param);
    auto depend_node = (cur_recv_qkv_node == nullptr ? cur_q : cur_recv_qkv_node);
    auto cur_send_qkv_node = GetCurrentSendQKVNode(pos, step, inner_step, sp_num, query_node, send_kv_node, spRankList,
                                                   g_device_manager->world_group(), depend_node, q_shape, kv_shape);
    (*latest_send_qkv) = cur_send_qkv_node;
    (*latest_recv_qkv) = cur_recv_qkv_node;

    if (cur_send_qkv_node != nullptr) {
      depend_node = cur_send_qkv_node;
    } else if (cur_recv_qkv_node != nullptr) {
      depend_node = cur_recv_qkv_node;
    } else {
      depend_node = cur_q;
    }
    auto cur_send_oml_node =
      GetCurrentSendOMLNode(pos, step, inner_step, sp_num, spRankList, send_softmax_max, send_softmax_sum,
                            send_attn_out, g_device_manager->world_group(), depend_node, q_shape);
    auto cur_recv_oml_node = GetCurrentRecvOMLNode(pos, step, inner_step, sp_num, spRankList, q_shape,
                                                   g_device_manager->world_group(), depend_node, tmp_param);
    (*latest_send_oml) = cur_send_oml_node;
    (*latest_recv_oml) = cur_recv_oml_node;
    if (cur_recv_qkv_node != nullptr) {
      comm_order << "1_";
    }
    if (cur_send_qkv_node != nullptr) {
      comm_order << "0_";
    }
    if (cur_send_oml_node != nullptr) {
      comm_order << "2";
    }
    if (cur_recv_oml_node != nullptr) {
      comm_order << "3";
    }
  }
  (*comm_order_str) = comm_order.str();

  SetPrimalAttr(latest_send_qkv, flash_index_str, FLASH_SP_COMM_QKV, rank_ring_index);
  SetPrimalAttr(latest_recv_qkv, flash_index_str, FLASH_SP_COMM_QKV, rank_ring_index);
  SetPrimalAttr(latest_send_oml, flash_index_str, FLASH_SP_COMM_OML, rank_ring_index);
  SetPrimalAttr(latest_recv_oml, flash_index_str, FLASH_SP_COMM_OML, rank_ring_index);
}

void HandleFAResult(int64_t actual_step, CNodePtr *acc_attention, CNodePtr *history_max, CNodePtr *history_sum,
                    CNodePtr *cur_attn_out, CNodePtr *cur_softmax_max, CNodePtr *cur_softmax_sum,
                    CNodePtr *send_softmax_max, CNodePtr *send_softmax_sum, CNodePtr *send_attn_out,
                    CNodePtr *latest_send_oml, CNodePtr *latest_recv_oml, CNodePtr *latest_fa_op, int64_t fa_b,
                    int64_t fa_s1, int64_t fa_n1, int64_t fa_h1, int64_t fa_index, Shape q_shape, int64_t pos,
                    int64_t sp_num, int64_t inner_step) {
  if (actual_step == 0) {  // first step
    *acc_attention = *cur_attn_out;
    *history_max = *cur_softmax_max;
    *history_sum = *cur_softmax_sum;
  } else if (actual_step == sp_num) {  // last step to update self
    if ((*latest_send_oml) != nullptr) {
      *cur_softmax_max = CreateDepend(*cur_softmax_max, *latest_send_oml);
      *cur_softmax_sum = CreateDepend(*cur_softmax_sum, *latest_send_oml);
      *cur_attn_out = CreateDepend(*cur_attn_out, *latest_send_oml);
    }
    UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum, *cur_attn_out,
                          fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step);
    *latest_fa_op = *acc_attention;
    if ((*latest_recv_oml) != nullptr) {
      DismantleRecvOMLTensor(*latest_recv_oml, cur_attn_out, cur_softmax_max, cur_softmax_sum, q_shape);
      UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum, *cur_attn_out,
                            fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step);
      *latest_fa_op = *acc_attention;
    }
  } else {
    if (pos < (sp_num / 2)) {
      if ((2 * pos < actual_step) && (actual_step < sp_num)) {  // compute others oml
        if (inner_step == 0) {
          UpdateAttentionOutput(send_softmax_max, send_softmax_sum, send_attn_out, *cur_softmax_max, *cur_softmax_sum,
                                *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                                actual_step);
          *latest_fa_op = *send_attn_out;
        } else {
          *send_attn_out = *cur_attn_out;
          *send_softmax_max = *cur_softmax_max;
          *send_softmax_sum = *cur_softmax_sum;
        }
      } else {
        UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                              *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                              actual_step);
        *latest_fa_op = *acc_attention;
      }
    } else {
      UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum, *cur_attn_out,
                            fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step);
      *latest_fa_op = *acc_attention;
      if ((*latest_recv_oml) != nullptr) {
        DismantleRecvOMLTensor(*latest_recv_oml, cur_attn_out, cur_softmax_max, cur_softmax_sum, q_shape);
        UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                              *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                              actual_step);
        *latest_fa_op = *acc_attention;
      }
    }
  }
}

CNodePtr CreateReplaceFlashSPGraph(const FuncGraphManagerPtr &manager,
                                   const std::vector<CNodePtr> &origin_nodes_topological, const CNodePtr &fa_score_node,
                                   FSPInfo *fsp_info, int fa_index, const AnfNodePtr &tmp_param_node) {
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }
  int64_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto spRankList = flash_score_info_ptr->GetSPRankList();
  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  auto query_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex + 1);
  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);

  auto input_layout = flash_score_info_ptr->input_layout();

  int64_t fa_n1 = GetValue<int64_t>(
    fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputHeadNumIndex + 1)
      ->cast<ValueNodePtr>()
      ->value());
  int64_t fa_b, fa_s1, fa_h1, fa_s2, fa_h2;
  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_h2);
  int64_t fa_d = fa_h1 / fa_n1;

  if (input_layout == FASInputLayoutMode::BSH) {
    ChangeQKVToBNSD(&query_node, &key_node, &value_node, &q_shape, &kv_shape, fa_b, fa_s1, fa_h1, fa_s2, fa_h2, fa_d,
                    fa_n1);
  }

  std::vector<AnfNodePtr> kv_a_nodes, kv_b_nodes;
  SplitKVNode(&kv_a_nodes, &kv_b_nodes, key_node, value_node);
  auto kv_a_concat = NewConcatNode(NewMakeTupleNode(kv_a_nodes), kIndex2);
  auto kv_b_concat = NewConcatNode(NewMakeTupleNode(kv_b_nodes), kIndex2);
  auto send_kv_node = {kv_a_concat, kv_b_concat};

  CNodePtr attn_out, softmax_max, softmax_sum, send_attn_out, send_softmax_max, send_softmax_sum;
  CNodePtr send_oml_tensor, recv_oml_tensor, send_qkv_tensor, recv_qkv_tensor;
  CNodePtr history_max, history_sum, acc_attention, local_fa_node;
  CNodePtr latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml, latest_fa_op;

  for (int step = 0; step < (sp_num / 2) + 1; ++step) {
    for (int inner_step = 0; inner_step < 2; ++inner_step) {
      auto rank_ring_index = GetRankIndex(pos, step, sp_num);
      auto actual_step = 2 * step + inner_step;
      if (actual_step == sp_num + 1) {
        continue;
      }
      AnfNodePtr cur_attn_mask, cur_q, cur_k, cur_v;
      if (recv_qkv_tensor != nullptr) {
        recv_qkv_tensor = CreateDepends(
          recv_qkv_tensor, {latest_fa_op, latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml});
      }
      GetCurrentQKVMask(kv_a_concat, kv_b_concat, recv_qkv_tensor, query_node, key_node, value_node, &cur_attn_mask,
                        &cur_q, &cur_k, &cur_v, actual_step, fa_s1, fa_s2, kv_a_nodes, kv_b_nodes, pos, sp_num,
                        inner_step);
      cur_q = CreateDepends(
        cur_q, {latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml, latest_fa_op, cur_attn_mask});

      CNodePtr tmp_param;
      if (tmp_param_node != nullptr) {
        tmp_param = NewReluNode(tmp_param_node, cur_q);
        cur_q = CreateDepend(cur_q, tmp_param);
      }

      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex] = cur_q;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = cur_k;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = cur_v;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputLayoutIndex] =
        NewValueNode<int64_t>(FASInputLayoutMode::BNSD);
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = cur_attn_mask;
      local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, actual_step, true);
      common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);
      latest_fa_op = local_fa_node;
      auto cur_softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
      auto cur_softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);
      auto cur_attn_out = NewTupleGetItemNode(local_fa_node, kIndex3);

      std::string comm_order_str;
      std::string ss_result = GetFlashIndexString(fa_index, actual_step);
      GetCurrentCommNode(rank_ring_index, &latest_send_qkv, &latest_recv_qkv, &latest_send_oml, &latest_recv_oml,
                         &comm_order_str, pos, step, inner_step, sp_num, query_node, send_kv_node, spRankList, cur_q,
                         send_softmax_max, send_softmax_sum, send_attn_out, tmp_param, q_shape, kv_shape, ss_result);
      local_fa_node->AddPrimalAttr("comm_order", MakeValue<std::string>(comm_order_str));
      local_fa_node->AddPrimalAttr("sp_num", MakeValue<int64_t>(sp_num));

      if (latest_recv_qkv != nullptr) {
        recv_qkv_tensor = latest_recv_qkv;
      }
      HandleFAResult(actual_step, &acc_attention, &history_max, &history_sum, &cur_attn_out, &cur_softmax_max,
                     &cur_softmax_sum, &send_softmax_max, &send_softmax_sum, &send_attn_out, &latest_send_oml,
                     &latest_recv_oml, &latest_fa_op, fa_b, fa_s1, fa_n1, fa_h1, fa_index, q_shape, pos, sp_num,
                     inner_step);
    }
  }
  acc_attention = CreateDepends(acc_attention, {latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml});
  acc_attention = NewCastNode(acc_attention, TypeId::kNumberTypeFloat16);
  if (input_layout == FASInputLayoutMode::BSH) {
    auto tmp_tup1 = parallel::CreateTuple({0, 2, 1, 3});
    acc_attention = NewTransposeNode(acc_attention, tmp_tup1);
    acc_attention = NewReshapeNode(acc_attention, {fa_b, fa_s1, fa_h1}, TypeId::kNumberTypeFloat16);
  }
  auto softmax_out = NewTupleGetItemNode(local_fa_node, 2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

CNodePtr CreateReplaceRingAttentionGraphByAllToAllv(const FuncGraphManagerPtr &manager,
                                                    const std::vector<CNodePtr> &origin_nodes_topological,
                                                    const CNodePtr &fa_score_node, FSPInfo *fsp_info, int fa_index) {
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }

  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  auto attn_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex + 1);
  auto actual_node =
    fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputActualSeqQlenIndex + 1);

  int64_t actual_shape_size = fsp_info->actual_seq_length_size();
  int64_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  int64_t send_rank_id = fsp_info->GetSendRankId(), recv_rank_id = fsp_info->GetRecvRankId();

  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto input_layout = flash_score_info_ptr->input_layout();
  if (input_layout != FASInputLayoutMode::BSH && input_layout != FASInputLayoutMode::BNSD) {
    return nullptr;
  }

  int64_t fa_n1 = GetValue<int64_t>(
    fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputHeadNumIndex + 1)
      ->cast<ValueNodePtr>()
      ->value());
  int64_t fa_b, fa_s1, fa_h1, fa_s2, fa_h2;
  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_h2);

  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  vector<AnfNodePtr> node_masks;
  for (int i = 0; i < sp_num; ++i) {
    GenerateActualMask(i, pos, sp_num, actual_shape_size, {fa_s1, fa_s2}, actual_node, &node_masks);
  }
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

    auto pos_index = GetUDMaskIndex(i, pos, sp_num);
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = key_node;
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = value_node;
    if (!IsValueNode<None>(attn_node)) {
      auto attn_mask_shape = operator_info->inputs_tensor_info()[kIndex6].tensor_layout().base_slice_shape().array();
      auto attn_mask_split_node = NewSplitNode(attn_node, attn_mask_shape.size() - kIndex1,
                                               sp_num);  // mask has been split in the last dim by sp_num
      actual_mask = NewTupleGetItemNode(attn_mask_split_node, pos_index);
    } else if (IsValueNode<None>(attn_node) && !node_masks.empty()) {  // eod reset attention mask
      actual_mask = node_masks[pos_index];
    } else {
      actual_mask = GetActualMask(i, pos, TypeId::kNumberTypeUInt8, Shape{fa_s1, fa_s2});
    }
    if (actual_mask != nullptr) {
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = actual_mask;
    }
    local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, i, true);
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

CNodePtr CreateReplaceRingAttentionGraphBySendRecv(const FuncGraphManagerPtr &manager,
                                                   const std::vector<CNodePtr> &origin_nodes_topological,
                                                   const CNodePtr &fa_score_node, FSPInfo *fsp_info, int fa_index,
                                                   const AnfNodePtr &tmp_param_node) {
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }

  auto query_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex + 1);
  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);

  int64_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  int64_t send_rank_id = fsp_info->GetSendRankId(), recv_rank_id = fsp_info->GetRecvRankId();

  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto input_layout = flash_score_info_ptr->input_layout();
  if (input_layout != FASInputLayoutMode::BSH && input_layout != FASInputLayoutMode::BNSD) {
    return nullptr;
  }

  int64_t fa_n1 = GetValue<int64_t>(
    fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputHeadNumIndex + 1)
      ->cast<ValueNodePtr>()
      ->value());
  int64_t fa_b, fa_s1, fa_h1, fa_s2, fa_h2;
  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_h2);

  CNodePtr local_fa_node, kv_received_tuple, softmax_max, softmax_sum, softmax_out, attention_output;
  CNodePtr history_max, history_sum, acc_attention, last_fa_node, last_comm_node, send_node, recv_node;
  AnfNodePtr actual_mask;
  for (int i = 0; i < sp_num; ++i) {
    if (i > 0) {
      last_fa_node = CreateDepends(last_fa_node, {send_node, recv_node});
      query_node = CreateDepends(query_node, {last_fa_node, send_node, recv_node});
      recv_node = CreateDepends(recv_node, {last_fa_node, send_node});

      auto kv_split = NewSplitNode(recv_node, kIndex0, kIndex2);
      key_node = NewTupleGetItemNode(kv_split, kIndex0);
      value_node = NewTupleGetItemNode(kv_split, kIndex1);
    }
    std::vector<AnfNodePtr> kv_nodes = {key_node, value_node};
    auto kv_tuple = NewMakeTupleNode(kv_nodes);
    auto kv_concat = NewConcatNode(kv_tuple, 0);
    auto kv_concat_tuple = kv_concat;
    CNodePtr tmp_param;
    if (i != sp_num - 1 && tmp_param_node != nullptr) {
      tmp_param = NewReluNode(tmp_param_node, kv_concat_tuple);
      query_node = CreateDepend(query_node, tmp_param);
    }
    query_node = CreateDepend(query_node, kv_concat_tuple);
    auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
    actual_mask = GetActualMask(i, pos, TypeId::kNumberTypeUInt8, Shape{fa_s1, fa_s2});

    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex] = query_node;
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = key_node;
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = value_node;
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = actual_mask;
    local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, i, false);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);
    last_fa_node = local_fa_node;

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
      last_fa_node = acc_attention;
    }
    if (i != sp_num - 1) {
      auto neigh_shape = kv_shape;
      neigh_shape[0] = neigh_shape[0] * kIndex2;
      if (pos % 2 == 0) {
        kv_concat_tuple = CreateDepend(kv_concat_tuple, query_node);
        send_node = NewSendNode(kv_concat_tuple, 0, send_rank_id, neigh_shape, TypeId::kNumberTypeFloat16,
                                g_device_manager->world_group());
        auto depend_node = (tmp_param == nullptr ? send_node : CreateDepend(tmp_param, send_node));
        recv_node = NewReceiveNode(depend_node, 0, recv_rank_id, neigh_shape, TypeId::kNumberTypeFloat16,
                                   g_device_manager->world_group());
      } else {
        auto depend_node = (tmp_param == nullptr ? query_node : CreateDepend(tmp_param, query_node));
        recv_node = NewReceiveNode(depend_node, 0, recv_rank_id, neigh_shape, TypeId::kNumberTypeFloat16,
                                   g_device_manager->world_group());
        send_node = NewSendNode(CreateDepend(kv_concat_tuple, recv_node), 0, send_rank_id, neigh_shape,
                                TypeId::kNumberTypeFloat16, g_device_manager->world_group());
      }
      send_node->AddPrimalAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(GetFlashIndexString(fa_index, i)));
      recv_node->AddPrimalAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(GetFlashIndexString(fa_index, i)));
      send_node->AddPrimalAttr(RING_ATTENTION_POS, MakeValue<int64_t>(pos));
      recv_node->AddPrimalAttr(RING_ATTENTION_POS, MakeValue<int64_t>(pos));
    }
  }
  acc_attention = CreateDepends(acc_attention, {send_node, recv_node});
  acc_attention = NewCastNode(acc_attention, TypeId::kNumberTypeFloat16);
  softmax_out = NewTupleGetItemNode(local_fa_node, kIndex2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

void CreateAndReplaceRingAttentionFAScore(const FuncGraphManagerPtr &manager,
                                          const std::vector<CNodePtr> &origin_nodes_topological,
                                          const CNodePtr &fa_score_node, FSPInfo *fsp_info, int i, bool use_send_recv,
                                          const AnfNodePtr &tmp_param) {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    MS_LOG(ERROR) << "ring attention & flash sp only supports semi parallel mode";
    return;
  }
  CNodePtr cnode;
  if (use_send_recv == true) {
    cnode = CreateReplaceRingAttentionGraphBySendRecv(manager, origin_nodes_topological, fa_score_node, fsp_info, i,
                                                      tmp_param);
  } else {
    cnode = CreateReplaceRingAttentionGraphByAllToAllv(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
  }
  MS_EXCEPTION_IF_NULL(cnode);
  (void)manager->Replace(fa_score_node, cnode);
}

void CreateAndReplaceFlashSPFAScore(const FuncGraphManagerPtr &manager,
                                    const std::vector<CNodePtr> &origin_nodes_topological,
                                    const CNodePtr &fa_score_node, FSPInfo *fsp_info, int i, bool use_send_recv,
                                    const AnfNodePtr &tmp_param) {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    MS_LOG(ERROR) << "ring attention & flash sp only supports semi parallel mode";
    return;
  }
  auto cnode = CreateReplaceFlashSPGraph(manager, origin_nodes_topological, fa_score_node, fsp_info, i, tmp_param);
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
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &params = func_graph->parameters();
  AnfNodePtr tmp_param;
  for (auto param : params) {
    if (RefParameterToActualParameter(param) != nullptr && !ParameterIsCloned(param)) {
      if (tmp_param == nullptr || param->fullname_with_scope().find("bias") != std::string::npos ||
          param->fullname_with_scope().find("feed_forward") != std::string::npos ||
          param->fullname_with_scope().find("weight") != std::string::npos) {
        tmp_param = param;
      }
    }
  }
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
    if ((!fa_score_node_prim->HasAttr(parallel::ENABLE_RING_ATTENTION) ||
         !GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RING_ATTENTION)))) &&
        (!fa_score_node_prim->HasAttr(parallel::ENABLE_FLASH_SP) ||
         !GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_FLASH_SP))))) {
      continue;
    }
    bool use_send_recv = false;
    if (fa_score_node_prim->HasAttr(parallel::ENABLE_RA_SEND_RECV) &&
        GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RA_SEND_RECV)))) {
      use_send_recv = GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RA_SEND_RECV)));
    }
    auto fsp_info = FSPInfo(fa_score_node);
    if (!CheckUserSettings(func_graph, &fsp_info)) {
      return false;
    }

    manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto orders = func_graph->GetOrderedCnodes();
    std::vector<CNodePtr> nodes_topological(orders.cbegin(), orders.cend());
    if (fa_score_node_prim->HasAttr(parallel::ENABLE_FLASH_SP) &&
        GetValue<bool>(fa_score_node_prim->GetAttr(parallel::ENABLE_FLASH_SP))) {
      CreateAndReplaceFlashSPFAScore(manager, nodes_topological, fa_score_node, &fsp_info, i, use_send_recv, tmp_param);
    } else {
      CreateAndReplaceRingAttentionFAScore(manager, nodes_topological, fa_score_node, &fsp_info, i, use_send_recv,
                                           tmp_param);
    }
  }
  return true;
}
}  // namespace parallel
}  // namespace mindspore
