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

#include <algorithm>
#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/pass/flash_sp.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/anfalgo.h"
#include "include/utils/comm_manager.h"
#include "include/utils/parallel_context.h"
#include "include/utils/utils.h"
#include "infer/make_tuple.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "ir/graph_utils.h"
#include "mindspore/ccsrc/frontend/parallel/graph_util/generate_graph.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/flash_attention_score_info.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
#include "include/op_enum.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/other_ops.h"
#include "primitive/sequence_ops.h"
#include "frontend/jit/ps/action.h"
#include "utils/trace_base.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_z.h"
#include "ir/func_graph_flag.h"

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
    MS_LOG_WITH_NODE(EXCEPTION, fa_score_node) << "ring attention only supports BSH and BNSD layout";
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
constexpr int64_t kAttnMaskSize = 2048;
constexpr int64_t kSplitNum = 2;

struct LayoutInfo {
  int64_t fa_b;
  int64_t fa_s1;
  int64_t fa_s2;
  int64_t fa_h1;
  int64_t fa_h2;
  int64_t fa_n1;
  int64_t input_layout;
  std::vector<int64_t> q_shape;
  std::vector<int64_t> kv_shape;
  TypeId output_id;
};

std::vector<CNodePtr> FindFWFlashAttentionScore(const FuncGraphManagerPtr &manager,
                                                const std::vector<AnfNodePtr> &origin_nodes_topological,
                                                bool *fine_grained_interleave) {
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
    if (!IsPrimitiveCNode(node, prim::kPrimConcat)) {
      continue;
    }
    auto cnode_prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(cnode_prim);
    if (cnode_prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
      *fine_grained_interleave = true;
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

CNodePtr NewDynReshapeNode(const AnfNodePtr &input_node, const AnfNodePtr &output_shape, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                            input_node, output_shape};
  auto reshape = input_node->func_graph()->NewCNode(reshape_inputs);
  MS_EXCEPTION_IF_NULL(reshape);
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
  auto func_graph = input_nodes[0]->func_graph();
  if (func_graph == nullptr) {
    func_graph = input_nodes[1]->func_graph();
  }
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_scope(input_nodes[0]->scope());
  return make_tuple;
}

CNodePtr NewMakeDynTupleNode(const std::vector<AnfNodePtr> &input_nodes) {
  // input_nodes are getitem nodes
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    make_tuple_inputs.push_back(input_nodes[i]);
  }
  auto make_tuple = input_nodes[input_nodes.size() - 1]->func_graph()->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_scope(input_nodes[input_nodes.size() - 1]->scope());
  return make_tuple;
}

CNodePtr NewSplitNode(const AnfNodePtr &split_node, size_t split_dim, size_t split_num) {
  if (split_num == 0) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, split_node) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(split_node);
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                          split_node, NewValueNode<int64_t>(split_dim),
                                          NewValueNode<int64_t>(split_num)};
  auto split = split_node->func_graph()->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);
  split->set_scope(split_node->scope());
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
                                 parallel::Shape neigh_shape, const TypeId &dtype) {
  MS_EXCEPTION_IF_NULL(input_node);
  // input_node is maketuple node
  std::vector<AnfNodePtr> ne_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimNeighborExchange->name())),
                                       input_node};
  auto neighbor_exchange = input_node->func_graph()->NewCNode(ne_inputs);
  MS_EXCEPTION_IF_NULL(neighbor_exchange);

  // RECV_TYPE
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
  auto func_graph = left_node->func_graph();
  if (func_graph == nullptr) {
    func_graph = right_node->func_graph();
  }
  auto add_node = func_graph->NewCNode(add_inputs);
  MS_EXCEPTION_IF_NULL(add_node);
  add_node->set_scope(left_node->scope());
  return add_node;
}

CNodePtr NewSubNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> sub_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSub->name())), left_node,
                                        right_node};
  auto func_graph = left_node->func_graph();
  if (func_graph == nullptr) {
    func_graph = right_node->func_graph();
  }
  auto sub_node = func_graph->NewCNode(sub_inputs);
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

CNodePtr NewScalarMulNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> div_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimScalarMul->name())),
                                        left_node, right_node};
  auto div_node = left_node->func_graph()->NewCNode(div_inputs);
  MS_EXCEPTION_IF_NULL(div_node);
  div_node->set_scope(left_node->scope());
  return div_node;
}

CNodePtr NewScalarSubNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> div_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimScalarSub->name())),
                                        left_node, right_node};
  auto div_node = right_node->func_graph()->NewCNode(div_inputs);
  MS_EXCEPTION_IF_NULL(div_node);
  div_node->set_scope(left_node->scope());
  return div_node;
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

CNodePtr NewScalarDivNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> div_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimScalarFloorDiv->name())),
                                        left_node, right_node};
  auto div_node = left_node->func_graph()->NewCNode(div_inputs);
  MS_EXCEPTION_IF_NULL(div_node);
  div_node->set_scope(left_node->scope());
  return div_node;
}

CNodePtr NewScalarAddNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  std::vector<AnfNodePtr> div_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimScalarAdd->name())),
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

CNodePtr NewDynshapeNode(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> dynshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimShape->name())),
                                             input_node};
  auto dynshape_node = input_node->func_graph()->NewCNode(dynshape_inputs);
  MS_EXCEPTION_IF_NULL(dynshape_node);
  return dynshape_node;
}

CNodePtr NewOnesNode(const AnfNodePtr &input_node, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> dynshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimOnes->name())),
                                             input_node, NewValueNode(static_cast<int64_t>(output_type))};
  auto ones_node = input_node->func_graph()->NewCNode(dynshape_inputs);
  MS_EXCEPTION_IF_NULL(ones_node);
  return ones_node;
}

CNodePtr NewZerosNode(const AnfNodePtr &input_node, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> dynshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimZeros->name())),
                                             input_node, NewValueNode(static_cast<int64_t>(output_type))};
  auto ones_node = input_node->func_graph()->NewCNode(dynshape_inputs);
  MS_EXCEPTION_IF_NULL(ones_node);
  return ones_node;
}

CNodePtr NewOnesNode1(const AnfNodePtr &input_node, const AnfNodePtr &shape, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> dynshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimOnes->name())), shape,
                                             NewValueNode(static_cast<int64_t>(output_type))};
  auto ones_node = input_node->func_graph()->NewCNode(dynshape_inputs);
  MS_EXCEPTION_IF_NULL(ones_node);
  return ones_node;
}

CNodePtr NewZerosNode1(const AnfNodePtr &input_node, const AnfNodePtr &shape, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> dynshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimZeros->name())), shape,
                                             NewValueNode(static_cast<int64_t>(output_type))};
  auto ones_node = input_node->func_graph()->NewCNode(dynshape_inputs);
  MS_EXCEPTION_IF_NULL(ones_node);
  return ones_node;
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

tensor::TensorPtr make_mask_tensor(TypeId type_id, ShapeVector shape, uint8_t value, bool is_causle) {
  tensor::TensorPtr mask_tensor = tensor::from_spec(type_id, shape, device::DeviceType::kCPU);
  int64_t tensor_size = SizeToLong(mask_tensor->DataSize());
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

AnfNodePtr dyn_make_mask_tensor(const AnfNodePtr &fa_s1, const AnfNodePtr &fa_s2, int value, bool is_causle) {
  CNodePtr fa_tuple_node = NewMakeTupleNode({fa_s1, fa_s2});
  CNodePtr mask_node;
  if (!is_causle) {
    if (value) {
      mask_node = NewOnesNode(fa_tuple_node, TypeId::kNumberTypeUInt8);
    } else {
      mask_node = NewZerosNode(fa_tuple_node, TypeId::kNumberTypeUInt8);
    }
  } else {
    mask_node = NewTriuNode(NewOnesNode(fa_tuple_node, TypeId::kNumberTypeUInt8), NewValueNode<int64_t>(1));
  }
  return mask_node;
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

CNodePtr NewDynRepeatNode(const AnfNodePtr &range, const AnfNodePtr &repeat_nums, const AnfNodePtr &output_size) {
  MS_EXCEPTION_IF_NULL(range);
  MS_EXCEPTION_IF_NULL(repeat_nums);
  std::vector<AnfNodePtr> repeat_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimRepeatInterleaveTensor->name())), range, repeat_nums,
    NewValueNode<int64_t>(0), output_size};
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

CNodePtr NewGatherNode(const AnfNodePtr &input_node, int64_t actual_shape) {
  MS_EXCEPTION_IF_NULL(input_node);
  tensor::TensorPtr const_tensor = tensor::from_spec(TypeId::kNumberTypeInt64, Shape{1}, device::DeviceType::kCPU);
  int64_t *int_data = reinterpret_cast<int64_t *>(const_tensor->data_c());
  int_data[0] = actual_shape - 1;
  std::vector<AnfNodePtr> gather_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimGather->name())),
                                           input_node, NewValueNode(MakeValue(const_tensor)), NewValueNode<int64_t>(0),
                                           NewValueNode<int64_t>(0)};
  auto node_gather = input_node->func_graph()->NewCNode(gather_inputs);
  MS_EXCEPTION_IF_NULL(node_gather);
  return node_gather;
}

AnfNodePtr NewTupletoTensorNode(const AnfNodePtr &input_node, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> tensorlist_node = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleToTensor->name())), input_node,
    NewValueNode(static_cast<int64_t>(output_type))};
  MS_EXCEPTION_IF_NULL(input_node->func_graph());
  auto tensor_node = input_node->func_graph()->NewCNode(tensorlist_node);
  MS_EXCEPTION_IF_NULL(tensor_node);
  return tensor_node;
}

AnfNodePtr NewTensortoTupleNode(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> tensorlist_node = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimTensorToTuple->name())), input_node};
  MS_EXCEPTION_IF_NULL(input_node->func_graph());
  auto tensor_node = input_node->func_graph()->NewCNode(tensorlist_node);
  MS_EXCEPTION_IF_NULL(tensor_node);
  return tensor_node;
}

AnfNodePtr NewScalartoTensorNode(const AnfNodePtr &input_node, const TypeId &output_type) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> tensorlist_node = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimScalarToTensor->name())), input_node,
    NewValueNode(static_cast<int64_t>(output_type))};
  MS_EXCEPTION_IF_NULL(input_node->func_graph());
  auto tensor_node = input_node->func_graph()->NewCNode(tensorlist_node);
  MS_EXCEPTION_IF_NULL(tensor_node);
  return tensor_node;
}

void GenerateEodMask(int index, int64_t rank_id, int64_t sp_num, int64_t actual_shape, const ShapeVector &s_shape,
                     const AnfNodePtr &actual_node, std::vector<AnfNodePtr> *node_masks) {
  AnfNodePtr actual_input = nullptr;
  auto actual_cnode = actual_node->cast<CNodePtr>();
  if (actual_cnode != nullptr && actual_cnode->inputs().size() > 1) {
    actual_input = actual_cnode->input(1);  // fa actual seq input is ValueToTuple, so get it first input
  }
  if (actual_input == nullptr || actual_shape == 0) {
    MS_LOG(INFO) << "Input of actual_seq_length is required when enable eod "
                    "reset attention mask.";
    return;
  }

  auto last_eod = NewGatherNode(NewTupletoTensorNode(actual_node, TypeId::kNumberTypeInt64), actual_shape);

  auto node_range = NewRangeNode(actual_input->func_graph(), actual_shape);

  auto node_roll = NewRollNode(actual_input);

  auto node_sub = NewSubNode(actual_input, node_roll);

  tensor::TensorPtr const_tensor =
    tensor::from_spec(TypeId::kNumberTypeInt64, Shape{actual_shape - 1}, device::DeviceType::kCPU);
  int64_t *int_data = reinterpret_cast<int64_t *>(const_tensor->data_c());

  for (int i = 0; i < actual_shape - 1; ++i) {
    int_data[i] = 0;
  }
  auto zeros_node = NewValueNode(MakeValue(const_tensor));

  auto const_seq = NewConcatNode(NewMakeTupleNode({last_eod, zeros_node}), 0);

  auto node_add = NewAddNode(node_sub, const_seq);

  auto node_repeat = NewDynRepeatNode(node_range, node_add, NewTupleGetItemNode(NewTensortoTupleNode(last_eod), 0));

  tensor::TensorPtr z_tensor = tensor::from_spec(TypeId::kNumberTypeInt64, Shape{1}, device::DeviceType::kCPU);
  int64_t *z_data = reinterpret_cast<int64_t *>(z_tensor->data_c());
  z_data[0] = sp_num * s_shape[0];

  auto node_repeat_shape = NewDynshapeNode(node_repeat);
  auto node_repeat_shape0 = NewTupleGetItemNode(node_repeat_shape, kIndex0);
  auto ones = NewOnesNode1(
    node_repeat, NewMakeTupleNode({NewScalarSubNode(NewValueNode<int64_t>(sp_num * s_shape[0]), node_repeat_shape0)}),
    TypeId::kNumberTypeInt64);

  auto zeros = NewZerosNode1(
    node_repeat, NewMakeTupleNode({NewScalarSubNode(NewValueNode<int64_t>(sp_num * s_shape[0]), node_repeat_shape0)}),
    TypeId::kNumberTypeInt64);

  auto node_repeat_w = NewConcatNode(NewMakeTupleNode({node_repeat, ones}), 0);

  auto node_repeat_h = NewConcatNode(NewMakeTupleNode({node_repeat, zeros}), 0);

  auto node_split_w = NewSplitNode(node_repeat_w, 0, sp_num);
  auto node_split_h = NewSplitNode(node_repeat_h, 0, sp_num);

  auto node_get_tuple_w = NewTupleGetItemNode(node_split_w, index);

  auto node_get_tuple_h = NewTupleGetItemNode(node_split_h, rank_id);

  auto node_w_tile = NewTileNode(node_get_tuple_w, parallel::CreateTuple({s_shape[0], 1}));

  auto node_reshape = NewReshapeNode(node_get_tuple_h, {s_shape[0], 1});

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

void DynGenerateEodMask(int index, int64_t rank_id, int64_t sp_num, int64_t actual_shape, const AnfNodePtr &fa_s1,
                        const AnfNodePtr &fa_s2, const AnfNodePtr &actual_node, std::vector<AnfNodePtr> *node_masks) {
  AnfNodePtr actual_input = nullptr;
  auto actual_cnode = actual_node->cast<CNodePtr>();
  if (actual_cnode != nullptr && actual_cnode->inputs().size() > 1) {
    actual_input = actual_cnode->input(1);  // fa actual seq input is ValueToTuple, so get it first input
  }
  if (actual_input == nullptr || actual_shape == 0) {
    MS_LOG(INFO) << "Input of actual_seq_length is required when enable eod "
                    "reset attention mask.";
    return;
  }
  auto node_range = NewRangeNode(actual_input->func_graph(), actual_shape);

  auto node_roll = NewRollNode(actual_input);

  auto node_sub = NewSubNode(actual_input, node_roll);

  tensor::TensorPtr const_tensor =
    tensor::from_spec(TypeId::kNumberTypeInt64, Shape{actual_shape - 1}, device::DeviceType::kCPU);
  int64_t *int_data = reinterpret_cast<int64_t *>(const_tensor->data_c());
  for (int i = 0; i < actual_shape - 1; ++i) {
    int_data[i] = 0;
  }
  auto zeros_node = NewValueNode(MakeValue(const_tensor));
  auto const_seq = NewConcatNode(
    NewMakeTupleNode(
      {NewReshapeNode(
         NewScalartoTensorNode(NewScalarMulNode(fa_s1, NewValueNode<int64_t>(sp_num)), TypeId::kNumberTypeInt64), {1}),
       zeros_node}),
    0);
  auto node_add = NewAddNode(node_sub, const_seq);

  auto node_repeat = NewDynRepeatNode(node_range, node_add, NewScalarMulNode(fa_s1, NewValueNode<int64_t>(sp_num)));

  auto node_split = NewSplitNode(node_repeat, 0, sp_num);

  auto node_get_tuple_w = NewTupleGetItemNode(node_split, index);

  auto node_get_tuple_h = NewTupleGetItemNode(node_split, rank_id);

  auto node_w_tile = NewTileNode(node_get_tuple_w, NewMakeTupleNode({fa_s1, NewValueNode<int64_t>(1)}));
  auto node_reshape =
    NewDynReshapeNode(node_get_tuple_h, NewMakeTupleNode({fa_s1, NewValueNode<int64_t>(1)}), TypeId::kNumberTypeInt64);

  auto node_h_tile = NewTileNode(node_reshape, NewMakeDynTupleNode({NewValueNode<int64_t>(1), fa_s2}));

  auto node_equal = NewEqualNode(node_h_tile, node_w_tile);
  std::vector<AnfNodePtr> logical_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimLogicalNot->name()))};

  if (rank_id == index) {
    auto node_tran_triu = NewTransposeNode(node_equal, parallel::CreateTuple({1, 0}));
    auto node_tril = NewTriuNode(node_tran_triu, NewValueNode<int64_t>(0));
    auto node_tran_tril = NewTransposeNode(node_tril, parallel::CreateTuple({1, 0}));
    logical_inputs.emplace_back(node_tran_tril);
  } else if (rank_id < index) {
    auto value_node0 = dyn_make_mask_tensor(fa_s1, fa_s2, 0, false);
    logical_inputs.emplace_back(value_node0);
  } else {
    logical_inputs.emplace_back(node_equal);
  }

  auto node_logical = node_equal->func_graph()->NewCNode(logical_inputs);

  node_masks->emplace_back(node_logical);
}

tensor::TensorPtr make_start_mask_tensor(TypeId type_id, ShapeVector shape) {
  tensor::TensorPtr mask_tensor = tensor::from_spec(type_id, shape, device::DeviceType::kCPU);
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

AnfNodePtr dyn_make_start_mask_tensor(const AnfNodePtr &fa_s1, const AnfNodePtr &fa_s2) {
  CNodePtr fa_tuple_node =
    NewMakeTupleNode({NewScalarDivNode(fa_s1, NewValueNode(2)), NewScalarDivNode(fa_s2, NewValueNode(2))});
  CNodePtr ones_node = NewOnesNode(fa_tuple_node, TypeId::kNumberTypeUInt8);
  CNodePtr triu_node = NewTriuNode(ones_node, NewValueNode<int64_t>(1));
  CNodePtr concat_node_0 = NewConcatNode(NewMakeTupleNode({triu_node, ones_node}), 1);
  CNodePtr concat_node_1 = NewConcatNode(NewMakeTupleNode({ones_node, triu_node}), 1);
  CNodePtr concat_node = NewConcatNode(NewMakeTupleNode({concat_node_0, concat_node_1}), 0);
  return concat_node;
}

tensor::TensorPtr make_end_mask_tensor(TypeId type_id, ShapeVector shape) {
  tensor::TensorPtr mask_tensor = tensor::from_spec(type_id, shape, device::DeviceType::kCPU);
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

AnfNodePtr dyn_make_end_mask_tensor(const AnfNodePtr &fa_s1, const AnfNodePtr &fa_s2) {
  CNodePtr fa_tuple_node =
    NewMakeTupleNode({NewScalarDivNode(fa_s1, NewValueNode(2)), NewScalarDivNode(fa_s2, NewValueNode(2))});
  CNodePtr ones_node_0 = NewOnesNode(fa_tuple_node, TypeId::kNumberTypeUInt8);
  CNodePtr ones_node_1 = NewOnesNode(fa_tuple_node, TypeId::kNumberTypeUInt8);
  CNodePtr ones_node_2 = NewOnesNode(fa_tuple_node, TypeId::kNumberTypeUInt8);
  CNodePtr concat_node_1 = NewConcatNode(NewMakeTupleNode({ones_node_1, ones_node_2}), 1);
  CNodePtr zeros_node = NewZerosNode(fa_tuple_node, TypeId::kNumberTypeUInt8);
  CNodePtr concat_node_0 = NewConcatNode(NewMakeTupleNode({zeros_node, ones_node_0}), 1);
  CNodePtr concat_node = NewConcatNode(NewMakeTupleNode({concat_node_1, concat_node_0}), 0);
  return concat_node;
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

AnfNodePtr DynGetActualMask(int index, int64_t rank_id, const AnfNodePtr &fa_s1_dyn, const AnfNodePtr &fa_s2_dyn) {
  AnfNodePtr actual_mask;
  if (index == 0) {
    actual_mask = dyn_make_mask_tensor(fa_s1_dyn, fa_s2_dyn, 0, true);
  } else if (index <= rank_id) {
    actual_mask = dyn_make_mask_tensor(fa_s1_dyn, fa_s2_dyn, 0, false);
  } else {
    actual_mask = dyn_make_mask_tensor(fa_s1_dyn, fa_s2_dyn, 1, false);
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
      pos = SizeToLong(rank_list_idx);
    }
  }
  return pos;
}

size_t GetRankIndex(int64_t rank_id, size_t step, size_t sp_size) {
  std::vector<int> rank_order;
  for (size_t i = 0; i < sp_size; i++) {
    if (i % (step + 1) == 0) {
      rank_order.push_back(i);
    }
  }
  for (size_t i = 1; i <= step; i++) {
    for (size_t j = i; j < sp_size; j += step + 1) {
      rank_order.push_back(j);
    }
  }
  size_t pos = -1;
  for (size_t rank_list_idx = 0; rank_list_idx < rank_order.size(); ++rank_list_idx) {
    if (rank_id == rank_order[rank_list_idx]) {
      pos = rank_list_idx;
    }
  }
  return pos;
}

void GetBSHFromShape(int64_t input_layout, Shape q_shape, Shape kv_shape, int64_t *fa_b, int64_t *fa_s1, int64_t *fa_h1,
                     int64_t *fa_s2, int64_t *fa_h2, int64_t *fa_n1, const CNodePtr &fa_score_node) {
  auto fa_input = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputHeadNumIndex + 1);
  MS_EXCEPTION_IF_NULL(fa_input);
  auto fa_input_cnode = fa_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(fa_input_cnode);
  *fa_n1 = GetValue<int64_t>(fa_input_cnode->value());
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

void GetBSHFromDynShape(int64_t input_layout, CNodePtr *fa_b_dyn, CNodePtr *fa_s1_dyn, CNodePtr *fa_h1_dyn,
                        CNodePtr *fa_s2_dyn, CNodePtr *fa_h2_dyn, int64_t *fa_n1, const CNodePtr &q_dynshape_node,
                        const CNodePtr &kv_dynshape_node, const CNodePtr &fa_score_node) {
  auto fa_input = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputHeadNumIndex + 1);
  MS_EXCEPTION_IF_NULL(fa_input);
  auto fa_input_cnode = fa_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(fa_input_cnode);
  *fa_n1 = GetValue<int64_t>(fa_input_cnode->value());
  if (input_layout == FASInputLayoutMode::BSH) {
    *fa_b_dyn = NewTupleGetItemNode(q_dynshape_node, kIndex0);
    *fa_s1_dyn = NewTupleGetItemNode(q_dynshape_node, kIndex1);
    *fa_h1_dyn = NewTupleGetItemNode(q_dynshape_node, kIndex2);
    *fa_s2_dyn = NewTupleGetItemNode(kv_dynshape_node, kIndex1);
    *fa_h2_dyn = NewTupleGetItemNode(kv_dynshape_node, kIndex2);
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    *fa_b_dyn = NewTupleGetItemNode(q_dynshape_node, kIndex0);
    *fa_s1_dyn = NewTupleGetItemNode(q_dynshape_node, kIndex2);
    *fa_h1_dyn =
      NewScalarMulNode(NewTupleGetItemNode(q_dynshape_node, kIndex1), NewTupleGetItemNode(q_dynshape_node, kIndex3));
    *fa_s2_dyn = NewTupleGetItemNode(kv_dynshape_node, kIndex2);
    *fa_h2_dyn =
      NewScalarMulNode(NewTupleGetItemNode(kv_dynshape_node, kIndex1), NewTupleGetItemNode(kv_dynshape_node, kIndex3));
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
  MS_EXCEPTION_IF_NULL(latter_cnode);
  for (size_t i = 0; i < former_nodes.size(); ++i) {
    auto former_node = former_nodes[i];
    if (former_node == nullptr) {
      continue;
    }
    auto func_graph = latter_cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          latter_cnode, former_node};
    auto depend = func_graph->NewCNode(depend_inputs);

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

CNodePtr NewDynStridedSliceNode(const AnfNodePtr &tensor_node, const std::vector<AnfNodePtr> &begin,
                                const std::vector<AnfNodePtr> &end, const Shape &strides) {
  MS_EXCEPTION_IF_NULL(tensor_node);
  std::vector<AnfNodePtr> stridedslice_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimStridedSlice->name())),
    tensor_node,
    NewMakeDynTupleNode(begin),
    NewMakeDynTupleNode(end),
    NewValueNode(MakeValue(strides)),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0)))};
  MS_EXCEPTION_IF_NULL(tensor_node->func_graph());
  auto stridedslice_node = tensor_node->func_graph()->NewCNode(stridedslice_inputs);
  MS_EXCEPTION_IF_NULL(stridedslice_node);
  stridedslice_node->set_scope(tensor_node->scope());
  return stridedslice_node;
}

CNodePtr NewDynStridedSliceNode1(const AnfNodePtr &tensor_node, const Shape &begin, const std::vector<AnfNodePtr> &end,
                                 const Shape &strides) {
  MS_EXCEPTION_IF_NULL(tensor_node);
  std::vector<AnfNodePtr> stridedslice_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimStridedSlice->name())),
    tensor_node,
    NewValueNode(MakeValue(begin)),
    NewMakeDynTupleNode(end),
    NewValueNode(MakeValue(strides)),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0))),
    NewValueNode(MakeValue(int64_t(0)))};
  MS_EXCEPTION_IF_NULL(tensor_node->func_graph());
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
                           const CNodePtr &softmax_max, const CNodePtr &softmax_sum, const CNodePtr &attention_output,
                           int64_t fa_b, int64_t fa_s1, int64_t fa_n1, int64_t fa_h1, int64_t input_layout,
                           int fa_index, int index, TypeId output_type_id, bool is_last_update = false,
                           bool need_split = false) {
  CNodePtr split_max_0;
  CNodePtr split_sum_0;
  CNodePtr split_attn_0;
  if (need_split) {
    auto split_max = NewSplitNode(*history_max, kDim2, kIndex2);
    *history_max = NewTupleGetItemNode(split_max, kIndex1);
    auto split_sum = NewSplitNode(*history_sum, kDim2, kIndex2);
    *history_sum = NewTupleGetItemNode(split_sum, kIndex1);
    auto axis = input_layout == FASInputLayoutMode::BSH ? kDim1 : kDim2;
    auto split_attn = NewSplitNode(*acc_attention, axis, kIndex2);
    *acc_attention = NewTupleGetItemNode(split_attn, kIndex1);

    split_max_0 = NewTupleGetItemNode(split_max, kIndex0);
    split_sum_0 = NewTupleGetItemNode(split_sum, kIndex0);
    split_attn_0 = NewTupleGetItemNode(split_attn, kIndex0);
  }
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
  auto e_m_h_div_item = NewCastNode(NewTupleGetItemNode(e_m_h_div_split, 0), output_type_id);
  if (input_layout == FASInputLayoutMode::BSH) {
    e_m_h_div_item = NewTransposeNode(e_m_h_div_item, parallel::CreateTuple({0, 2, 1, 3}));
  }
  auto e_m_h_div_concat = NewTileNode(e_m_h_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  if (input_layout == FASInputLayoutMode::BSH) {
    auto real_seq = need_split ? fa_s1 / kSplitNum : fa_s1;
    e_m_h_div_concat = NewReshapeNode(e_m_h_div_concat, {fa_b, real_seq, fa_h1});
  }

  auto e_m_i_div_split = NewSplitNode(e_m_i_div, 3, 8);
  auto e_m_i_div_item = NewCastNode(NewTupleGetItemNode(e_m_i_div_split, 0), output_type_id);
  if (input_layout == FASInputLayoutMode::BSH) {
    e_m_i_div_item = NewTransposeNode(e_m_i_div_item, parallel::CreateTuple({0, 2, 1, 3}));
  }
  auto e_m_i_div_concat = NewTileNode(e_m_i_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  if (input_layout == FASInputLayoutMode::BSH) {
    auto real_seq = need_split ? fa_s1 / kSplitNum : fa_s1;
    e_m_i_div_concat = NewReshapeNode(e_m_i_div_concat, {fa_b, real_seq, fa_h1});
  }
  auto weighted_history = NewMulNode(e_m_h_div_concat, *acc_attention);
  auto weighted_attention = NewMulNode(e_m_i_div_concat, attention_output);
  (*acc_attention) = NewAddNode(weighted_history, weighted_attention);
  if (need_split) {
    auto merge_max = NewMakeTupleNode({split_max_0, temp_max});
    temp_max = NewConcatNode(merge_max, kIndex2);

    auto merge_sum = NewMakeTupleNode({split_sum_0, l});
    l = NewConcatNode(merge_sum, kIndex2);

    auto merge_attn = NewMakeTupleNode({split_attn_0, *acc_attention});
    auto axis = input_layout == FASInputLayoutMode::BSH ? kDim1 : kDim2;
    *acc_attention = NewConcatNode(merge_attn, axis);
  }
  (*history_max) = temp_max;
  *acc_attention = CreateDepend(*acc_attention, *history_max);
  (*history_sum) = l;
  *acc_attention = CreateDepend(*acc_attention, *history_sum);
  common::AnfAlgo::SetNodeAttr(kAttrAccumulatedAttention, MakeValue(1), *acc_attention);
  common::AnfAlgo::SetNodeAttr("FLASH_INDEX", MakeValue<std::string>(GetFlashIndexString(fa_index, index)),
                               *acc_attention);
  if (is_last_update) {
    weighted_attention->AddPrimalAttr(RING_ATTENTION_UPDATE_MUL, MakeValue<int>(fa_index));
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_UPDATE_MAX, MakeValue<int>(fa_index), *history_max);
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_UPDATE_SUM, MakeValue<int>(fa_index), *history_sum);
  }
}

void DynUpdateAttentionOutput(CNodePtr *history_max, CNodePtr *history_sum, CNodePtr *acc_attention,
                              const CNodePtr &softmax_max, const CNodePtr &softmax_sum, CNodePtr attention_output,
                              const CNodePtr &fa_b, const CNodePtr &fa_s1, int64_t fa_n1, const CNodePtr &fa_h1,
                              int64_t input_layout, int fa_index, int index) {
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
  auto e_m_h_div_concat = NewTileNode(
    e_m_h_div_item, NewMakeDynTupleNode({NewValueNode<int64_t>(1), NewValueNode<int64_t>(1), NewValueNode<int64_t>(1),
                                         NewScalarDivNode(fa_h1, NewValueNode<int64_t>(fa_n1))}));
  auto e_m_i_div_split = NewSplitNode(e_m_i_div, 3, 8);
  auto e_m_i_div_item = NewTupleGetItemNode(e_m_i_div_split, 0);
  auto e_m_i_div_concat = NewTileNode(
    e_m_i_div_item, NewMakeDynTupleNode({NewValueNode<int64_t>(1), NewValueNode<int64_t>(1), NewValueNode<int64_t>(1),
                                         NewScalarDivNode(fa_h1, NewValueNode<int64_t>(fa_n1))}));
  if (input_layout == FASInputLayoutMode::BSH) {
    (*acc_attention) = NewDynReshapeNode(*acc_attention,
                                         NewMakeTupleNode({fa_b, fa_s1, NewValueNode<int64_t>(fa_n1),
                                                           NewScalarDivNode(fa_h1, NewValueNode<int64_t>(fa_n1))}),
                                         TypeId::kNumberTypeFloat16);
    attention_output = NewDynReshapeNode(attention_output,
                                         NewMakeTupleNode({fa_b, fa_s1, NewValueNode<int64_t>(fa_n1),
                                                           NewScalarDivNode(fa_h1, NewValueNode<int64_t>(fa_n1))}),
                                         TypeId::kNumberTypeFloat16);
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
    (*acc_attention) =
      NewDynReshapeNode(*acc_attention, NewMakeTupleNode({fa_b, fa_s1, fa_h1}), TypeId::kNumberTypeFloat32);
  }
  (*history_max) = temp_max;
  (*history_sum) = l;
  (*acc_attention) = NewCastNode(*acc_attention, TypeId::kNumberTypeFloat16);
}

CNodePtr ConstructSendOMLTensor(const AnfNodePtr &send_softmax_max, const AnfNodePtr &send_softmax_sum,
                                const AnfNodePtr &send_attn_out, TypeId output_type_id) {
  auto send_softmax_max_cast = NewCastNode(send_softmax_max, output_type_id);
  auto send_softmax_sum_cast = NewCastNode(send_softmax_sum, output_type_id);
  std::vector<AnfNodePtr> oml_nodes = {send_attn_out, send_softmax_max_cast, send_softmax_sum_cast};
  auto oml_tuple = NewMakeTupleNode(oml_nodes);
  return NewConcatNode(oml_tuple, kIndex3);
}

void DismantleRecvOMLTensor(const AnfNodePtr &recv_oml_tensor, CNodePtr *cur_attn_out, CNodePtr *cur_softmax_max,
                            CNodePtr *cur_softmax_sum, Shape q_shape) {
  (*cur_attn_out) =
    NewStridedSliceNode(recv_oml_tensor, {0, 0, 0, 0}, {q_shape[0], q_shape[1], q_shape[2], q_shape[3]}, {1, 1, 1, 1});
  (*cur_softmax_max) =
    NewCastNode(NewStridedSliceNode(recv_oml_tensor, {0, 0, 0, q_shape[3]},
                                    {q_shape[0], q_shape[1], q_shape[2], q_shape[3] + 8}, {1, 1, 1, 1}),
                TypeId::kNumberTypeFloat32);
  (*cur_softmax_sum) =
    NewCastNode(NewStridedSliceNode(recv_oml_tensor, {0, 0, 0, q_shape[3] + 8},
                                    {q_shape[0], q_shape[1], q_shape[2], q_shape[3] + 16}, {1, 1, 1, 1}),
                TypeId::kNumberTypeFloat32);
}

void DynDismantleRecvOMLTensor(const AnfNodePtr &recv_oml_tensor, CNodePtr *cur_attn_out, CNodePtr *cur_softmax_max,
                               CNodePtr *cur_softmax_sum, const AnfNodePtr &q_shape) {
  (*cur_attn_out) = NewDynStridedSliceNode1(recv_oml_tensor, {0, 0, 0, 0},
                                            {NewTupleGetItemNode(q_shape, 0), NewTupleGetItemNode(q_shape, 1),
                                             NewTupleGetItemNode(q_shape, 2), NewTupleGetItemNode(q_shape, 3)},
                                            {1, 1, 1, 1});
  (*cur_attn_out) = NewCastNode(*cur_attn_out, TypeId::kNumberTypeFloat32);
  (*cur_softmax_max) = NewDynStridedSliceNode(
    recv_oml_tensor,
    {NewValueNode<int64_t>(0), NewValueNode<int64_t>(0), NewValueNode<int64_t>(0), NewTupleGetItemNode(q_shape, 3)},
    {NewTupleGetItemNode(q_shape, 0), NewTupleGetItemNode(q_shape, 1), NewTupleGetItemNode(q_shape, 2),
     NewScalarAddNode(NewTupleGetItemNode(q_shape, 3), NewValueNode<int64_t>(8))},
    {1, 1, 1, 1});
  (*cur_softmax_max) = NewCastNode(*cur_softmax_max, TypeId::kNumberTypeFloat32);
  (*cur_softmax_sum) = NewDynStridedSliceNode(
    recv_oml_tensor,
    {NewValueNode<int64_t>(0), NewValueNode<int64_t>(0), NewValueNode<int64_t>(0),
     NewScalarAddNode(NewTupleGetItemNode(q_shape, 3), NewValueNode<int64_t>(8))},
    {NewTupleGetItemNode(q_shape, 0), NewTupleGetItemNode(q_shape, 1), NewTupleGetItemNode(q_shape, 2),
     NewScalarAddNode(NewTupleGetItemNode(q_shape, 3), NewValueNode<int64_t>(16))},
    {1, 1, 1, 1});
  (*cur_softmax_sum) = NewCastNode(*cur_softmax_sum, TypeId::kNumberTypeFloat32);
}

int64_t GetSendQKVDstRank(size_t rank, size_t step, size_t sp_size) { return (rank + step + 1) % sp_size; }

int64_t GetRecvQKVSrcRank(size_t rank, size_t step, size_t sp_size) { return (rank + sp_size - step - 1) % sp_size; }

int64_t GetSendOMLDstRank(size_t rank, size_t step, size_t sp_size) { return (rank + sp_size - step) % sp_size; }

int64_t GetRecvOMLSrcRank(size_t rank, size_t step, size_t sp_size) { return (rank + step) % sp_size; }

enum class TagType {
  query = 0,
  kv_a = 1,
  kv_b = 2,
  oml = 3,
};

int64_t GetSendRecvTag(int64_t src, int64_t dest, TagType data_type) {
  auto src_string = std::to_string(src + 1);
  auto dest_string = std::to_string(dest + 1);
  auto data_type_string = std::to_string(static_cast<int>(data_type));
  auto res_string = src_string + dest_string + data_type_string;
  return std::stoi(res_string);
}

CNodePtr GetCurrentSendQKVNode(size_t pos, size_t step, size_t inner_step, size_t sp_num, const AnfNodePtr &query_node,
                               const std::vector<CNodePtr> &send_kv_node, RankList spRankList,
                               const std::string &qkv_group, const AnfNodePtr &pre_node, Shape q_shape, Shape kv_shape,
                               TypeId output_type_id) {
  if (step < (sp_num / kIndex2)) {  // [0, sp-1]
    auto send_qkv_dst_rank = GetSendQKVDstRank(pos, step, sp_num);
    if (pos + step + 1 >= sp_num) {  // send q
      if (inner_step == 0) {
        return NewSendNode(CreateDepend(query_node, pre_node), GetSendRecvTag(pos, send_qkv_dst_rank, TagType::query),
                           spRankList[send_qkv_dst_rank], q_shape, output_type_id, qkv_group);
      }
    } else {  // send kv
      auto data_type = (inner_step == 0 ? TagType::kv_b : TagType::kv_a);
      auto tmp_node = send_kv_node[(inner_step + 1) % 2];
      if (step < ((sp_num / kIndex2) - kIndex1)) {  // [0, sp-3]
        return NewSendNode(CreateDepend(tmp_node, pre_node), GetSendRecvTag(pos, send_qkv_dst_rank, data_type),
                           spRankList[send_qkv_dst_rank], kv_shape, output_type_id, qkv_group);
      } else if (inner_step == 0) {
        return NewSendNode(CreateDepend(tmp_node, pre_node), GetSendRecvTag(pos, send_qkv_dst_rank, data_type),
                           spRankList[send_qkv_dst_rank], kv_shape, output_type_id, qkv_group);
      }
    }
  }
  return nullptr;
}

CNodePtr GetCurrentRecvQKVNode(size_t pos, size_t step, size_t inner_step, size_t sp_num, RankList spRankList,
                               Shape q_shape, Shape kv_shape, const std::string &qkv_group, const AnfNodePtr &pre_node,
                               TypeId output_type_id) {
  if (step < (sp_num / kIndex2)) {  // [0, sp-1]
    auto recv_qkv_src_rank = GetRecvQKVSrcRank(pos, step, sp_num);
    if (pos < step + kIndex1) {
      if (inner_step == kIndex0) {  // recv q
        return NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, TagType::query),
                              spRankList[recv_qkv_src_rank], q_shape, output_type_id, qkv_group);
      }
    } else {  // recv kv
      auto data_type = inner_step == kIndex0 ? TagType::kv_b : TagType::kv_a;
      if (step < (sp_num / kIndex2) - kIndex1) {
        return NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, data_type),
                              spRankList[recv_qkv_src_rank], kv_shape, output_type_id, qkv_group);
      } else {
        if (inner_step == 0) {
          return NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, data_type),
                                spRankList[recv_qkv_src_rank], kv_shape, output_type_id, qkv_group);
        }
      }
    }
  }
  return nullptr;
}

CNodePtr GetCurrentSendOMLNode(size_t pos, size_t step, size_t inner_step, size_t sp_num, RankList spRankList,
                               const AnfNodePtr &send_softmax_max, const AnfNodePtr &send_softmax_sum,
                               const AnfNodePtr &send_attn_out, const std::string &send_group,
                               const AnfNodePtr &pre_node, Shape q_shape, TypeId output_type_id) {
  auto recv_oml_shape = q_shape;
  recv_oml_shape[kIndex3] = recv_oml_shape[kIndex3] + kIndex16;
  if (step > kIndex0 && step < (sp_num / kIndex2) + kIndex1) {
    if (pos < (sp_num / kIndex2)) {
      auto send_oml_dst_rank = GetSendOMLDstRank(pos, step, sp_num);
      if (pos < step) {
        if (step < (sp_num / kIndex2)) {
          if (inner_step == kIndex1) {
            return NewSendNode(
              CreateDepend(ConstructSendOMLTensor(send_softmax_max, send_softmax_sum, send_attn_out, output_type_id),
                           pre_node),
              GetSendRecvTag(pos, send_oml_dst_rank, TagType::oml), spRankList[send_oml_dst_rank], recv_oml_shape,
              output_type_id, send_group);
          }
        } else {
          if (inner_step == kIndex0) {
            return NewSendNode(
              CreateDepend(ConstructSendOMLTensor(send_softmax_max, send_softmax_sum, send_attn_out, output_type_id),
                           pre_node),
              GetSendRecvTag(pos, send_oml_dst_rank, TagType::oml), spRankList[send_oml_dst_rank], recv_oml_shape,
              output_type_id, send_group);
          }
        }
      }
    }
  }
  return nullptr;
}

CNodePtr GetCurrentRecvOMLNode(size_t pos, size_t step, size_t inner_step, size_t sp_num, RankList spRankList,
                               Shape q_shape, const std::string &recv_group, const AnfNodePtr &pre_node,
                               TypeId output_type_id) {
  if (step <= kIndex0 || step >= (sp_num / kIndex2) + kIndex1) {
    return nullptr;
  }
  if (pos >= (sp_num / kIndex2)) {
    auto recv_oml_src_rank = GetRecvOMLSrcRank(pos, step, sp_num);
    if (pos + step >= sp_num) {
      if (step < (sp_num / kIndex2)) {
        if (inner_step == kIndex1) {
          auto recv_oml_shape = q_shape;
          recv_oml_shape[kIndex3] = recv_oml_shape[kIndex3] + kIndex16;
          return NewReceiveNode(pre_node, GetSendRecvTag(recv_oml_src_rank, pos, TagType::oml),
                                spRankList[recv_oml_src_rank], recv_oml_shape, output_type_id, recv_group);
        }
      } else {
        if (inner_step == 0) {
          auto recv_oml_q_shape = q_shape;
          recv_oml_q_shape[kIndex3] = recv_oml_q_shape[kIndex3] + kIndex16;
          return NewReceiveNode(pre_node, GetSendRecvTag(recv_oml_src_rank, pos, TagType::oml),
                                spRankList[recv_oml_src_rank], recv_oml_q_shape, output_type_id, recv_group);
        }
      }
    }
  }
  return nullptr;
}

void ChangeQKVToBNSD(AnfNodePtr *query_node, AnfNodePtr *key_node, AnfNodePtr *value_node, Shape *q_shape,
                     Shape *kv_shape, int64_t fa_b, int64_t fa_s1, int64_t fa_h1, int64_t fa_s2, int64_t fa_h2,
                     int64_t fa_d, int64_t fa_n1) {
  (*query_node) = NewReshapeNode((*query_node), {fa_b, fa_s1, fa_n1, fa_d});
  AnfNodePtr tmp_tup = parallel::CreateTuple({0, 2, 1, 3});
  (*query_node) = NewTransposeNode((*query_node), tmp_tup);

  (*key_node) = NewReshapeNode((*key_node), {fa_b, fa_s2, fa_h2 / fa_d, fa_d});
  (*key_node) = NewTransposeNode((*key_node), tmp_tup);

  (*value_node) = NewReshapeNode((*value_node), {fa_b, fa_s2, fa_h2 / fa_d, fa_d});
  (*value_node) = NewTransposeNode((*value_node), tmp_tup);
  (*q_shape) = {fa_b, fa_n1, fa_s1, fa_d};
  (*kv_shape) = {fa_b, fa_h2 / fa_d, fa_s2, fa_d};
}

void DynChangeQKVToBNSD(AnfNodePtr *query_node, AnfNodePtr *key_node, AnfNodePtr *value_node, CNodePtr *q_shape,
                        CNodePtr *kv_shape, const AnfNodePtr &fa_b, const AnfNodePtr &fa_s1, const AnfNodePtr &fa_h1,
                        const AnfNodePtr &fa_s2, const AnfNodePtr &fa_h2, const AnfNodePtr &fa_d, int64_t fa_n1) {
  (*query_node) = NewDynReshapeNode((*query_node), NewMakeTupleNode({fa_b, fa_s1, NewValueNode<int64_t>(fa_n1), fa_d}),
                                    TypeId::kNumberTypeFloat16);
  AnfNodePtr tmp_tup = parallel::CreateTuple({0, 2, 1, 3});
  (*query_node) = NewTransposeNode((*query_node), tmp_tup);

  (*key_node) = NewDynReshapeNode((*key_node), NewMakeTupleNode({fa_b, fa_s2, NewScalarDivNode(fa_h2, fa_d), fa_d}),
                                  TypeId::kNumberTypeFloat16);
  (*key_node) = NewTransposeNode((*key_node), tmp_tup);

  (*value_node) = NewDynReshapeNode((*value_node), NewMakeTupleNode({fa_b, fa_s2, NewScalarDivNode(fa_h2, fa_d), fa_d}),
                                    TypeId::kNumberTypeFloat16);
  (*value_node) = NewTransposeNode((*value_node), tmp_tup);
  (*q_shape) = NewMakeTupleNode({fa_b, NewValueNode<int64_t>(fa_n1), fa_s1, fa_d});
  (*kv_shape) = NewMakeTupleNode({fa_b, NewScalarDivNode(fa_h2, fa_d), fa_s2, fa_d});
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
                       size_t actual_step, int64_t fa_s1, int64_t fa_s2, const std::vector<AnfNodePtr> &kv_a_nodes,
                       const std::vector<AnfNodePtr> &kv_b_nodes, size_t pos, size_t sp_num, int64_t inner_step) {
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
    if (pos < (sp_num / kIndex2)) {
      if ((kIndex2 * pos < actual_step) && (actual_step < sp_num)) {  // recv_qkv_tensor is q, compute others oml
        (*cur_q) = recv_qkv_tensor;
        if (inner_step == kIndex0) {
          (*cur_k) = kv_b_nodes[kIndex0];
          (*cur_v) = kv_b_nodes[kIndex1];
        } else {
          (*cur_k) = kv_a_nodes[kIndex0];
          (*cur_v) = kv_a_nodes[kIndex1];
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

void DynGetCurrentQKVMask(const AnfNodePtr &kv_a_concat, const AnfNodePtr &kv_b_concat,
                          const AnfNodePtr &recv_qkv_tensor, const AnfNodePtr &query_node, const AnfNodePtr &key_node,
                          const AnfNodePtr &value_node, AnfNodePtr *cur_attn_mask, AnfNodePtr *cur_q, AnfNodePtr *cur_k,
                          AnfNodePtr *cur_v, int64_t actual_step, const CNodePtr &fa_s1, const CNodePtr &fa_s2,
                          const std::vector<AnfNodePtr> &kv_a_nodes, const std::vector<AnfNodePtr> &kv_b_nodes,
                          int64_t pos, int64_t sp_num, int64_t inner_step) {
  if (actual_step == 0) {
    (*cur_attn_mask) = dyn_make_start_mask_tensor(fa_s1, fa_s2);
    (*cur_q) = query_node;
    (*cur_k) = key_node;
    (*cur_v) = value_node;
    (*cur_q) = CreateDepend((*cur_q), kv_a_concat);
    (*cur_q) = CreateDepend((*cur_q), kv_b_concat);
  } else if (actual_step == sp_num) {
    (*cur_attn_mask) = dyn_make_end_mask_tensor(fa_s1, fa_s2);
    (*cur_q) = query_node;
    (*cur_k) = key_node;
    (*cur_v) = value_node;
  } else {
    (*cur_attn_mask) = dyn_make_mask_tensor(fa_s1, NewScalarDivNode(fa_s2, NewValueNode<int64_t>(kIndex2)), 0, false);
    if (pos < (sp_num / SizeToLong(kIndex2))) {
      if ((SizeToLong(kIndex2) * pos < actual_step) &&
          (actual_step < sp_num)) {  // recv_qkv_tensor is q, compute others oml
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
                        CNodePtr *latest_send_oml, CNodePtr *latest_recv_oml, std::string *comm_order_str, size_t pos,
                        size_t step, size_t inner_step, size_t sp_num, const AnfNodePtr &query_node,
                        const std::vector<CNodePtr> &send_kv_node, RankList spRankList,
                        const AnfNodePtr &send_softmax_max, const AnfNodePtr &send_softmax_sum,
                        const AnfNodePtr &send_attn_out, Shape q_shape, Shape kv_shape,
                        const std::string &flash_index_str, AnfNodePtr *cur_q, TypeId output_type_id) {
  std::stringstream comm_order;
  if (rank_ring_index % kIndex2 == 0) {  // send first
    auto cur_send_qkv_node =
      GetCurrentSendQKVNode(pos, step, inner_step, sp_num, query_node, send_kv_node, spRankList,
                            g_device_manager->world_group(), *cur_q, q_shape, kv_shape, output_type_id);
    auto depend_node = (cur_send_qkv_node == nullptr ? *cur_q : cur_send_qkv_node);
    auto cur_recv_qkv_node = GetCurrentRecvQKVNode(pos, step, inner_step, sp_num, spRankList, q_shape, kv_shape,
                                                   g_device_manager->world_group(), depend_node, output_type_id);
    (*latest_send_qkv) = cur_send_qkv_node;
    (*latest_recv_qkv) = cur_recv_qkv_node;

    if (cur_recv_qkv_node != nullptr) {
      depend_node = cur_recv_qkv_node;
    } else if (cur_send_qkv_node != nullptr) {
      depend_node = cur_send_qkv_node;
    } else {
      depend_node = *cur_q;
    }
    auto cur_send_oml_node =
      GetCurrentSendOMLNode(pos, step, inner_step, sp_num, spRankList, send_softmax_max, send_softmax_sum,
                            send_attn_out, g_device_manager->world_group(), depend_node, q_shape, output_type_id);
    auto cur_recv_oml_node = GetCurrentRecvOMLNode(pos, step, inner_step, sp_num, spRankList, q_shape,
                                                   g_device_manager->world_group(), depend_node, output_type_id);
    (*latest_recv_oml) = cur_recv_oml_node;
    (*latest_send_oml) = cur_send_oml_node;
    *cur_q = CreateDepends(*cur_q, {cur_send_qkv_node, cur_recv_qkv_node, cur_send_oml_node, cur_recv_oml_node});
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
                                                   g_device_manager->world_group(), *cur_q, output_type_id);
    auto depend_node = (cur_recv_qkv_node == nullptr ? *cur_q : cur_recv_qkv_node);
    auto cur_send_qkv_node =
      GetCurrentSendQKVNode(pos, step, inner_step, sp_num, query_node, send_kv_node, spRankList,
                            g_device_manager->world_group(), depend_node, q_shape, kv_shape, output_type_id);
    (*latest_send_qkv) = cur_send_qkv_node;
    (*latest_recv_qkv) = cur_recv_qkv_node;

    if (cur_send_qkv_node != nullptr) {
      depend_node = cur_send_qkv_node;
    } else if (cur_recv_qkv_node != nullptr) {
      depend_node = cur_recv_qkv_node;
    } else {
      depend_node = *cur_q;
    }
    auto cur_send_oml_node =
      GetCurrentSendOMLNode(pos, step, inner_step, sp_num, spRankList, send_softmax_max, send_softmax_sum,
                            send_attn_out, g_device_manager->world_group(), depend_node, q_shape, output_type_id);
    auto cur_recv_oml_node = GetCurrentRecvOMLNode(pos, step, inner_step, sp_num, spRankList, q_shape,
                                                   g_device_manager->world_group(), depend_node, output_type_id);
    (*latest_send_oml) = cur_send_oml_node;
    (*latest_recv_oml) = cur_recv_oml_node;
    *cur_q = CreateDepends(*cur_q, {cur_recv_qkv_node, cur_send_qkv_node, cur_send_oml_node, cur_recv_oml_node});
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

void HandleFAResult(size_t actual_step, CNodePtr *acc_attention, CNodePtr *history_max, CNodePtr *history_sum,
                    CNodePtr *cur_attn_out, CNodePtr *cur_softmax_max, CNodePtr *cur_softmax_sum,
                    CNodePtr *send_softmax_max, CNodePtr *send_softmax_sum, CNodePtr *send_attn_out,
                    CNodePtr *latest_send_oml, CNodePtr *latest_recv_oml, CNodePtr *latest_fa_op, int64_t fa_b,
                    int64_t fa_s1, int64_t fa_n1, int64_t fa_h1, int64_t fa_index, Shape q_shape, size_t pos,
                    size_t sp_num, int64_t inner_step, TypeId output_type_id) {
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
    bool need_update = *latest_recv_oml == nullptr ? true : false;
    UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum, *cur_attn_out,
                          fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step, output_type_id,
                          need_update);
    *latest_fa_op = *acc_attention;
    if ((*latest_recv_oml) != nullptr) {
      *latest_recv_oml = CreateDepend(*latest_recv_oml, *latest_fa_op);
      DismantleRecvOMLTensor(*latest_recv_oml, cur_attn_out, cur_softmax_max, cur_softmax_sum, q_shape);
      UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum, *cur_attn_out,
                            fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step, output_type_id,
                            true);
      *latest_fa_op = *acc_attention;
    }
  } else {
    if (pos < (sp_num / kIndex2)) {
      if ((kIndex2 * pos < actual_step) && (actual_step < sp_num)) {  // compute others oml
        if (inner_step == 0) {
          UpdateAttentionOutput(send_softmax_max, send_softmax_sum, send_attn_out, *cur_softmax_max, *cur_softmax_sum,
                                *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                                actual_step, output_type_id);
          *latest_fa_op = *send_attn_out;
        } else {
          *send_attn_out = *cur_attn_out;
          *send_softmax_max = *cur_softmax_max;
          *send_softmax_sum = *cur_softmax_sum;
        }
      } else {
        UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                              *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step,
                              output_type_id);
        *latest_fa_op = *acc_attention;
      }
    } else {
      UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum, *cur_attn_out,
                            fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step, output_type_id);
      *latest_fa_op = *acc_attention;
      if ((*latest_recv_oml) != nullptr) {
        *latest_recv_oml = CreateDepend(*latest_recv_oml, *latest_fa_op);
        DismantleRecvOMLTensor(*latest_recv_oml, cur_attn_out, cur_softmax_max, cur_softmax_sum, q_shape);
        UpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                              *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step,
                              output_type_id);
        *latest_fa_op = *acc_attention;
      }
    }
  }
}
void DynHandleFAResult(size_t actual_step, CNodePtr *acc_attention, CNodePtr *history_max, CNodePtr *history_sum,
                       CNodePtr *cur_attn_out, CNodePtr *cur_softmax_max, CNodePtr *cur_softmax_sum,
                       CNodePtr *send_softmax_max, CNodePtr *send_softmax_sum, CNodePtr *send_attn_out,
                       CNodePtr *latest_send_oml, CNodePtr *latest_recv_oml, CNodePtr *latest_fa_op,
                       const CNodePtr &fa_b, const CNodePtr &fa_s1, int64_t fa_n1, const CNodePtr &fa_h1,
                       int64_t fa_index, const CNodePtr &q_shape, size_t pos, size_t sp_num, int64_t inner_step) {
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
    DynUpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum, *cur_attn_out,
                             fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index, actual_step);
    *latest_fa_op = *acc_attention;
    if ((*latest_recv_oml) != nullptr) {
      DynDismantleRecvOMLTensor(*latest_recv_oml, cur_attn_out, cur_softmax_max, cur_softmax_sum, q_shape);
      DynUpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                               *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                               actual_step);
      *latest_fa_op = *acc_attention;
    }
  } else {
    if (pos < (sp_num / kIndex2)) {
      if ((kIndex2 * pos < actual_step) && (actual_step < sp_num)) {  // compute others oml
        if (inner_step == 0) {
          DynUpdateAttentionOutput(send_softmax_max, send_softmax_sum, send_attn_out, *cur_softmax_max,
                                   *cur_softmax_sum, *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD,
                                   fa_index, actual_step);
          *latest_fa_op = *send_attn_out;
        } else {
          *send_attn_out = *cur_attn_out;
          *send_softmax_max = *cur_softmax_max;
          *send_softmax_sum = *cur_softmax_sum;
        }
      } else {
        DynUpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                                 *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                                 actual_step);
        *latest_fa_op = *acc_attention;
      }
    } else {
      DynUpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                               *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                               actual_step);
      *latest_fa_op = *acc_attention;
      if ((*latest_recv_oml) != nullptr) {
        DynDismantleRecvOMLTensor(*latest_recv_oml, cur_attn_out, cur_softmax_max, cur_softmax_sum, q_shape);
        DynUpdateAttentionOutput(history_max, history_sum, acc_attention, *cur_softmax_max, *cur_softmax_sum,
                                 *cur_attn_out, fa_b, fa_s1, fa_n1, fa_h1, FASInputLayoutMode::BNSD, fa_index,
                                 actual_step);
        *latest_fa_op = *acc_attention;
      }
    }
  }
}

bool IsDynamicShape(const CNodePtr &fa_score_node) {
  MS_EXCEPTION_IF_NULL(fa_score_node);
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  for (size_t i = 0; i < q_shape.size(); ++i) {
    if (q_shape[i] == -1) {
      return true;
    }
  }
  for (size_t i = 0; i < kv_shape.size(); ++i) {
    if (kv_shape[i] == -1) {
      return true;
    }
  }
  return false;
}

CNodePtr CreateReplaceFlashSPGraph(const FuncGraphManagerPtr &manager,
                                   const std::vector<CNodePtr> &origin_nodes_topological, const CNodePtr &fa_score_node,
                                   FSPInfo *fsp_info, int fa_index) {
  std::vector<AnfNodePtr> fa_inputs, kv_a_nodes, kv_b_nodes;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    MS_EXCEPTION_IF_NULL(fa_score_node->input(i + 1));
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }
  size_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto spRankList = flash_score_info_ptr->GetSPRankList();
  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  auto query_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex + 1);
  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(fa_score_node, kIndex3);

  auto input_layout = flash_score_info_ptr->input_layout();
  int64_t fa_b, fa_s1, fa_h1, fa_s2, fa_h2, fa_n1, fa_d;
  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_h2, &fa_n1, fa_score_node);
  fa_d = fa_h1 / fa_n1;
  if (input_layout == FASInputLayoutMode::BSH) {
    ChangeQKVToBNSD(&query_node, &key_node, &value_node, &q_shape, &kv_shape, fa_b, fa_s1, fa_h1, fa_s2, fa_h2, fa_d,
                    fa_n1);
  }

  SplitKVNode(&kv_a_nodes, &kv_b_nodes, key_node, value_node);
  auto kv_a_concat = NewConcatNode(NewMakeTupleNode(kv_a_nodes), kIndex2);
  auto kv_b_concat = NewConcatNode(NewMakeTupleNode(kv_b_nodes), kIndex2);
  auto send_kv_node = {kv_a_concat, kv_b_concat};

  CNodePtr attn_out, softmax_max, softmax_sum, send_attn_out, send_softmax_max, send_softmax_sum;
  CNodePtr send_oml_tensor, recv_oml_tensor, send_qkv_tensor, recv_qkv_tensor;
  CNodePtr history_max, history_sum, acc_attention, local_fa_node;
  CNodePtr latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml, latest_fa_op;

  for (size_t step = kIndex0; step < (sp_num / kIndex2) + kIndex1; ++step) {
    for (size_t inner_step = kIndex0; inner_step < kIndex2; ++inner_step) {
      auto rank_ring_index = GetRankIndex(pos, step, sp_num);
      auto actual_step = kIndex2 * step + inner_step;
      if (actual_step == sp_num + kIndex1) {
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

      std::string comm_order_str;
      std::string ss_result = GetFlashIndexString(fa_index, actual_step);
      GetCurrentCommNode(rank_ring_index, &latest_send_qkv, &latest_recv_qkv, &latest_send_oml, &latest_recv_oml,
                         &comm_order_str, pos, step, inner_step, sp_num, query_node, send_kv_node, spRankList,
                         send_softmax_max, send_softmax_sum, send_attn_out, q_shape, kv_shape, ss_result, &cur_q,
                         output_type_id);
      if (latest_recv_qkv != nullptr) {
        recv_qkv_tensor = latest_recv_qkv;
      }

      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex] = cur_q;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = cur_k;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = cur_v;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputLayoutIndex] =
        NewValueNode<int64_t>(FASInputLayoutMode::BNSD);
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = cur_attn_mask;
      local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, actual_step, true);
      common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);
      local_fa_node->AddPrimalAttr("comm_order", MakeValue<std::string>(comm_order_str));
      local_fa_node->AddPrimalAttr("sp_num", MakeValue<int64_t>(sp_num));
      local_fa_node->set_user_data<parallel::OperatorInfo>(operator_info);
      latest_fa_op = local_fa_node;

      auto cur_softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
      auto cur_softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);
      auto cur_attn_out = NewTupleGetItemNode(local_fa_node, kIndex3);
      HandleFAResult(actual_step, &acc_attention, &history_max, &history_sum, &cur_attn_out, &cur_softmax_max,
                     &cur_softmax_sum, &send_softmax_max, &send_softmax_sum, &send_attn_out, &latest_send_oml,
                     &latest_recv_oml, &latest_fa_op, fa_b, fa_s1, fa_n1, fa_h1, fa_index, q_shape, pos, sp_num,
                     inner_step, output_type_id);
    }
  }
  acc_attention = CreateDepends(acc_attention, {latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml});
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_UPDATE_ATTN, MakeValue<int>(fa_index), acc_attention);
  acc_attention = NewCastNode(acc_attention, output_type_id);
  if (input_layout == FASInputLayoutMode::BSH) {
    auto tmp_tup1 = parallel::CreateTuple({0, 2, 1, 3});
    acc_attention = NewTransposeNode(acc_attention, tmp_tup1);
    acc_attention = NewReshapeNode(acc_attention, {fa_b, fa_s1, fa_h1});
  }
  auto softmax_out = NewTupleGetItemNode(local_fa_node, 2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

CNodePtr DynCreateReplaceFlashSPGraphResult(CNodePtr *acc_attention, const CNodePtr &latest_send_qkv,
                                            const CNodePtr &latest_recv_qkv, const CNodePtr &latest_send_oml,
                                            const CNodePtr &latest_recv_oml, int64_t input_layout,
                                            const CNodePtr &fa_b_dyn, const CNodePtr &fa_s1_dyn,
                                            const CNodePtr &fa_h1_dyn, const CNodePtr &local_fa_node,
                                            const CNodePtr &history_max, const CNodePtr &history_sum) {
  *acc_attention = CreateDepends(*acc_attention, {latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml});
  *acc_attention = NewCastNode(*acc_attention, TypeId::kNumberTypeFloat16);
  if (input_layout == FASInputLayoutMode::BSH) {
    auto tmp_tup1 = parallel::CreateTuple({0, 2, 1, 3});
    *acc_attention = NewTransposeNode(*acc_attention, tmp_tup1);
    *acc_attention =
      NewDynReshapeNode(*acc_attention, NewMakeTupleNode({fa_b_dyn, fa_s1_dyn, fa_h1_dyn}), TypeId::kNumberTypeFloat16);
  }
  auto softmax_out = NewTupleGetItemNode(local_fa_node, 2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, *acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

CNodePtr DynCreateReplaceFlashSPGraph(const FuncGraphManagerPtr &manager,
                                      const std::vector<CNodePtr> &origin_nodes_topological,
                                      const CNodePtr &fa_score_node, FSPInfo *fsp_info, int fa_index) {
  std::vector<AnfNodePtr> fa_inputs, kv_a_nodes, kv_b_nodes;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    MS_EXCEPTION_IF_NULL(fa_score_node->input(i + 1));
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }
  size_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto spRankList = flash_score_info_ptr->GetSPRankList();
  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  auto query_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex + 1);
  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(fa_score_node, kIndex3);

  CNodePtr q_dynshape_node, kv_dynshape_node;
  q_dynshape_node = NewDynshapeNode(fa_score_node->input(kIndex0 + 1));
  kv_dynshape_node = NewDynshapeNode(fa_score_node->input(kIndex1 + 1));

  auto input_layout = flash_score_info_ptr->input_layout();
  int64_t fa_b, fa_s1, fa_h1, fa_s2, fa_h2, fa_n1, fa_d;
  CNodePtr fa_b_dyn, fa_s1_dyn, fa_h1_dyn, fa_s2_dyn, fa_h2_dyn, fa_d_dyn;

  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_h2, &fa_n1, fa_score_node);
  GetBSHFromDynShape(input_layout, &fa_b_dyn, &fa_s1_dyn, &fa_h1_dyn, &fa_s2_dyn, &fa_h2_dyn, &fa_n1, q_dynshape_node,
                     kv_dynshape_node, fa_score_node);
  fa_d = fa_h1 / fa_n1;
  fa_d_dyn = NewScalarDivNode(fa_h1_dyn, NewValueNode<int64_t>(fa_n1));

  if (input_layout == FASInputLayoutMode::BSH) {
    DynChangeQKVToBNSD(&query_node, &key_node, &value_node, &q_dynshape_node, &kv_dynshape_node, fa_b_dyn, fa_s1_dyn,
                       fa_h1_dyn, fa_s2_dyn, fa_h2_dyn, fa_d_dyn, fa_n1);
    q_shape = {fa_b, fa_n1, fa_s1, fa_d};
    kv_shape = {fa_b, fa_h2 / fa_d, fa_s2, fa_d};
  }

  SplitKVNode(&kv_a_nodes, &kv_b_nodes, key_node, value_node);
  auto kv_a_concat = NewConcatNode(NewMakeTupleNode(kv_a_nodes), kIndex2);
  auto kv_b_concat = NewConcatNode(NewMakeTupleNode(kv_b_nodes), kIndex2);
  auto send_kv_node = {kv_a_concat, kv_b_concat};

  CNodePtr attn_out, softmax_max, softmax_sum, send_attn_out, send_softmax_max, send_softmax_sum;
  CNodePtr send_oml_tensor, recv_oml_tensor, send_qkv_tensor, recv_qkv_tensor;
  CNodePtr history_max, history_sum, acc_attention, local_fa_node;
  CNodePtr latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml, latest_fa_op;

  for (size_t step = kIndex0; step < (sp_num / kIndex2) + kIndex1; ++step) {
    for (size_t inner_step = kIndex0; inner_step < kIndex2; ++inner_step) {
      auto rank_ring_index = GetRankIndex(pos, step, sp_num);
      auto actual_step = kIndex2 * step + inner_step;
      if (actual_step == sp_num + kIndex1) {
        continue;
      }
      AnfNodePtr cur_attn_mask, cur_q, cur_k, cur_v;
      if (recv_qkv_tensor != nullptr) {
        recv_qkv_tensor = CreateDepends(
          recv_qkv_tensor, {latest_fa_op, latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml});
      }
      DynGetCurrentQKVMask(kv_a_concat, kv_b_concat, recv_qkv_tensor, query_node, key_node, value_node, &cur_attn_mask,
                           &cur_q, &cur_k, &cur_v, actual_step, fa_s1_dyn, fa_s2_dyn, kv_a_nodes, kv_b_nodes, pos,
                           sp_num, inner_step);
      cur_q = CreateDepends(
        cur_q, {latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml, latest_fa_op, cur_attn_mask});

      std::string comm_order_str, ss_result = GetFlashIndexString(fa_index, actual_step);
      GetCurrentCommNode(rank_ring_index, &latest_send_qkv, &latest_recv_qkv, &latest_send_oml, &latest_recv_oml,
                         &comm_order_str, pos, step, inner_step, sp_num, query_node, send_kv_node, spRankList,
                         send_softmax_max, send_softmax_sum, send_attn_out, q_shape, kv_shape, ss_result, &cur_q,
                         output_type_id);
      if (latest_recv_qkv != nullptr) {
        recv_qkv_tensor = latest_recv_qkv;
      }

      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex] = cur_q;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = cur_k;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = cur_v;
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputLayoutIndex] =
        NewValueNode<int64_t>(FASInputLayoutMode::BNSD);
      fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = cur_attn_mask;
      local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, actual_step, true);
      common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);
      local_fa_node->AddPrimalAttr("comm_order", MakeValue<std::string>(comm_order_str));
      local_fa_node->AddPrimalAttr("sp_num", MakeValue<int64_t>(sp_num));
      latest_fa_op = local_fa_node;

      auto cur_softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
      auto cur_softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);
      auto cur_attn_out = NewTupleGetItemNode(local_fa_node, kIndex3);
      DynHandleFAResult(actual_step, &acc_attention, &history_max, &history_sum, &cur_attn_out, &cur_softmax_max,
                        &cur_softmax_sum, &send_softmax_max, &send_softmax_sum, &send_attn_out, &latest_send_oml,
                        &latest_recv_oml, &latest_fa_op, fa_b_dyn, fa_s1_dyn, fa_n1, fa_h1_dyn, fa_index,
                        q_dynshape_node, pos, sp_num, inner_step);
    }
  }

  auto attention_results = DynCreateReplaceFlashSPGraphResult(
    &acc_attention, latest_send_qkv, latest_recv_qkv, latest_send_oml, latest_recv_oml, input_layout, fa_b_dyn,
    fa_s1_dyn, fa_h1_dyn, local_fa_node, history_max, history_sum);
  return attention_results;
}

void SetFAInputs(const AnfNodePtr &query_node, const AnfNodePtr &key_node, const AnfNodePtr &value_node,
                 const AnfNodePtr &attn_node, const std::shared_ptr<OperatorInfo> &operator_info,
                 const std::vector<AnfNodePtr> &eod_masks, int64_t sp_num, int64_t index, int64_t pos,
                 const Shape &shape, std::vector<AnfNodePtr> *fa_inputs, AnfNodePtr *first_actual_mask,
                 AnfNodePtr *full_mask) {
  AnfNodePtr actual_mask;
  auto pos_index = GetUDMaskIndex(index, pos, sp_num);
  if (query_node != nullptr) {
    (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = query_node;
  }
  (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = key_node;
  (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = value_node;
  if (!IsValueNode<None>(attn_node)) {
    auto attn_mask_shape = operator_info->inputs_tensor_info()[kIndex6].tensor_layout().base_slice_shape().array();
    auto attn_mask_split_node = NewSplitNode(attn_node, attn_mask_shape.size() - kIndex1,
                                             sp_num);  // mask has been split in the last dim by sp_num
    actual_mask = NewTupleGetItemNode(attn_mask_split_node, pos_index);
  } else if (IsValueNode<None>(attn_node) && !eod_masks.empty()) {  // eod reset attention mask
    actual_mask = eod_masks[pos_index];
  } else {
    if (index == 0) {
      (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputSparseModeIndex] =
        CreatInt64Imm(kIndex3);
      actual_mask = *first_actual_mask;
    } else {
      (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputSparseModeIndex] =
        CreatInt64Imm(kIndex0);
      actual_mask = (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputPaddingMaskIndex];
      if (index > pos) {
        actual_mask = *full_mask;
        (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = actual_mask;
        (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputSparseModeIndex] =
          CreatInt64Imm(kIndex4);
        (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputPreTokensIndex] =
          CreatInt64Imm(kIndex0);
        (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputNextTokensIndex] =
          CreatInt64Imm(kIndex0);
      }
    }
  }
  if (actual_mask != nullptr) {
    (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = actual_mask;
  }
}

void DynSetFAInputs(const AnfNodePtr &query_node, const AnfNodePtr &key_node, const AnfNodePtr &value_node,
                    const AnfNodePtr &attn_node, const std::shared_ptr<OperatorInfo> &operator_info,
                    const std::vector<AnfNodePtr> &eod_masks, int64_t sp_num, int64_t index, int64_t pos,
                    const AnfNodePtr &fa_s1, const AnfNodePtr &fa_s2, std::vector<AnfNodePtr> *fa_inputs) {
  AnfNodePtr actual_mask;
  auto pos_index = GetUDMaskIndex(index, pos, sp_num);
  if (query_node != nullptr) {
    (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = query_node;
  }
  (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = key_node;
  (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = value_node;
  if (!IsValueNode<None>(attn_node)) {
    auto attn_mask_shape = operator_info->inputs_tensor_info()[kIndex6].tensor_layout().base_slice_shape().array();
    auto attn_mask_split_node = NewSplitNode(attn_node, attn_mask_shape.size() - kIndex1,
                                             sp_num);  // mask has been split in the last dim by sp_num
    actual_mask = NewTupleGetItemNode(attn_mask_split_node, pos_index);
  } else if (IsValueNode<None>(attn_node) && !eod_masks.empty()) {  // eod reset attention mask
    actual_mask = eod_masks[pos_index];
  } else {
    actual_mask = DynGetActualMask(index, pos, fa_s1, fa_s2);
  }
  if (actual_mask != nullptr) {
    (*fa_inputs)[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = actual_mask;
  }
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
  int64_t sp_num = SizeToLong(fsp_info->GetSPNum());
  int64_t rank_id = SizeToLong(fsp_info->GetRankId());
  int64_t send_rank_id = fsp_info->GetSendRankId(), recv_rank_id = fsp_info->GetRecvRankId();

  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto input_layout = flash_score_info_ptr->input_layout();
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(fa_score_node, kIndex3);

  int64_t fa_b, fa_s1, fa_h1, fa_s2, fa_h2, fa_n1;
  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_h2, &fa_n1, fa_score_node);

  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  std::vector<AnfNodePtr> eod_masks;
  for (int i = 0; i < sp_num; ++i) {
    GenerateEodMask(i, pos, sp_num, actual_shape_size, {fa_s1, fa_s2}, actual_node, &eod_masks);
  }
  CNodePtr local_fa_node, kv_received_tuple, softmax_max, softmax_sum, softmax_out, attention_output;
  CNodePtr history_max, history_sum, acc_attention;
  AnfNodePtr actual_mask, first_actual_mask, full_mask;
  first_actual_mask = GetActualMask(0, pos, TypeId::kNumberTypeUInt8, {2048, 2048});
  auto full_mask_tensor = make_mask_tensor(TypeId::kNumberTypeUInt8, {2048, 2048}, 0, true);
  full_mask = NewValueNode(MakeValue(full_mask_tensor));
  for (int i = 0; i < sp_num; ++i) {
    std::vector<AnfNodePtr> kv_nodes = {key_node, value_node};
    auto kv_tuple = NewMakeTupleNode(kv_nodes);
    auto kv_concat = NewConcatNode(kv_tuple, 0);
    std::vector<AnfNodePtr> concat_tuple = {kv_concat};
    auto kv_concat_tuple = NewMakeTupleNode(concat_tuple);
    if (i != sp_num - 1) {
      auto neigh_shape = kv_shape;
      if (neigh_shape[0] != -1) {
        neigh_shape[0] = neigh_shape[0] * kIndex2;
      }
      kv_received_tuple = NewNeighborExchangeNode(kv_concat_tuple, {send_rank_id}, {recv_rank_id}, fa_index, i,
                                                  neigh_shape, output_type_id);
    }

    SetFAInputs(nullptr, key_node, value_node, attn_node, operator_info, eod_masks, sp_num, i, pos, Shape{fa_s1, fa_s2},
                &fa_inputs, &first_actual_mask, &full_mask);

    local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, i, false);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);

    if (i != sp_num - 1) {
      auto kv_exchanged_item = NewTupleGetItemNode(kv_received_tuple, kIndex0);
      auto kv_split = NewSplitNode(kv_exchanged_item, kIndex0, kIndex2);
      key_node = NewTupleGetItemNode(kv_split, kIndex0);
      value_node = NewTupleGetItemNode(kv_split, kIndex1);
    }

    attention_output = NewTupleGetItemNode(local_fa_node, kIndex3);
    softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
    softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);

    if (i == 0) {
      acc_attention = attention_output->cast<CNodePtr>();
      history_max = softmax_max->cast<CNodePtr>();
      history_sum = softmax_sum->cast<CNodePtr>();
    } else {
      UpdateAttentionOutput(&history_max, &history_sum, &acc_attention, softmax_max, softmax_sum, attention_output,
                            fa_b, fa_s1, fa_n1, fa_h1, input_layout, fa_index, i, output_type_id);
    }
  }
  acc_attention = NewCastNode(acc_attention, output_type_id);
  softmax_out = NewTupleGetItemNode(local_fa_node, kIndex2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

CNodePtr DynCreateReplaceRingAttentionGraphByAllToAllv(const FuncGraphManagerPtr &manager,
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
  int64_t sp_num = SizeToLong(fsp_info->GetSPNum());
  int64_t rank_id = SizeToLong(fsp_info->GetRankId());
  int64_t send_rank_id = fsp_info->GetSendRankId(), recv_rank_id = fsp_info->GetRecvRankId();

  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto input_layout = flash_score_info_ptr->input_layout();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(fa_score_node, kIndex3);

  CNodePtr q_dynshape_node, kv_dynshape_node;

  int64_t fa_n1;
  CNodePtr fa_b_dyn, fa_s1_dyn, fa_h1_dyn, fa_s2_dyn, fa_h2_dyn;
  q_dynshape_node = NewDynshapeNode(fa_score_node->input(kIndex0 + 1));
  kv_dynshape_node = NewDynshapeNode(fa_score_node->input(kIndex1 + 1));
  GetBSHFromDynShape(input_layout, &fa_b_dyn, &fa_s1_dyn, &fa_h1_dyn, &fa_s2_dyn, &fa_h2_dyn, &fa_n1, q_dynshape_node,
                     kv_dynshape_node, fa_score_node);

  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  std::vector<AnfNodePtr> eod_masks;
  for (int i = 0; i < sp_num; ++i) {
    DynGenerateEodMask(i, pos, sp_num, actual_shape_size, fa_s1_dyn, fa_s2_dyn, actual_node, &eod_masks);
  }
  CNodePtr local_fa_node, kv_received_tuple, softmax_max, softmax_sum, softmax_out, attention_output;
  CNodePtr history_max, history_sum, acc_attention;
  AnfNodePtr actual_mask;
  for (int i = 0; i < sp_num; ++i) {
    std::vector<AnfNodePtr> kv_nodes = {key_node, value_node};
    auto kv_tuple = NewMakeTupleNode(kv_nodes);
    auto kv_concat = NewConcatNode(kv_tuple, 0);
    std::vector<AnfNodePtr> concat_tuple = {kv_concat};
    auto kv_concat_tuple = NewMakeTupleNode(concat_tuple);
    if (i != sp_num - 1) {
      auto neigh_shape = kv_shape;
      if (neigh_shape[0] != -1) {
        neigh_shape[0] = neigh_shape[0] * kIndex2;
      }
      kv_received_tuple = NewNeighborExchangeNode(kv_concat_tuple, {send_rank_id}, {recv_rank_id}, fa_index, i,
                                                  neigh_shape, output_type_id);
    }

    DynSetFAInputs(nullptr, key_node, value_node, attn_node, operator_info, eod_masks, sp_num, i, pos, fa_s1_dyn,
                   fa_s2_dyn, &fa_inputs);

    local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, i, true);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);

    if (i != sp_num - 1) {
      auto kv_exchanged_item = NewTupleGetItemNode(kv_received_tuple, kIndex0);
      auto kv_split = NewSplitNode(kv_exchanged_item, kIndex0, kIndex2);
      key_node = NewTupleGetItemNode(kv_split, kIndex0);
      value_node = NewTupleGetItemNode(kv_split, kIndex1);
    }

    attention_output = NewTupleGetItemNode(local_fa_node, kIndex3);
    softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
    softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);

    if (i == 0) {
      acc_attention = attention_output->cast<CNodePtr>();
      history_max = softmax_max->cast<CNodePtr>();
      history_sum = softmax_sum->cast<CNodePtr>();
    } else {
      DynUpdateAttentionOutput(&history_max, &history_sum, &acc_attention, softmax_max, softmax_sum, attention_output,
                               fa_b_dyn, fa_s1_dyn, fa_n1, fa_h1_dyn, input_layout, fa_index, i);
    }
  }
  acc_attention = NewCastNode(acc_attention, output_type_id);
  softmax_out = NewTupleGetItemNode(local_fa_node, kIndex2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

void CreateCommNodeForRA(AnfNodePtr *query_node, const AnfNodePtr &value_node, const CNodePtr &last_fa_node,
                         const CNodePtr &last_send_node, const CNodePtr &last_recv_node, int64_t pos,
                         int64_t send_rank_id, int64_t recv_rank_id, int fa_index, TypeId output_type_id, size_t step,
                         const Shape &kv_shape, AnfNodePtr *key_node, CNodePtr *send_node, CNodePtr *recv_node) {
  std::vector<AnfNodePtr> kv_nodes = {*key_node, value_node};
  auto kv_tuple = NewMakeTupleNode(kv_nodes);
  auto kv_concat_tuple = NewConcatNode(kv_tuple, 0);
  auto neigh_shape = kv_shape;
  if (neigh_shape[kIndex0] != -1) {
    neigh_shape[kIndex0] = neigh_shape[kIndex0] * kIndex2;
  }
  if (pos % kIndex2 == kIndex0) {
    kv_concat_tuple = CreateDepends(kv_concat_tuple, {last_fa_node, last_recv_node});
    *send_node =
      NewSendNode(kv_concat_tuple, 0, send_rank_id, neigh_shape, output_type_id, g_device_manager->world_group());
    *recv_node =
      NewReceiveNode(*send_node, 0, recv_rank_id, neigh_shape, output_type_id, g_device_manager->world_group());
    *query_node = CreateDepends(*query_node, {*recv_node});
    *query_node = CreateDepends(*query_node, {last_fa_node, last_recv_node});
  } else {
    auto depend_node = CreateDepends(kv_concat_tuple, {last_fa_node, last_send_node});
    *recv_node =
      NewReceiveNode(depend_node, 0, recv_rank_id, neigh_shape, output_type_id, g_device_manager->world_group());
    *send_node = NewSendNode(CreateDepend(kv_concat_tuple, *recv_node), 0, send_rank_id, neigh_shape, output_type_id,
                             g_device_manager->world_group());
    *query_node = CreateDepends(*query_node, {*send_node});
    *query_node = CreateDepends(*query_node, {last_fa_node, last_send_node});
  }
  (*send_node)->AddPrimalAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(GetFlashIndexString(fa_index, step)));
  (*recv_node)->AddPrimalAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(GetFlashIndexString(fa_index, step)));
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(GetFlashIndexString(fa_index, step)),
                               (*send_node));
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(GetFlashIndexString(fa_index, step)),
                               (*recv_node));
  (*send_node)->AddPrimalAttr(RING_ATTENTION_POS, MakeValue<int64_t>(pos));
  (*recv_node)->AddPrimalAttr(RING_ATTENTION_POS, MakeValue<int64_t>(pos));
}

CNodePtr CreateReplaceRingAttentionGraphBySendRecv(const FuncGraphManagerPtr &manager,
                                                   const std::vector<CNodePtr> &origin_nodes_topological,
                                                   const CNodePtr &fa_score_node, FSPInfo *fsp_info, int fa_index) {
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }

  auto query_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex + 1);
  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  auto attn_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex + 1);
  auto actual_node =
    fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputActualSeqQlenIndex + 1);

  int64_t actual_shape_size = fsp_info->actual_seq_length_size();
  size_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  int64_t send_rank_id = fsp_info->GetSendRankId(), recv_rank_id = fsp_info->GetRecvRankId();

  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();

  auto input_layout = flash_score_info_ptr->input_layout();
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(fa_score_node, kIndex3);

  int64_t fa_b, fa_s1, fa_h1, fa_s2, fa_h2, fa_n1;

  GetBSHFromShape(input_layout, q_shape, kv_shape, &fa_b, &fa_s1, &fa_h1, &fa_s2, &fa_h2, &fa_n1, fa_score_node);

  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  std::vector<AnfNodePtr> eod_masks;
  for (int i = 0; i < static_cast<int>(sp_num); ++i) {
    GenerateEodMask(i, pos, static_cast<int>(sp_num), actual_shape_size, {fa_s1, fa_s2}, actual_node, &eod_masks);
  }
  CNodePtr local_fa_node, kv_received_tuple, softmax_max, softmax_sum, softmax_out, attention_output;
  CNodePtr history_max, history_sum, acc_attention, last_fa_node, last_comm_node, send_node, recv_node;
  CNodePtr last_send_node;
  CNodePtr last_recv_node;
  AnfNodePtr actual_mask, first_actual_mask, full_mask;
  first_actual_mask = GetActualMask(0, pos, TypeId::kNumberTypeUInt8, {2048, 2048});
  auto full_mask_tensor = make_mask_tensor(TypeId::kNumberTypeUInt8, {2048, 2048}, 0, true);
  full_mask = NewValueNode(MakeValue(full_mask_tensor));
  for (size_t i = 0; i < sp_num; ++i) {
    if (i > 0) {
      last_recv_node = CreateDepends(last_recv_node, {last_fa_node, last_send_node});
      auto kv_split = NewSplitNode(last_recv_node, kIndex0, kIndex2);
      key_node = NewTupleGetItemNode(kv_split, kIndex0);
      value_node = NewTupleGetItemNode(kv_split, kIndex1);
    }

    if (i != sp_num - kIndex1) {
      CreateCommNodeForRA(&query_node, value_node, last_fa_node, last_send_node, last_recv_node, pos, send_rank_id,
                          recv_rank_id, fa_index, output_type_id, i, kv_shape, &key_node, &send_node, &recv_node);
    }

    SetFAInputs(query_node, key_node, value_node, attn_node, operator_info, eod_masks, sp_num, i, pos,
                Shape{fa_s1, fa_s2}, &fa_inputs, &first_actual_mask, &full_mask);

    local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, i, false);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);
    local_fa_node->AddPrimalAttr("sp_num", MakeValue<int64_t>(sp_num));
    local_fa_node->set_user_data<parallel::OperatorInfo>(operator_info);
    last_fa_node = local_fa_node;
    last_send_node = send_node;
    last_recv_node = recv_node;

    softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
    softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);
    attention_output = NewTupleGetItemNode(local_fa_node, kIndex3);
    if (i == 0) {
      acc_attention = attention_output->cast<CNodePtr>();
      history_max = softmax_max->cast<CNodePtr>();
      history_sum = softmax_sum->cast<CNodePtr>();
    } else {
      UpdateAttentionOutput(&history_max, &history_sum, &acc_attention, softmax_max, softmax_sum, attention_output,
                            fa_b, fa_s1, fa_n1, fa_h1, input_layout, fa_index, i, output_type_id, (i == sp_num - 1));
      last_fa_node = acc_attention;
    }
  }
  acc_attention = CreateDepends(acc_attention, {last_send_node, last_recv_node});
  softmax_out = NewTupleGetItemNode(local_fa_node, kIndex2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_UPDATE_ATTN, MakeValue<int>(fa_index), acc_attention);
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

CNodePtr DynCreateReplaceRingAttentionGraphBySendRecv(const FuncGraphManagerPtr &manager,
                                                      const std::vector<CNodePtr> &origin_nodes_topological,
                                                      const CNodePtr &fa_score_node, FSPInfo *fsp_info, int fa_index) {
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }

  auto query_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex + 1);
  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  auto attn_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex + 1);
  auto actual_node =
    fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputActualSeqQlenIndex + 1);

  int64_t actual_shape_size = fsp_info->actual_seq_length_size();
  size_t sp_num = fsp_info->GetSPNum(), rank_id = fsp_info->GetRankId();
  int64_t send_rank_id = fsp_info->GetSendRankId(), recv_rank_id = fsp_info->GetRecvRankId();

  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();

  CNodePtr q_dynshape_node, kv_dynshape_node;

  q_dynshape_node = NewDynshapeNode(fa_score_node->input(kIndex0 + 1));
  kv_dynshape_node = NewDynshapeNode(fa_score_node->input(kIndex1 + 1));

  auto input_layout = flash_score_info_ptr->input_layout();
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(fa_score_node, kIndex3);

  int64_t fa_n1;
  CNodePtr fa_b_dyn, fa_s1_dyn, fa_h1_dyn, fa_s2_dyn, fa_h2_dyn;

  GetBSHFromDynShape(input_layout, &fa_b_dyn, &fa_s1_dyn, &fa_h1_dyn, &fa_s2_dyn, &fa_h2_dyn, &fa_n1, q_dynshape_node,
                     kv_dynshape_node, fa_score_node);

  auto pos = GetPosInSpDevice(flash_score_info_ptr, rank_id);
  std::vector<AnfNodePtr> eod_masks;
  for (int i = 0; i < static_cast<int>(sp_num); ++i) {
    DynGenerateEodMask(i, pos, static_cast<int>(sp_num), actual_shape_size, fa_s1_dyn, fa_s2_dyn, actual_node,
                       &eod_masks);
  }
  CNodePtr local_fa_node, kv_received_tuple, softmax_max, softmax_sum, softmax_out, attention_output;
  CNodePtr history_max, history_sum, acc_attention, last_fa_node, last_comm_node, send_node, recv_node;
  CNodePtr last_send_node;
  CNodePtr last_recv_node;
  AnfNodePtr actual_mask;
  for (size_t i = 0; i < sp_num; ++i) {
    if (i > 0) {
      last_recv_node = CreateDepends(last_recv_node, {last_fa_node, last_send_node});
      auto kv_split = NewSplitNode(last_recv_node, kIndex0, kIndex2);
      key_node = NewTupleGetItemNode(kv_split, kIndex0);
      value_node = NewTupleGetItemNode(kv_split, kIndex1);
    }

    if (i != sp_num - kIndex1) {
      CreateCommNodeForRA(&query_node, value_node, last_fa_node, last_send_node, last_recv_node, pos, send_rank_id,
                          recv_rank_id, fa_index, output_type_id, i, kv_shape, &key_node, &send_node, &recv_node);
    }

    key_node = CreateDepends(key_node, {last_fa_node, last_send_node, last_recv_node});

    DynSetFAInputs(query_node, key_node, value_node, attn_node, operator_info, eod_masks, sp_num, i, pos, fa_s1_dyn,
                   fa_s2_dyn, &fa_inputs);

    local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, i, false);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, local_fa_node);
    local_fa_node->AddPrimalAttr("sp_num", MakeValue<int64_t>(sp_num));
    local_fa_node->set_user_data<parallel::OperatorInfo>(operator_info);
    last_fa_node = local_fa_node;
    last_send_node = send_node;
    last_recv_node = recv_node;

    softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
    softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);
    attention_output = NewTupleGetItemNode(local_fa_node, kIndex3);
    if (i == 0) {
      acc_attention = attention_output->cast<CNodePtr>();
      history_max = softmax_max->cast<CNodePtr>();
      history_sum = softmax_sum->cast<CNodePtr>();
    } else {
      DynUpdateAttentionOutput(&history_max, &history_sum, &acc_attention, softmax_max, softmax_sum, attention_output,
                               fa_b_dyn, fa_s1_dyn, fa_n1, fa_h1_dyn, input_layout, fa_index, i);
      last_fa_node = acc_attention;
    }
  }
  acc_attention = CreateDepends(acc_attention, {last_send_node, last_recv_node});
  acc_attention = NewCastNode(acc_attention, output_type_id);
  softmax_out = NewTupleGetItemNode(local_fa_node, kIndex2);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

void GetLayoutInfo(LayoutInfo *li, const CNodePtr &fa_score_node) {
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  li->q_shape = operator_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  li->kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto fa_input = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputHeadNumIndex + 1);
  MS_EXCEPTION_IF_NULL(fa_input);
  auto fa_inpute_cnode = fa_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(fa_inpute_cnode);
  li->fa_n1 = GetValue<int64_t>(fa_inpute_cnode->value());
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  li->input_layout = flash_score_info_ptr->input_layout();
  if (li->input_layout == FASInputLayoutMode::BSH) {
    li->fa_b = li->q_shape[kIndex0];
    li->fa_s1 = li->q_shape[kIndex1];
    li->fa_h1 = li->q_shape[kIndex2];
    li->fa_s2 = li->q_shape[kIndex1];
    li->fa_h2 = li->q_shape[kIndex2];
  } else if (li->input_layout == FASInputLayoutMode::BNSD) {
    li->fa_b = li->q_shape[kIndex0];
    li->fa_s1 = li->q_shape[kIndex2];
    li->fa_h1 = li->q_shape[kIndex1] * li->q_shape[kIndex3];
    li->fa_s2 = li->kv_shape[kIndex2];
    li->fa_h2 = li->kv_shape[kIndex1] * li->kv_shape[kIndex3];
  }
  li->output_id = common::AnfAlgo::GetOutputInferDataType(fa_score_node, kIndex3);
}

void CreateP2PCommunication(const std::vector<AnfNodePtr> &orig_qkv, const CNodePtr &last_recv_node, int64_t pos,
                            int64_t send_rank_id, int64_t recv_rank_id, int fa_index, size_t step, const LayoutInfo &li,
                            CNodePtr *send_node, CNodePtr *recv_node) {
  AnfNodePtr send_data;
  if (step == 0) {
    std::vector<AnfNodePtr> kv_nodes = {orig_qkv[kIndex1], orig_qkv[kIndex2]};
    auto kv_tuple = NewMakeTupleNode(kv_nodes);
    send_data = NewConcatNode(kv_tuple, 0);
  } else {
    send_data = last_recv_node;
  }

  auto neigh_shape = li.kv_shape;
  if (neigh_shape[kIndex0] != -1) {
    neigh_shape[kIndex0] = neigh_shape[kIndex0] * kIndex2;
  }
  if (pos % kIndex2 == 0) {
    *send_node = NewSendNode(send_data, 0, send_rank_id, neigh_shape, li.output_id, g_device_manager->world_group());
    *recv_node =
      NewReceiveNode(*send_node, 0, recv_rank_id, neigh_shape, li.output_id, g_device_manager->world_group());
  } else {
    *recv_node = NewReceiveNode(send_data, 0, recv_rank_id, neigh_shape, li.output_id, g_device_manager->world_group());
    *send_node = NewSendNode(CreateDepend(send_data, *recv_node), 0, send_rank_id, neigh_shape, li.output_id,
                             g_device_manager->world_group());
  }
  string str_fa_index = GetFlashIndexString(fa_index, static_cast<int>(step));
  (*send_node)->AddPrimalAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(str_fa_index));
  (*recv_node)->AddPrimalAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(str_fa_index));
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(str_fa_index), (*send_node));
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(str_fa_index), (*recv_node));
  (*send_node)->AddPrimalAttr(RING_ATTENTION_POS, MakeValue<int64_t>(pos));
  (*recv_node)->AddPrimalAttr(RING_ATTENTION_POS, MakeValue<int64_t>(pos));
}

void PrepareQKV(const std::vector<AnfNodePtr> &orig_qkv, const CNodePtr &last_recv_node, size_t i, size_t rank,
                const LayoutInfo &li, std::vector<AnfNodePtr> *inputs) {
  AnfNodePtr cur_q;
  AnfNodePtr cur_k;
  AnfNodePtr cur_v;
  if (i == 0) {
    cur_q = orig_qkv[kIndex0];
    cur_k = orig_qkv[kIndex1];
    cur_v = orig_qkv[kIndex2];
  } else if (i > 0) {
    auto recv_kv_split = NewSplitNode(last_recv_node, 0, kIndex2);
    auto recv_k = NewTupleGetItemNode(recv_kv_split, kIndex0);
    auto recv_v = NewTupleGetItemNode(recv_kv_split, kIndex1);
    auto axis = li.input_layout == FASInputLayoutMode::BSH ? kDim1 : kDim2;
    if (i <= rank) {
      cur_q = orig_qkv[kIndex0];
      auto k_split = NewSplitNode(recv_k, axis, kIndex2);
      auto v_split = NewSplitNode(recv_v, axis, kIndex2);
      cur_k = NewTupleGetItemNode(k_split, kIndex0);
      cur_v = NewTupleGetItemNode(v_split, kIndex0);
    } else {
      auto q_split = NewSplitNode(orig_qkv[kIndex0], axis, kIndex2);
      cur_q = NewTupleGetItemNode(q_split, kIndex1);
      cur_k = recv_k;
      cur_v = recv_v;
    }
  }
  inputs->emplace_back(cur_q);
  inputs->emplace_back(cur_k);
  inputs->emplace_back(cur_v);
}

void CreateFANode(const CNodePtr &fa_score_node, const std::vector<AnfNodePtr> &cur_qkv, const LayoutInfo &li,
                  size_t step, int64_t pos, int fa_index, size_t sp_num, CNodePtr *local_fa_node) {
  std::vector<AnfNodePtr> fa_inputs;
  for (size_t i = 0; i < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; ++i) {
    fa_inputs.push_back(fa_score_node->input(i + 1));
  }

  fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex] = cur_qkv[kIndex0];
  fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex] = cur_qkv[kIndex1];
  fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex] = cur_qkv[kIndex2];
  if (step == 0) {
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputSparseModeIndex] = CreatInt64Imm(kIndex3);
    auto actual_mask = GetActualMask(0, pos, TypeId::kNumberTypeUInt8, {kAttnMaskSize, kAttnMaskSize});
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = actual_mask;
  }

  *local_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_index, static_cast<int>(step), false);
  common::AnfAlgo::CopyNodeAttrs(fa_score_node, *local_fa_node);
  (*local_fa_node)->AddPrimalAttr("sp_num", MakeValue<int64_t>(static_cast<int>(sp_num)));
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  (*local_fa_node)->set_user_data<parallel::OperatorInfo>(operator_info);
}

void AdjustCPDependency(const FuncGraphManagerPtr &manager, const std::vector<AnfNodePtr> &depend_nodes, size_t pos,
                        CNodePtr *local_fa_node, CNodePtr *send_node, CNodePtr *recv_node) {
  if (pos % kIndex2 == 0) {
    auto send_input = (*send_node)->input(1);
    send_input = CreateDepends(send_input, depend_nodes);
    manager->SetEdge(*send_node, 1, send_input);
  } else {
    auto recv_input = (*recv_node)->input(1);
    recv_input = CreateDepends(recv_input, depend_nodes);
    manager->SetEdge(*recv_node, 1, recv_input);
  }

  auto fa_input = (*local_fa_node)->input(1);
  fa_input = CreateDepends(fa_input, depend_nodes);
  fa_input = CreateDepends(fa_input, {*send_node, *recv_node});
  manager->SetEdge(*local_fa_node, 1, fa_input);
}

void UpdateAttentionOutputViaSubgraph(CNodePtr *history_max, CNodePtr *history_sum, CNodePtr *acc_attention,
                                      const CNodePtr &softmax_max, const CNodePtr &softmax_sum,
                                      const CNodePtr &attention_output, const FuncGraphPtr &sub_graph, int64_t fa_b,
                                      int64_t fa_s1, int64_t fa_n1, int64_t fa_h1, int64_t input_layout, int fa_index,
                                      int index, TypeId output_type_id, bool is_last_update = false,
                                      bool need_split = false) {
  CNodePtr split_max_0;
  CNodePtr split_sum_0;
  CNodePtr split_attn_0;
  if (need_split) {
    auto split_max = NewSplitNode(*history_max, kDim2, kIndex2);
    *history_max = NewTupleGetItemNode(split_max, kIndex1);
    auto split_sum = NewSplitNode(*history_sum, kDim2, kIndex2);
    *history_sum = NewTupleGetItemNode(split_sum, kIndex1);
    auto axis = input_layout == FASInputLayoutMode::BSH ? kDim1 : kDim2;
    auto split_attn = NewSplitNode(*acc_attention, axis, kIndex2);
    *acc_attention = NewTupleGetItemNode(split_attn, kIndex1);

    split_max_0 = NewTupleGetItemNode(split_max, kIndex0);
    split_sum_0 = NewTupleGetItemNode(split_sum, kIndex0);
    split_attn_0 = NewTupleGetItemNode(split_attn, kIndex0);
  }
  auto temp_max = NewMaxNode(*history_max, softmax_max);
  auto m_h_sub_temp = NewSubNode(*history_max, temp_max);
  auto m_i_sub_temp = NewSubNode(softmax_max, temp_max);
  auto e_m_h_temp = NewExpNode(m_h_sub_temp);
  auto e_m_i_temp = NewExpNode(m_i_sub_temp);
  auto e_l_h = NewMulNode(e_m_h_temp, *history_sum);
  auto e_l_i = NewMulNode(e_m_i_temp, softmax_sum);
  auto l = NewAddNode(e_l_h, e_l_i);

  std::vector<AnfNodePtr> update_subgraph_inputs = {NewValueNode(sub_graph), e_l_h, l, e_l_i};

  auto func_graph = softmax_max->func_graph();
  auto call_update_node = func_graph->NewCNode(update_subgraph_inputs);

  auto e_m_i_div_concat = NewTupleGetItemNode(call_update_node, 0);
  auto e_m_h_div_concat = NewTupleGetItemNode(call_update_node, 1);

  auto weighted_history = NewMulNode(e_m_h_div_concat, *acc_attention);
  auto weighted_attention = NewMulNode(e_m_i_div_concat, attention_output);
  (*acc_attention) = NewAddNode(weighted_history, weighted_attention);
  if (need_split) {
    auto merge_max = NewMakeTupleNode({split_max_0, temp_max});
    temp_max = NewConcatNode(merge_max, kIndex2);

    auto merge_sum = NewMakeTupleNode({split_sum_0, l});
    l = NewConcatNode(merge_sum, kIndex2);

    auto merge_attn = NewMakeTupleNode({split_attn_0, *acc_attention});
    auto axis = input_layout == FASInputLayoutMode::BSH ? kDim1 : kDim2;
    *acc_attention = NewConcatNode(merge_attn, axis);
  }
  (*history_max) = temp_max;
  *acc_attention = CreateDepend(*acc_attention, *history_max);
  (*history_sum) = l;
  *acc_attention = CreateDepend(*acc_attention, *history_sum);
  common::AnfAlgo::SetNodeAttr(kAttrAccumulatedAttention, MakeValue(1), *acc_attention);
  common::AnfAlgo::SetNodeAttr("FLASH_INDEX", MakeValue<std::string>(GetFlashIndexString(fa_index, index)),
                               *acc_attention);
  if (is_last_update) {
    weighted_attention->AddPrimalAttr(RING_ATTENTION_UPDATE_MUL, MakeValue<int>(fa_index));
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_UPDATE_MAX, MakeValue<int>(fa_index), *history_max);
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_UPDATE_SUM, MakeValue<int>(fa_index), *history_sum);
  }
}

FuncGraphPtr CreateUpdateAttentionOutputSubgraphWithSpilt(int64_t fa_b, int64_t fa_s1, int64_t fa_n1, int64_t fa_h1,
                                                          int64_t input_layout, TypeId output_type_id) {
  auto sub_graph = std::make_shared<FuncGraph>();

  auto param_e_l_h = sub_graph->add_parameter();
  auto param_l = sub_graph->add_parameter();
  auto param_e_l_i = sub_graph->add_parameter();

  auto e_m_h_div = NewDivNode(param_e_l_h, param_l);
  auto e_m_i_div = NewDivNode(param_e_l_i, param_l);
  auto e_m_h_div_split = NewSplitNode(e_m_h_div, 3, 8);
  auto e_m_h_div_item = NewCastNode(NewTupleGetItemNode(e_m_h_div_split, 0), output_type_id);
  if (input_layout == FASInputLayoutMode::BSH) {
    e_m_h_div_item = NewTransposeNode(e_m_h_div_item, parallel::CreateTuple({0, 2, 1, 3}));
  }
  auto e_m_h_div_concat = NewTileNode(e_m_h_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  if (input_layout == FASInputLayoutMode::BSH) {
    auto real_seq = fa_s1 / kSplitNum;
    e_m_h_div_concat = NewReshapeNode(e_m_h_div_concat, {fa_b, real_seq, fa_h1});
  }

  auto e_m_i_div_split = NewSplitNode(e_m_i_div, 3, 8);
  auto e_m_i_div_item = NewCastNode(NewTupleGetItemNode(e_m_i_div_split, 0), output_type_id);
  if (input_layout == FASInputLayoutMode::BSH) {
    e_m_i_div_item = NewTransposeNode(e_m_i_div_item, parallel::CreateTuple({0, 2, 1, 3}));
  }
  auto e_m_i_div_concat = NewTileNode(e_m_i_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  if (input_layout == FASInputLayoutMode::BSH) {
    auto real_seq = fa_s1 / kSplitNum;
    e_m_i_div_concat = NewReshapeNode(e_m_i_div_concat, {fa_b, real_seq, fa_h1});
  }

  std::vector<AnfNodePtr> output_nodes = {e_m_i_div_concat, e_m_h_div_concat};
  auto output_node = NewMakeTupleNode(output_nodes);
  sub_graph->set_output(output_node);
  return sub_graph;
}

FuncGraphPtr CreateUpdateAttentionOutputSubgraphWithoutSpilt(int64_t fa_b, int64_t fa_s1, int64_t fa_n1, int64_t fa_h1,
                                                             int64_t input_layout, TypeId output_type_id) {
  auto sub_graph = std::make_shared<FuncGraph>();

  auto param_e_l_h = sub_graph->add_parameter();
  auto param_l = sub_graph->add_parameter();
  auto param_e_l_i = sub_graph->add_parameter();

  auto e_m_h_div = NewDivNode(param_e_l_h, param_l);
  auto e_m_i_div = NewDivNode(param_e_l_i, param_l);
  auto e_m_h_div_split = NewSplitNode(e_m_h_div, 3, 8);
  auto e_m_h_div_item = NewCastNode(NewTupleGetItemNode(e_m_h_div_split, 0), output_type_id);
  if (input_layout == FASInputLayoutMode::BSH) {
    e_m_h_div_item = NewTransposeNode(e_m_h_div_item, parallel::CreateTuple({0, 2, 1, 3}));
  }
  auto e_m_h_div_concat = NewTileNode(e_m_h_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  if (input_layout == FASInputLayoutMode::BSH) {
    auto real_seq = fa_s1;
    e_m_h_div_concat = NewReshapeNode(e_m_h_div_concat, {fa_b, real_seq, fa_h1});
  }

  auto e_m_i_div_split = NewSplitNode(e_m_i_div, 3, 8);
  auto e_m_i_div_item = NewCastNode(NewTupleGetItemNode(e_m_i_div_split, 0), output_type_id);
  if (input_layout == FASInputLayoutMode::BSH) {
    e_m_i_div_item = NewTransposeNode(e_m_i_div_item, parallel::CreateTuple({0, 2, 1, 3}));
  }
  auto e_m_i_div_concat = NewTileNode(e_m_i_div_item, parallel::CreateTuple({1, 1, 1, fa_h1 / fa_n1}));
  if (input_layout == FASInputLayoutMode::BSH) {
    auto real_seq = fa_s1;
    e_m_i_div_concat = NewReshapeNode(e_m_i_div_concat, {fa_b, real_seq, fa_h1});
  }

  std::vector<AnfNodePtr> output_nodes = {e_m_i_div_concat, e_m_h_div_concat};
  auto output_node = NewMakeTupleNode(output_nodes);
  sub_graph->set_output(output_node);
  return sub_graph;
}

CNodePtr CreateReplaceRingAttentionCP(const FuncGraphManagerPtr &manager,
                                      const std::vector<CNodePtr> &origin_nodes_topological,
                                      const CNodePtr &fa_score_node, FSPInfo *fsp_info, int fa_index) {
  auto query_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputQueryIndex + 1);
  auto key_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  std::vector<AnfNodePtr> orig_qkv = {query_node, key_node, value_node};
  size_t sp_num = fsp_info->GetSPNum();
  size_t rank_id = fsp_info->GetRankId();
  int64_t send_rank_id = fsp_info->GetSendRankId();
  int64_t recv_rank_id = fsp_info->GetRecvRankId();
  LayoutInfo li;
  GetLayoutInfo(&li, fa_score_node);
  std::shared_ptr<OperatorInfo> operator_info = fa_score_node->user_data<parallel::OperatorInfo>();
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto pos = GetPosInSpDevice(flash_score_info_ptr, static_cast<int>(rank_id));
  CNodePtr local_fa_node;
  CNodePtr last_fa_node;
  CNodePtr history_max;
  CNodePtr history_sum;
  CNodePtr acc_attention;
  CNodePtr last_send_node;
  CNodePtr last_recv_node;
  CNodePtr send_node;
  CNodePtr recv_node;
  FuncGraphPtr update_split_required_graph;
  FuncGraphPtr update_split_excluded_graph;

  for (size_t i = 0; i < sp_num; ++i) {
    if (i > 0) {
      last_recv_node = CreateDepends(last_recv_node, {last_fa_node, last_send_node});
    }
    std::vector<AnfNodePtr> cur_qkv;
    PrepareQKV(orig_qkv, last_recv_node, i, pos, li, &cur_qkv);
    if (i < sp_num - 1) {
      CreateP2PCommunication(cur_qkv, last_recv_node, pos, send_rank_id, recv_rank_id, fa_index, i, li, &send_node,
                             &recv_node);
    }
    CreateFANode(fa_score_node, cur_qkv, li, i, pos, fa_index, sp_num, &local_fa_node);
    if (i < sp_num - 1) {
      std::vector<AnfNodePtr> last_nodes = {last_fa_node, last_send_node, last_recv_node};
      AdjustCPDependency(manager, last_nodes, pos, &local_fa_node, &send_node, &recv_node);
    }
    last_fa_node = local_fa_node;
    last_send_node = send_node;
    last_recv_node = recv_node;

    bool last_step = i == sp_num - 1;
    auto softmax_max = NewTupleGetItemNode(local_fa_node, kIndex0);
    auto softmax_sum = NewTupleGetItemNode(local_fa_node, kIndex1);
    auto attention_output = NewTupleGetItemNode(local_fa_node, kIndex3);
    if (i == 0) {
      acc_attention = attention_output->cast<CNodePtr>();
      history_max = softmax_max->cast<CNodePtr>();
      history_sum = softmax_sum->cast<CNodePtr>();
      update_split_required_graph = CreateUpdateAttentionOutputSubgraphWithSpilt(li.fa_b, li.fa_s1, li.fa_n1, li.fa_h1,
                                                                                 li.input_layout, li.output_id);
      update_split_excluded_graph = CreateUpdateAttentionOutputSubgraphWithoutSpilt(
        li.fa_b, li.fa_s1, li.fa_n1, li.fa_h1, li.input_layout, li.output_id);
      update_split_required_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
      update_split_excluded_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    } else if (static_cast<int>(i) <= pos) {
      UpdateAttentionOutputViaSubgraph(&history_max, &history_sum, &acc_attention, softmax_max, softmax_sum,
                                       attention_output, update_split_excluded_graph, li.fa_b, li.fa_s1, li.fa_n1,
                                       li.fa_h1, li.input_layout, fa_index, static_cast<int>(i), li.output_id,
                                       last_step);
      last_fa_node = acc_attention;
    } else {
      UpdateAttentionOutputViaSubgraph(&history_max, &history_sum, &acc_attention, softmax_max, softmax_sum,
                                       attention_output, update_split_required_graph, li.fa_b, li.fa_s1, li.fa_n1,
                                       li.fa_h1, li.input_layout, fa_index, static_cast<int>(i), li.output_id,
                                       last_step, true);
      last_fa_node = acc_attention;
    }
  }
  auto softmax_out = NewTupleGetItemNode(local_fa_node, kIndex2);
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_UPDATE_ATTN, MakeValue<int>(fa_index), acc_attention);
  std::vector<AnfNodePtr> output_tuple = {history_max, history_sum, softmax_out, acc_attention};
  auto attention_results = NewMakeTupleNode(output_tuple);
  return attention_results;
}

void CreateAndReplaceRingAttentionFAScore(const FuncGraphManagerPtr &manager,
                                          const std::vector<CNodePtr> &origin_nodes_topological,
                                          const CNodePtr &fa_score_node, FSPInfo *fsp_info, int i) {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    MS_LOG(ERROR) << "ring attention & flash sp only supports semi parallel mode";
    return;
  }
  auto fa_score_node_prim = GetCNodePrimitive(fa_score_node);
  MS_EXCEPTION_IF_NULL(fa_score_node_prim);
  bool use_send_recv = false;
  if (fa_score_node_prim->HasAttr(parallel::ENABLE_RA_SEND_RECV) &&
      GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RA_SEND_RECV)))) {
    use_send_recv = GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RA_SEND_RECV)));
  }
  bool enable_ra_context_parallel = false;
  if (fa_score_node_prim->HasAttr(parallel::ENABLE_RA_CONTEXT_PARALLEL) &&
      GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RA_CONTEXT_PARALLEL)))) {
    enable_ra_context_parallel = GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RA_CONTEXT_PARALLEL)));
  }
  CNodePtr cnode;
  bool has_dyn_shape = IsDynamicShape(fa_score_node);
  if (enable_ra_context_parallel) {
    if (has_dyn_shape) {
      MS_LOG(EXCEPTION) << "Ring attention cp don't support dynamic shape currently.";
    }
    cnode = CreateReplaceRingAttentionCP(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
  } else if (use_send_recv) {
    if (!has_dyn_shape) {
      cnode = CreateReplaceRingAttentionGraphBySendRecv(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
    } else {
      cnode =
        DynCreateReplaceRingAttentionGraphBySendRecv(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
    }
  } else {
    if (!has_dyn_shape) {
      cnode = CreateReplaceRingAttentionGraphByAllToAllv(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
    } else {
      cnode =
        DynCreateReplaceRingAttentionGraphByAllToAllv(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
    }
  }
  MS_EXCEPTION_IF_NULL(cnode);
  (void)manager->Replace(fa_score_node, cnode);
}

void CreateAndReplaceFlashSPFAScore(const FuncGraphManagerPtr &manager,
                                    const std::vector<CNodePtr> &origin_nodes_topological,
                                    const CNodePtr &fa_score_node, FSPInfo *fsp_info, int i) {
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel) {
    MS_LOG(ERROR) << "ring attention & flash sp only supports semi parallel mode";
    return;
  }
  bool has_dyn_shape = IsDynamicShape(fa_score_node);
  if (!has_dyn_shape) {
    auto cnode = CreateReplaceFlashSPGraph(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
    MS_EXCEPTION_IF_NULL(cnode);
    (void)manager->Replace(fa_score_node, cnode);
  } else {
    auto cnode = DynCreateReplaceFlashSPGraph(manager, origin_nodes_topological, fa_score_node, fsp_info, i);
    MS_EXCEPTION_IF_NULL(cnode);
    (void)manager->Replace(fa_score_node, cnode);
  }
}

bool CheckUserSettings(const FuncGraphPtr &fg, FSPInfo *fsp_info) {
  fsp_info->DisplayInfo();

  int64_t sp_num = SizeToLong(fsp_info->GetSPNum());
  if (sp_num <= 1) {
    MS_LOG(WARNING) << "FSP: To activate the pass, sp num " << sp_num << "should between larger than 1 ";
    return false;
  }
  return true;
}
}  // namespace

bool SetFlashSP(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto ret = func_graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);

  bool fine_grained_interleave = false;
  std::vector<CNodePtr> fa_score_nodes =
    FindFWFlashAttentionScore(manager, origin_nodes_topological, &fine_grained_interleave);
  bool is_changed = false;

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
    if (fine_grained_interleave) {
      MS_LOG(EXCEPTION) << "RingAttention is not supported when fine_grained_interleave is enable.";
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
      CreateAndReplaceFlashSPFAScore(manager, nodes_topological, fa_score_node, &fsp_info, i);
    } else {
      CreateAndReplaceRingAttentionFAScore(manager, nodes_topological, fa_score_node, &fsp_info, i);
    }
    is_changed = true;
  }
  if (is_changed) {
    MS_LOG(WARNING) << "Ring attention will be deprecated in subsequent versions. "
                       "It is recommended to use other sequence parallel methods as a replacement.";
  }
  return is_changed;
}

static bool NeedRASendRecvAttach(const FuncGraphManagerPtr &manager) {
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kAutoParallel && parallel_mode != kSemiAutoParallel) {
    return false;
  }
  return true;
}

static int GetFaIndex(const std::string &ring_attention_index) {
  size_t underscore_pos = ring_attention_index.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Flash_Index ERROR";
  }

  std::string first_number_str = ring_attention_index.substr(0, underscore_pos);
  return std::stoi(first_number_str);
}

static int GetStepIndex(const std::string &ring_attention_index) {
  size_t underscore_pos = ring_attention_index.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Flash_Index ERROR";
  }
  std::string second_number_str = ring_attention_index.substr(underscore_pos + 1);
  return std::stoi(second_number_str);
}

// Get the bprop subgraph node used in the main grad graph, by the FuncGraphPtr node which contains forward node and
// bprop subgraph.
static AnfNodePtr GetDoutGetItemByFuncGraphNode(const AnfNodePtr &node, const NodeUsersMap &node_users_map) {
  auto call_node_users = node_users_map.at(node);
  CNodePtr usr_node;
  for (auto &node_user_pair : call_node_users) {
    auto node_usr = node_user_pair.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(node_usr, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto index = GetTupleGetItemIndex(node_usr);
    if (index == 1) {
      usr_node = node_usr;
    }
  }
  if (usr_node == nullptr || !IsPrimitiveCNode(usr_node, prim::kPrimTupleGetItem)) {
    return nullptr;
  }
  auto get_item_usrs = node_users_map.at(usr_node);
  if (get_item_usrs.size() == 0) {
    return nullptr;
  } else if (get_item_usrs.size() != 1) {
    MS_LOG(WARNING) << "Get Multi grad usrs. Use first.";
  }
  // Get the bprop subgraph node used in the main grad graph.
  auto bprop_node = get_item_usrs.begin()->first;
  auto bprop_users = node_users_map.at(bprop_node);
  if (bprop_users.size() == 0) {
    return nullptr;
  } else {
    for (auto &bprop_user_pair : bprop_users) {
      auto bprop_usr = bprop_user_pair.first->cast<CNodePtr>();
      if (!IsPrimitiveCNode(bprop_usr, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto index = GetTupleGetItemIndex(bprop_usr);
      if (index == 1) {
        return bprop_usr;
      }
    }
  }
  return nullptr;
}

static CNodePtr GetFuncGraphCNodeByForwardGetItem(const CNodePtr &cnode) {
  CNodePtr cnode_input_graph_node;
  if (!IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    MS_LOG(INFO) << "not kPrimTupleGetItem: ";
    return nullptr;
  }
  auto index = GetTupleGetItemIndex(cnode);
  if (index == 0) {
    cnode_input_graph_node = cnode->input(kIndex1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_input_graph_node);
    if (!IsValueNode<FuncGraph>(cnode_input_graph_node->input(kIndex0))) {
      return nullptr;
    }
  }
  return cnode_input_graph_node;
}

static CNodePtr GetForwardCnodeByFuncGraphCNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<FuncGraph>(cnode->input(kIndex0))) {
    return nullptr;
  }
  auto graph = GetValueNode<FuncGraphPtr>(cnode->input(kIndex0));
  MS_EXCEPTION_IF_NULL(graph);
  auto sub_graph_output = graph->output();
  // Find the subgraph output node that matches 'Tuple(forward_op, grad_bprop_subgraph)'
  if (!IsPrimitiveCNode(sub_graph_output, prim::kPrimMakeTuple)) {
    return nullptr;
  }
  auto output_tuple_cnode = sub_graph_output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(output_tuple_cnode);
  auto output_tuple_cnode_input = output_tuple_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(output_tuple_cnode_input);
  return output_tuple_cnode_input->cast<CNodePtr>();
}

static CNodePtr GetLastInputFuncGraphCNode(const CNodePtr &cnode, const int input_index) {
  CNodePtr cnode_input_graph_node = cnode;
  MS_EXCEPTION_IF_NULL(cnode_input_graph_node);
  // input_index decide the q, k, v
  auto cnode_input_q_get = cnode_input_graph_node->input(input_index)->cast<CNodePtr>();
  while (1) {
    cnode_input_graph_node = GetFuncGraphCNodeByForwardGetItem(cnode_input_q_get);
    if (cnode_input_graph_node == nullptr) {
      MS_LOG(EXCEPTION) << "Can not find the input graph cnode for the node: " << cnode->DebugString();
    }
    auto input_forward_cnode = GetForwardCnodeByFuncGraphCNode(cnode_input_graph_node);
    if (input_forward_cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Can not find the input forward node for the node: " << cnode->DebugString();
    }
    if (!IsPrimitiveCNode(input_forward_cnode, prim::kPrimDepend) &&
        !IsPrimitiveCNode(input_forward_cnode, prim::kPrimReceive) &&
        !IsPrimitiveCNode(input_forward_cnode, prim::kPrimSend)) {
      break;
    } else {
      // skip the Depend, Receive and Send, get the last input node.
      cnode_input_q_get = cnode_input_graph_node->input(kIndex1)->cast<CNodePtr>();
    }
  }
  return cnode_input_graph_node;
}

std::vector<AnfNodePtr> GetLastFAInputCNodeVector(const CNodePtr cnode, const std::string &origin_index,
                                                  const NodeUsersMap &node_users_map) {
  std::vector<AnfNodePtr> fa_input_node_vector;
  // q input
  CNodePtr cnode_q_input_graph_node = GetLastInputFuncGraphCNode(cnode, kIndex1);
  MS_EXCEPTION_IF_NULL(cnode_q_input_graph_node);
  // find the getitem node of the fa input bprop graph.
  auto fa_input1_bprop_get_item = GetDoutGetItemByFuncGraphNode(cnode_q_input_graph_node, node_users_map);
  MS_EXCEPTION_IF_NULL(fa_input1_bprop_get_item);
  fa_input_node_vector.push_back(fa_input1_bprop_get_item);
  MS_LOG(INFO) << "Find the FA q input bprop getitem node to attach: " << fa_input1_bprop_get_item->DebugString()
               << ". The RING_ATTENTION_INDEX/FLASH_INDEX:" << origin_index;

  // k input
  CNodePtr cnode_k_input_graph_node = GetLastInputFuncGraphCNode(cnode, kIndex2);
  MS_EXCEPTION_IF_NULL(cnode_k_input_graph_node);
  auto fa_input2_bprop_get_item = GetDoutGetItemByFuncGraphNode(cnode_k_input_graph_node, node_users_map);
  MS_EXCEPTION_IF_NULL(fa_input2_bprop_get_item);
  fa_input_node_vector.push_back(fa_input2_bprop_get_item);
  MS_LOG(INFO) << "Find the FA k input bprop getitem node to attach: " << fa_input2_bprop_get_item->DebugString()
               << ". The RING_ATTENTION_INDEX/FLASH_INDEX:" << origin_index;

  // v input
  CNodePtr cnode_v_input_graph_node = GetLastInputFuncGraphCNode(cnode, kIndex3);
  MS_EXCEPTION_IF_NULL(cnode_v_input_graph_node);
  auto fa_input3_bprop_get_item = GetDoutGetItemByFuncGraphNode(cnode_v_input_graph_node, node_users_map);
  MS_EXCEPTION_IF_NULL(fa_input3_bprop_get_item);
  fa_input_node_vector.push_back(fa_input3_bprop_get_item);
  MS_LOG(INFO) << "Find the FA v input bprop getitem node to attach: " << fa_input3_bprop_get_item->DebugString()
               << ". The RING_ATTENTION_INDEX/FLASH_INDEX:" << origin_index;
  return fa_input_node_vector;
}

bool AttachCommTupleNodeToFA(const std::map<int, std::vector<AnfNodePtr>> &index_make_tuple_input_map,
                             std::map<int, std::vector<AnfNodePtr>> *index_fa_input_bprop_getitem_map,
                             const FuncGraphManagerPtr &manager) {
  for (auto &index_fa_bprop_getitem_vector : *index_fa_input_bprop_getitem_map) {
    auto index = index_fa_bprop_getitem_vector.first;

    auto make_tuple_input_it = index_make_tuple_input_map.find(index);
    if (make_tuple_input_it == index_make_tuple_input_map.end()) {
      continue;
    }
    std::vector<AnfNodePtr> make_tuple_input = make_tuple_input_it->second;
    FuncGraphPtr grad_graph = make_tuple_input[kIndex1]->func_graph();
    MS_EXCEPTION_IF_NULL(grad_graph);
    auto make_tuple = grad_graph->NewCNode(make_tuple_input);
    MS_EXCEPTION_IF_NULL(make_tuple);

    auto fa_bprop_getitem_vector = index_fa_bprop_getitem_vector.second;
    for (auto fa_bprop_getitem : fa_bprop_getitem_vector) {
      if (grad_graph != fa_bprop_getitem->func_graph()) {
        MS_LOG(EXCEPTION) << "Got wrong grad subgraph when attaching RA/FlashSP grad send/recv tp the grad FA inputs.";
      }
      std::vector<AnfNodePtr> attach_node_input = {NewValueNode(prim::kPrimDepend), fa_bprop_getitem, make_tuple};
      auto attach_node = grad_graph->NewCNode(attach_node_input);
      MS_EXCEPTION_IF_NULL(attach_node);
      MS_LOG(INFO) << "Attach_node for RA/FlashSP Send/Recv grad node: " << attach_node->DebugString();
      manager->Replace(fa_bprop_getitem, attach_node);
    }
  }
  return true;
}

void ProcessIndexMakeTupleInputMap(const AnfNodePtr &node, const CNodePtr &forward_cnode,
                                   const NodeUsersMap &node_users_map, const std::string &origin_index,
                                   std::map<int, std::vector<AnfNodePtr>> *index_make_tuple_input_map) {
  MS_LOG(INFO) << "Start to Handle the RA/FlashSP Send/Recv grad attaching for the forward comm node: "
               << forward_cnode->DebugString();
  auto comm_bprop_get_item = GetDoutGetItemByFuncGraphNode(node, node_users_map);
  MS_EXCEPTION_IF_NULL(comm_bprop_get_item);
  if (index_make_tuple_input_map->find(GetFaIndex(origin_index)) == index_make_tuple_input_map->end()) {
    int fa_index = GetFaIndex(origin_index);
    std::vector<AnfNodePtr> new_make_tuple_input = {NewValueNode(prim::kPrimMakeTuple), comm_bprop_get_item};
    index_make_tuple_input_map->insert({fa_index, new_make_tuple_input});
  } else {
    auto &make_tuple_input = index_make_tuple_input_map->at(GetFaIndex(origin_index));
    FuncGraphPtr graph1 = comm_bprop_get_item->func_graph();
    MS_EXCEPTION_IF_NULL(graph1);
    if (graph1 != make_tuple_input[kIndex1]->func_graph()) {
      MS_LOG(EXCEPTION) << "The grad send/recv node does not belong to the same grad subgraph.";
    }
    make_tuple_input.emplace_back(comm_bprop_get_item);
  }
  MS_LOG(INFO) << "Find the comm bprop getitem node to be attached: " << comm_bprop_get_item->DebugString()
               << ", the corresponding forward node: " << forward_cnode->DebugString();
}

bool FlashSPSendRecvNodeAttach(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  if (root->has_flag(FLASH_SP_SEND_RECV_HAS_ATTACHED)) {
    return false;
  }
  root->set_flag(FLASH_SP_SEND_RECV_HAS_ATTACHED, true);
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!NeedRASendRecvAttach(manager)) {
    return false;
  }
  auto ret_after = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = DeepScopedGraphSearch(ret_after);
  const auto &node_users_map = manager->node_users();
  std::map<int, std::vector<AnfNodePtr>> index_fa_input_bprop_getitem_map;
  std::map<int, std::vector<AnfNodePtr>> index_make_tuple_input_map;

  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto forward_cnode = GetForwardCnodeByFuncGraphCNode(cnode);
    if (forward_cnode == nullptr) {
      continue;
    }
    std::string origin_index;
    if (!forward_cnode->HasPrimalAttr(RING_ATTENTION_INDEX) && !forward_cnode->HasPrimalAttr(FLASH_INDEX)) {
      continue;
    }
    if (forward_cnode->HasPrimalAttr(RING_ATTENTION_INDEX)) {
      origin_index = GetValue<std::string>(forward_cnode->GetPrimalAttr(RING_ATTENTION_INDEX));
    } else {
      origin_index = GetValue<std::string>(forward_cnode->GetPrimalAttr(FLASH_INDEX));
    }
    if (IsPrimitiveCNode(forward_cnode, prim::kPrimFlashAttentionScore) && GetStepIndex(origin_index) == 0) {
      MS_LOG(INFO) << "Start to Handle the RA/FlashSP Send/Recv grad attaching for the forward FA node: "
                   << forward_cnode->DebugString();
      int fa_index = GetFaIndex(origin_index);
      // find the last node of fa q, k and v input as the attaching node of send/receive tuple, skipping depend, send
      // and receive.
      std::vector<AnfNodePtr> fa_input_node_vector = GetLastFAInputCNodeVector(cnode, origin_index, node_users_map);

      MS_LOG(INFO) << "Find the FA input bprop getitem node vector to attach, the corresponding forward FA node: "
                   << forward_cnode->DebugString() << ", RING_ATTENTION_INDEX/FLASH_INDEX:" << origin_index;
      if (!fa_input_node_vector.empty()) {
        index_fa_input_bprop_getitem_map.insert({fa_index, fa_input_node_vector});
      }
      continue;
    }
    if (!IsPrimitiveCNode(forward_cnode, prim::kPrimReceive)) {
      continue;
    }
    ProcessIndexMakeTupleInputMap(node, forward_cnode, node_users_map, origin_index, &index_make_tuple_input_map);
  }

  if (index_fa_input_bprop_getitem_map.empty()) {
    MS_LOG(INFO) << "No RA/FlashSP Send/Recv grad is found to be attached.";
    return false;
  }
  MS_LOG(WARNING) << "Ring attention will be deprecated in subsequent versions. "
                     "It is recommended to use other sequence parallel methods as a replacement.";
  return AttachCommTupleNodeToFA(index_make_tuple_input_map, &index_fa_input_bprop_getitem_map, manager);
}
}  // namespace parallel
}  // namespace mindspore
