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

#include <map>
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <queue>
#include <utility>
#include "frontend/parallel/pass/overlap_grad_ring_attention.h"
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "infer/make_tuple.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"
#include "include/utils/parallel_context.h"
#include "include/utils/comm_manager.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/jit/ps/action.h"
#include "mindspore/ccsrc/frontend/parallel/graph_util/generate_graph.h"
#include "include/op_enum.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score_grad.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/flash_attention_score_info.h"
#include "frontend/parallel/pass/flash_sp.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

struct FaGradCompareMethod {
  bool operator()(const std::string &a, const std::string &b) const {
    int a_1, a_2, b_1, b_2;
    std::istringstream a_stream(a);
    std::istringstream b_stream(b);
    char underline;
    a_stream >> a_1 >> underline >> a_2;
    b_stream >> b_1 >> underline >> b_2;

    return a_1 == b_1 ? a_2 < b_2 : a_1 > b_1;
  }
};

namespace mindspore {
namespace parallel {
namespace {
constexpr int kRingStep2 = 2;

struct FAGradInputInfo {
  AnfNodePtr attn_out;
  AnfNodePtr max;
  AnfNodePtr sum;
  AnfNodePtr dout;
};

std::string GetPreviousStr(std::string origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string first_number_str = origin_str.substr(0, underscore_pos);
  int first_number = std::stoi(first_number_str);

  std::string second_number_str = origin_str.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str) + 1;

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

std::string GetNextStr(std::string origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string first_number_str = origin_str.substr(0, underscore_pos);
  int first_number = std::stoi(first_number_str);

  std::string second_number_str = origin_str.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str) - 1;

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

std::string GetFirstStr(std::string origin_str, int64_t sp_num) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string first_number_str = origin_str.substr(0, underscore_pos);
  int64_t first_number = std::stoi(first_number_str);

  int64_t second_number = sp_num - 1;

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

std::string GetLastStr(std::string origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string first_number_str = origin_str.substr(0, underscore_pos);
  int64_t first_number = std::stoi(first_number_str);

  int64_t second_number = 0;

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

int GetSecondNumber(std::string origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string second_number_str = origin_str.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str);
  return second_number;
}

void FindFAGradInputNode(const CNodePtr &node, std::map<int64_t, AnfNodePtr> *attention_out_map,
                         std::map<int64_t, AnfNodePtr> *softmax_max_map, std::map<int64_t, AnfNodePtr> *softmax_sum_map,
                         std::map<int64_t, AnfNodePtr> *dout_map) {
  if (node != nullptr) {
    if (node->HasPrimalAttr(RING_ATTENTION_UPDATE_MUL) && node->HasPrimalAttr("forward_unique_id")) {
      auto flash_index = GetValue<int>(node->GetPrimalAttr(RING_ATTENTION_UPDATE_MUL));
      dout_map->insert({flash_index, node});
    }

    if (common::AnfAlgo::HasNodeAttr(RING_ATTENTION_UPDATE_MAX, node) && !node->HasPrimalAttr("forward_unique_id")) {
      auto flash_index = common::AnfAlgo::GetNodeAttr<int>(node, RING_ATTENTION_UPDATE_MAX);
      if (softmax_max_map->count(flash_index) != 0 && !node->HasAttr("duplicated")) {
        return;
      }
      (*softmax_max_map)[flash_index] = node;
    }

    if (common::AnfAlgo::HasNodeAttr(RING_ATTENTION_UPDATE_SUM, node) && !node->HasPrimalAttr("forward_unique_id")) {
      auto flash_index = common::AnfAlgo::GetNodeAttr<int>(node, RING_ATTENTION_UPDATE_SUM);
      if (softmax_sum_map->count(flash_index) != 0 && !node->HasAttr("duplicated")) {
        return;
      }
      (*softmax_sum_map)[flash_index] = node;
    }

    if (common::AnfAlgo::HasNodeAttr(RING_ATTENTION_UPDATE_ATTN, node) && !node->HasPrimalAttr("forward_unique_id")) {
      auto flash_index = common::AnfAlgo::GetNodeAttr<int>(node, RING_ATTENTION_UPDATE_ATTN);
      if (attention_out_map->count(flash_index) != 0 && !node->HasAttr("duplicated")) {
        return;
      }
      (*attention_out_map)[flash_index] = node;
    }
  }
}

void FindTargetNode(std::vector<AnfNodePtr> *origin_nodes_topological, std::map<std::string, AnfNodePtr> *fwd_fas,
                    std::map<std::string, AnfNodePtr, FaGradCompareMethod> *grad_fa_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_map, std::map<std::string, AnfNodePtr> *grad_send_map,
                    std::map<std::string, AnfNodePtr> *kv_recv_map, CNodePtr *loss_node,
                    std::map<int64_t, AnfNodePtr> *attention_out_map, std::map<int64_t, AnfNodePtr> *softmax_max_map,
                    std::map<int64_t, AnfNodePtr> *softmax_sum_map, std::map<int64_t, AnfNodePtr> *dout_map) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  for (auto &anf_node : *origin_nodes_topological) {
    if (!anf_node->isa<CNode>()) {
      continue;
    }
    auto node = anf_node->cast<CNodePtr>();
    if (node != nullptr && node->HasPrimalAttr(FLASH_LOSS_NODE) && pipeline_stages <= 1) {
      (*loss_node) = node;
    }
    FindFAGradInputNode(node, attention_out_map, softmax_max_map, softmax_sum_map, dout_map);
    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX)) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      fwd_fas->insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_fa_map)[flash_index] = node;
    }

    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_recv_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimReceive) && common::AnfAlgo::HasNodeAttr(RING_ATTENTION_INDEX, node)) {
      auto flash_index = common::AnfAlgo::GetNodeAttr<std::string>(node, RING_ATTENTION_INDEX);
      if (node->HasPrimalAttr("forward_unique_id")) {
        (*grad_send_map).insert({flash_index, node});
      } else if (node->HasAttr("duplicated")) {
        (*kv_recv_map).insert({flash_index + "duplicated", node});
      } else {
        (*kv_recv_map).insert({flash_index, node});
      }
    }
  }
}

void DynFindTargetNode(std::vector<AnfNodePtr> *origin_nodes_topological,
                       std::map<std::string, AnfNodePtr> *grad_fa_map, std::map<std::string, AnfNodePtr> *grad_recv_map,
                       std::map<std::string, AnfNodePtr> *grad_send_map, CNodePtr *loss_node) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  for (auto &anf_node : *origin_nodes_topological) {
    CNodePtr node = anf_node->cast<CNodePtr>();
    if (node != nullptr && node->HasPrimalAttr(FLASH_LOSS_NODE) && pipeline_stages <= 1) {
      (*loss_node) = node;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimReceive) && !IsPrimitiveCNode(node, prim::kPrimSend) &&
        !IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      continue;
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_fa_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_recv_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_send_map).insert({flash_index, node});
    }
  }
}

CNodePtr NewSendNode(const AnfNodePtr &send_data, int64_t tag, int64_t dest_rank, const Shape &send_shape,
                     TypeId type_id, const std::string &group_name, const RankList &rank_list) {
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
  auto long_rank_list = parallel::g_device_manager->FindRankListByHashName(group_name);
  std::vector<uint32_t> group_rank_ids;
  std::transform(long_rank_list.begin(), long_rank_list.end(), std::inserter(group_rank_ids, group_rank_ids.begin()),
                 [](int64_t e) -> uint32_t { return static_cast<uint32_t>(e); });
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue(group_rank_ids), send_node);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(send_data, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(send_data, 0);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, send_node.get());
  send_node->set_scope(send_data->scope());
  MS_EXCEPTION_IF_NULL(send_node->abstract());

  return send_node;
}

CNodePtr NewReceiveNode(const AnfNodePtr &parameter, int64_t tag, int64_t src_rank, const Shape &recv_shape,
                        TypeId type_id, const std::string &group_name, const RankList &rank_list) {
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
  auto long_rank_list = parallel::g_device_manager->FindRankListByHashName(group_name);
  std::vector<uint32_t> group_rank_ids;
  std::transform(long_rank_list.begin(), long_rank_list.end(), std::inserter(group_rank_ids, group_rank_ids.begin()),
                 [](int64_t e) -> uint32_t { return static_cast<uint32_t>(e); });
  common::AnfAlgo::SetNodeAttr(kAttrGroupRankIds, MakeValue(group_rank_ids), recv_node);

  common::AnfAlgo::SetOutputInferTypeAndShape({type_id}, {recv_shape}, recv_node.get());
  recv_node->set_scope(parameter->scope());
  MS_EXCEPTION_IF_NULL(recv_node->abstract());

  return recv_node;
}

CNodePtr NewConcatNode(const AnfNodePtr &input_node, size_t concat_dim) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto concat_dim_node = NewValueNode(MakeValue(static_cast<int64_t>(concat_dim)));
  concat_dim_node->set_abstract(std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(concat_dim)));
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(prim::kPrimConcat), input_node, concat_dim_node};
  auto concat = input_node->func_graph()->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  if (shape[concat_dim] > 0) {
    shape[concat_dim] *= kIndex2;
  }
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, concat.get());
  concat->set_scope(input_node->scope());
  MS_EXCEPTION_IF_NULL(concat->abstract());
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

  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_nodes[0], 0);
  std::vector<TypeId> dtypes(input_nodes.size(), dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(input_nodes[0], 0);
  std::vector<ShapeVector> shapes(input_nodes.size(), shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, make_tuple.get());
  make_tuple->set_scope(input_nodes[0]->scope());
  MS_EXCEPTION_IF_NULL(make_tuple->abstract());
  return make_tuple;
}

CNodePtr NewSplitNode(const AnfNodePtr &split_node, size_t split_dim, size_t split_num) {
  if (split_num == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(split_node);
  auto split_dim_node = NewValueNode<int64_t>(split_dim);
  split_dim_node->set_abstract(std::make_shared<abstract::AbstractScalar>(SizeToLong(split_dim)));
  auto split_num_node = NewValueNode<int64_t>(split_num);
  split_num_node->set_abstract(std::make_shared<abstract::AbstractScalar>(SizeToLong(split_num)));
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(prim::kPrimSplit), split_node, split_dim_node, split_num_node};
  auto split = split_node->func_graph()->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);

  auto dtype = common::AnfAlgo::GetOutputInferDataType(split_node, 0);
  std::vector<TypeId> dtypes(split_num, dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(split_node, 0);
  if (shape[split_dim] > 0) {
    shape[split_dim] /= SizeToLong(split_num);
  }
  std::vector<ShapeVector> shapes(split_num, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());
  split->set_scope(split_node->scope());
  MS_EXCEPTION_IF_NULL(split->abstract());
  return split;
}

AnfNodePtr NewTupleGetItemNode(const AnfNodePtr &input_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto idx = NewValueNode(SizeToLong(output_index));
  MS_EXCEPTION_IF_NULL(idx);
  idx->set_abstract(std::make_shared<abstract::AbstractScalar>(SizeToLong(output_index)));
  auto getitem = input_node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input_node, idx});
  MS_EXCEPTION_IF_NULL(getitem);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, output_index)};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(input_node, output_index)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, getitem.get());
  getitem->set_scope(input_node->scope());
  MS_EXCEPTION_IF_NULL(getitem->abstract());
  return getitem;
}

CNodePtr NewAddNode(const AnfNodePtr &left_node, const AnfNodePtr &right_node) {
  std::vector<AnfNodePtr> add_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimAdd->name())), left_node,
                                        right_node};
  auto add_node = left_node->func_graph()->NewCNode(add_inputs);
  MS_EXCEPTION_IF_NULL(add_node);
  add_node->set_scope(left_node->scope());
  add_node->set_abstract(left_node->abstract()->Clone());
  return add_node;
}

AnfNodePtr CreateDepend(const AnfNodePtr &latter_node, const AnfNodePtr &former_node, const CNodePtr &node) {
  if (former_node == nullptr) {
    return latter_node;
  }
  MS_EXCEPTION_IF_NULL(latter_node);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), latter_node, former_node};
  auto depend_node = node->func_graph()->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_abstract(latter_node->abstract()->Clone());
  return depend_node;
}

void GetPreNode(const std::map<std::string, AnfNodePtr> &grad_recv_map,
                const std::map<std::string, AnfNodePtr> &grad_send_map, const std::string &new_str,
                CNodePtr *pre_grad_recv_node, CNodePtr *pre_grad_send_node) {
  if (grad_recv_map.find(new_str) != grad_recv_map.end() && grad_send_map.find(new_str) != grad_send_map.end()) {
    *pre_grad_recv_node = grad_recv_map.at(new_str)->cast<CNodePtr>();
    *pre_grad_send_node = grad_send_map.at(new_str)->cast<CNodePtr>();
  }
}

bool IsFwdBeginningNode(const std::string &origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string second_number_str = origin_str.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str);
  if (second_number == 0) {
    return true;
  }
  return false;
}

std::shared_ptr<OperatorInfo> GetAttentionInfo(const std::map<std::string, AnfNodePtr> &fa_map) {
  CNodePtr fwd_score;
  for (auto &fa : fa_map) {
    auto fa_score_node_i = fa.second;
    auto fa_score_node_prim = GetCNodePrimitive(fa_score_node_i);
    MS_EXCEPTION_IF_NULL(fa_score_node_prim);
    if (!fa_score_node_prim->HasAttr(parallel::ENABLE_RING_ATTENTION) ||
        !GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RING_ATTENTION)))) {
      continue;
    }
    fwd_score = fa_score_node_i->cast<CNodePtr>();
  }
  MS_EXCEPTION_IF_NULL(fwd_score);
  std::shared_ptr<OperatorInfo> operator_info = fwd_score->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  return operator_info;
}

void GradFirstStepCommKV(const std::string &cur_str, std::map<std::string, AnfNodePtr> *fa_map,
                         const FuncGraphPtr &graph, CNodePtr *grad_fa_node, int64_t pos, int64_t send_rank_id,
                         int64_t recv_rank_id, const Shape &neigh_shape, TypeId output_type_id,
                         std::map<std::string, AnfNodePtr> *grad_send_kv_map,
                         std::map<std::string, AnfNodePtr> *grad_recv_kv_map) {
  auto key_node = (*grad_fa_node)->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = (*grad_fa_node)->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  std::vector<AnfNodePtr> kv_nodes = {key_node, value_node};
  auto kv_tuple = NewMakeTupleNode(kv_nodes);
  auto kv_concat = NewConcatNode(kv_tuple, 0);
  AnfNodePtr send_node, recv_node;
  auto dout = (*grad_fa_node)->input(4);
  auto sp_num = GetValue<int64_t>((*grad_fa_node)->GetPrimalAttr("sp_num"));
  auto first_str = GetFirstStr(cur_str, sp_num);
  auto fwd_last_fa_node = (*fa_map).at(first_str)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(fwd_last_fa_node);
  auto operator_info = fwd_last_fa_node->user_data<parallel::OperatorInfo>();
  auto fa_info = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto rank_list = fa_info->GetSPRankList();
  if (pos % kIndex2 == kIndex0) {
    send_node = NewSendNode(CreateDepend(kv_concat, dout, kv_concat), 0, send_rank_id, neigh_shape, output_type_id,
                            g_device_manager->world_group(), rank_list);
    recv_node = NewReceiveNode(send_node, 0, recv_rank_id, neigh_shape, output_type_id, g_device_manager->world_group(),
                               rank_list);
  } else {
    recv_node = NewReceiveNode(CreateDepend(kv_concat, dout, kv_concat), 0, recv_rank_id, neigh_shape, output_type_id,
                               g_device_manager->world_group(), rank_list);
    send_node = NewSendNode(CreateDepend(kv_concat, recv_node, kv_concat), 0, send_rank_id, neigh_shape, output_type_id,
                            g_device_manager->world_group(), rank_list);
  }
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(cur_str + "grad"), send_node);
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(cur_str + "grad"), recv_node);
  (*grad_send_kv_map).insert({cur_str, send_node});
  (*grad_recv_kv_map).insert({cur_str, recv_node});
  auto grad_fa_node_input = (*grad_fa_node)->input(1);
  grad_fa_node_input = CreateDepend(grad_fa_node_input, send_node, (*grad_fa_node));
  grad_fa_node_input = CreateDepend(grad_fa_node_input, recv_node, (*grad_fa_node));
  auto manager = graph->manager();
  manager->SetEdge((*grad_fa_node), 1, grad_fa_node_input);
}

void GetCommInfo(int64_t *send_rank_id, int64_t *recv_rank_id, std::shared_ptr<OperatorInfo> *operator_info,
                 int64_t *pos, RankList *rank_list) {
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>((*operator_info));

  auto rankList = flash_score_info_ptr->GetSPRankList();
  (*pos) = 0;
  bool is_find = false;
  int64_t dev_rank_id = g_device_manager->global_rank();
  for (size_t i = 0; i < rankList.size(); ++i) {
    if (dev_rank_id == rankList[i]) {
      (*pos) = SizeToLong(i);
      is_find = true;
    }
  }
  if (!is_find) {
    MS_LOG(EXCEPTION) << "RA can not find pos in ranklist";
  }
  (*recv_rank_id) = rankList[((*pos) + 1) % rankList.size()];
  (*send_rank_id) = rankList[((*pos) + rankList.size() - 1) % rankList.size()];
  *rank_list = rankList;
}

void CreateCommForRAGrad(const FuncGraphPtr &graph, const CNodePtr &pre_grad_recv_kv_node,
                         const CNodePtr &pre_grad_send_kv_node, const CNodePtr &last_fa_grad, const CNodePtr &loss_node,
                         int64_t pos, int64_t send_rank_id, int64_t recv_rank_id, const Shape &neigh_shape,
                         TypeId output_type_id, const RankList &rank_list, CNodePtr *grad_fa_node,
                         CNodePtr *grad_recv_node, CNodePtr *grad_send_node, AnfNodePtr *send_node,
                         AnfNodePtr *recv_node) {
  auto manager = graph->manager();
  auto split_input = CreateDepend(pre_grad_recv_kv_node, last_fa_grad, *grad_send_node);
  auto kv_split = NewSplitNode(split_input, kIndex0, kIndex2);
  auto key_node = NewTupleGetItemNode(kv_split, kIndex0);
  auto value_node = NewTupleGetItemNode(kv_split, kIndex1);

  key_node = CreateDepend(key_node, last_fa_grad, *grad_send_node);  // cur fa wait pre fa
  if (pos % kIndex2 == 0) {
    auto kv_concat = CreateDepend(split_input, *grad_recv_node, *grad_send_node);
    *send_node = NewSendNode(CreateDepend(kv_concat, pre_grad_recv_kv_node, *grad_send_node), 0, send_rank_id,
                             neigh_shape, output_type_id, g_device_manager->world_group(), rank_list);
    *recv_node = NewReceiveNode(*send_node, 0, recv_rank_id, neigh_shape, output_type_id,
                                g_device_manager->world_group(), rank_list);
    auto grad_send_input = (*grad_send_node)->input(1);
    grad_send_input = CreateDepend(grad_send_input, pre_grad_recv_kv_node, *grad_send_node);
    manager->SetEdge(*grad_send_node, 1, grad_send_input);

    // ensure grad split which is output of receive(grad send) sort after fa
    manager->Replace(*grad_send_node, CreateDepend(*grad_send_node, *grad_fa_node, *grad_send_node));

    auto grad_recv_input = (*grad_recv_node)->input(1);
    manager->SetEdge(*grad_recv_node, 1, CreateDepend(grad_recv_input, *grad_send_node, *grad_recv_node));

    // ensure fa sort after current grad send/recv
    key_node = CreateDepend(key_node, *grad_recv_node, *grad_recv_node);
    manager->SetEdge(*grad_fa_node, kIndex2, key_node);
    manager->SetEdge(*grad_fa_node, kIndex3, value_node);

    manager->Replace(*grad_fa_node, CreateDepend(*grad_fa_node, *grad_recv_node, *grad_fa_node));
  } else {
    auto kv_concat = CreateDepend(split_input, *grad_send_node, *grad_send_node);
    *recv_node = NewReceiveNode(CreateDepend(kv_concat, pre_grad_send_kv_node, kv_split), 0, recv_rank_id, neigh_shape,
                                output_type_id, g_device_manager->world_group(), rank_list);
    *send_node = NewSendNode(CreateDepend(kv_concat, *recv_node, kv_split), 0, send_rank_id, neigh_shape,
                             output_type_id, g_device_manager->world_group(), rank_list);
    auto grad_recv_input = (*grad_recv_node)->input(1);
    grad_recv_input = CreateDepend(grad_recv_input, pre_grad_send_kv_node, *grad_recv_node);
    manager->SetEdge(*grad_recv_node, 1, grad_recv_input);

    // ensure grad split which is output of receive(grad send) sort after fa
    manager->Replace(*grad_send_node, CreateDepend(*grad_send_node, *grad_fa_node, *grad_send_node));

    auto grad_send_input = (*grad_send_node)->input(1);
    manager->SetEdge(*grad_send_node, 1, CreateDepend(grad_send_input, *grad_recv_node, *grad_send_node));

    // ensure fa sort after current grad send/recv
    key_node = CreateDepend(key_node, *grad_send_node, *grad_send_node);
    manager->SetEdge(*grad_fa_node, kIndex2, key_node);
    manager->SetEdge(*grad_fa_node, kIndex3, value_node);

    manager->Replace(*grad_fa_node, CreateDepend(*grad_fa_node, *grad_send_node, *grad_fa_node));
  }
}
}  // namespace

void DynOverlapGradRingAttention(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  std::map<std::string, AnfNodePtr> grad_fa_map;
  std::map<std::string, AnfNodePtr> grad_send_map;
  std::map<std::string, AnfNodePtr> grad_recv_map;
  auto ret = graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  CNodePtr loss_node;

  DynFindTargetNode(&origin_nodes_topological, &grad_fa_map, &grad_recv_map, &grad_send_map, &loss_node);
  MS_LOG(WARNING) << "Ring attention will be deprecated in subsequent versions. "
                     "It is recommended to use other sequence parallel methods as a replacement.";
  for (auto it = grad_send_map.begin(); it != grad_send_map.end(); ++it) {
    auto grad_fa_node = grad_fa_map.at(it->first)->cast<CNodePtr>();
    auto grad_recv_node = grad_recv_map.at(it->first)->cast<CNodePtr>();
    auto grad_send_node = it->second->cast<CNodePtr>();

    CNodePtr pre_grad_recv_node;
    CNodePtr pre_grad_send_node;
    auto new_str = GetPreviousStr(it->first);
    GetPreNode(grad_recv_map, grad_send_map, new_str, &pre_grad_recv_node, &pre_grad_send_node);
    MS_EXCEPTION_IF_NULL(grad_recv_node);
    auto primal_attr = grad_recv_node->GetPrimalAttr(RING_ATTENTION_POS);
    MS_EXCEPTION_IF_NULL(primal_attr);
    auto grad_recv_pos = GetValue<int64_t>(primal_attr);
    if (grad_recv_pos % kIndex2 == 0) {
      MS_EXCEPTION_IF_NULL(grad_fa_node);
      MS_EXCEPTION_IF_NULL(grad_send_node);
      auto grad_send_input = grad_send_node->input(1);
      MS_EXCEPTION_IF_NULL(grad_send_input);
      for (size_t i = 0; i < grad_fa_node->size(); i++) {
        auto grad_fa_input_node = grad_fa_node->input(i);
        if (grad_fa_input_node != nullptr && grad_fa_input_node->func_graph() != nullptr) {
          grad_send_input = CreateDepend(grad_send_input, grad_fa_input_node, grad_send_node);
        }
      }
      grad_send_input = CreateDepend(grad_send_input, loss_node, grad_send_node);
      manager->SetEdge(grad_send_node, 1, grad_send_input);

      auto grad_fa_node_input = grad_fa_node->input(1);
      for (size_t i = 0; i < grad_send_node->size(); i++) {
        auto grad_send_input_node = grad_send_node->input(i);
        if (grad_send_input_node != nullptr && grad_send_input_node->func_graph() != nullptr) {
          grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_send_input_node, grad_fa_node);
        }
      }
      manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);

      if (pre_grad_recv_node != nullptr) {
        auto grad_send_input_new = grad_send_node->input(1);
        manager->SetEdge(grad_send_node, 1, CreateDepend(grad_send_input_new, pre_grad_recv_node, grad_send_node));
      }

      auto grad_recv_input = grad_recv_node->input(1);
      manager->SetEdge(grad_recv_node, 1, CreateDepend(grad_recv_input, grad_send_node, grad_recv_node));

      manager->Replace(grad_recv_node, CreateDepend(grad_recv_node, grad_fa_node, grad_recv_node));

      std::vector<AnfNodePtr> depend3_inputs{NewValueNode(prim::kPrimDepend), grad_fa_node, grad_recv_node};
      auto depend_node3 = grad_fa_node->func_graph()->NewCNode(depend3_inputs);
      MS_EXCEPTION_IF_NULL(depend_node3);
      depend_node3->set_abstract(grad_fa_node->abstract()->Clone());
      manager->Replace(grad_fa_node, CreateDepend(grad_fa_node, grad_recv_node, grad_fa_node));
    } else {
      MS_EXCEPTION_IF_NULL(grad_fa_node);
      MS_EXCEPTION_IF_NULL(grad_recv_node);
      auto grad_recv_input = grad_recv_node->input(1);
      for (size_t i = 0; i < grad_fa_node->size(); i++) {
        auto grad_fa_input_node = grad_fa_node->input(i);
        if (grad_fa_input_node != nullptr && grad_fa_input_node->func_graph() != nullptr) {
          grad_recv_input = CreateDepend(grad_recv_input, grad_fa_input_node, grad_recv_node);
        }
      }
      grad_recv_input = CreateDepend(grad_recv_input, loss_node, grad_recv_node);
      manager->SetEdge(grad_recv_node, 1, grad_recv_input);

      auto grad_fa_node_input = grad_fa_node->input(1);
      for (size_t i = 0; i < grad_recv_node->size(); i++) {
        auto grad_recv_input_node = grad_recv_node->input(i);
        if (grad_recv_input_node != nullptr && grad_recv_input_node->func_graph() != nullptr) {
          grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_recv_input_node, grad_fa_node);
        }
      }
      manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);

      if (pre_grad_send_node != nullptr) {
        auto grad_recv_input_new = grad_recv_node->input(1);
        manager->SetEdge(grad_recv_node, 1, CreateDepend(grad_recv_input_new, pre_grad_send_node, grad_recv_node));
      }
      MS_EXCEPTION_IF_NULL(grad_send_node);
      auto grad_send_input = grad_send_node->input(1);
      manager->SetEdge(grad_send_node, 1, CreateDepend(grad_send_input, grad_recv_node, grad_send_node));
      manager->Replace(grad_send_node, CreateDepend(grad_send_node, grad_fa_node, grad_send_node));
      manager->Replace(grad_fa_node, CreateDepend(grad_fa_node, grad_send_node, grad_fa_node));
    }
  }
}

bool IsRingAttentionCP(const AnfNodePtr &fa_node) {
  auto fa_score_node_prim = GetCNodePrimitive(fa_node);
  MS_EXCEPTION_IF_NULL(fa_score_node_prim);
  return fa_score_node_prim->HasAttr(parallel::ENABLE_RING_ATTENTION) &&
         GetValue<bool>(fa_score_node_prim->GetAttr(parallel::ENABLE_RING_ATTENTION)) &&
         fa_score_node_prim->HasAttr(parallel::ENABLE_RA_CONTEXT_PARALLEL) &&
         GetValue<bool>(fa_score_node_prim->GetAttr(parallel::ENABLE_RA_CONTEXT_PARALLEL));
}

void CreateParameterWhenLazyInline(const FuncGraphPtr &fwd_graph, const FuncGraphPtr &bck_graph,
                                   const std::vector<AnfNodePtr> &inputs, std::vector<AnfNodePtr> *outputs) {
  auto ret = fwd_graph->get_return();
  auto manager = fwd_graph->manager();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  for (const auto &node : origin_nodes_topological) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto call_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(call_cnode);
    if (call_cnode->inputs().size() <= 1 || !IsValueNode<FuncGraph>(call_cnode->input(1))) {
      continue;
    }
    auto g = GetValueNode<FuncGraphPtr>(call_cnode->input(1));  // partial
    MS_EXCEPTION_IF_NULL(g);
    if (g == bck_graph) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(call_cnode->input(1));
      auto sub_graph_parameters = sub_func_graph->parameters();
      auto new_user_graph_parameters(sub_graph_parameters);

      auto call_inputs(call_cnode->inputs());
      for (const auto &in : inputs) {
        call_inputs.insert(call_inputs.begin() + kIndex2, in);

        auto new_parameter = sub_func_graph->add_parameter();
        new_parameter->set_abstract(in->abstract());
        new_user_graph_parameters.insert(new_user_graph_parameters.begin(), new_parameter);
        outputs->emplace_back(new_parameter);
      }
      auto new_call_cnode = call_cnode->func_graph()->NewCNode(call_inputs);
      MS_EXCEPTION_IF_NULL(new_call_cnode);
      new_call_cnode->set_scope(call_cnode->scope());
      new_call_cnode->set_abstract(call_cnode->abstract()->Clone());
      (void)manager->Replace(call_cnode, new_call_cnode);
      sub_func_graph->set_parameters(new_user_graph_parameters);
    }
  }
}

void ChangeFAGradInput(std::map<std::string, AnfNodePtr, FaGradCompareMethod> *grad_fa_map, const FuncGraphPtr &graph,
                       std::map<int64_t, AnfNodePtr> *attention_out_map, std::map<int64_t, AnfNodePtr> *softmax_max_map,
                       std::map<int64_t, AnfNodePtr> *softmax_sum_map, std::map<int64_t, AnfNodePtr> *dout_map) {
  auto manager = graph->manager();
  for (auto it = (*grad_fa_map).rbegin(); it != (*grad_fa_map).rend(); ++it) {
    auto cur_grad_fa_node = it->second->cast<CNodePtr>();
    size_t underscore_pos = (it->first).find('_');
    std::string first_number_str = (it->first).substr(0, underscore_pos);
    int first_number = std::stoi(first_number_str);

    auto dout_node = dout_map->find(first_number)->second;
    auto softmax_max_node = softmax_max_map->find(first_number)->second;
    auto softmax_sum_node = softmax_sum_map->find(first_number)->second;
    auto attention_out_node = attention_out_map->find(first_number)->second;
    MS_EXCEPTION_IF_NULL(dout_node);
    MS_EXCEPTION_IF_NULL(softmax_max_node);
    MS_EXCEPTION_IF_NULL(softmax_sum_node);
    MS_EXCEPTION_IF_NULL(attention_out_node);
    MS_EXCEPTION_IF_NULL(cur_grad_fa_node);

    auto fwd_graph = softmax_max_node->func_graph();
    auto bck_graph = cur_grad_fa_node->func_graph();
    MS_EXCEPTION_IF_NULL(fwd_graph);
    MS_EXCEPTION_IF_NULL(bck_graph);
    if (fwd_graph != bck_graph) {
      std::vector<AnfNodePtr> inputs = {softmax_max_node, softmax_sum_node, attention_out_node};
      std::vector<AnfNodePtr> outputs;
      CreateParameterWhenLazyInline(fwd_graph, bck_graph, inputs, &outputs);
      if (outputs.size() != inputs.size()) {
        MS_LOG(EXCEPTION) << "The output size is not equal to input size when enable ring attention.";
      }
      softmax_max_node = outputs[kIndex0];
      softmax_sum_node = outputs[kIndex1];
      attention_out_node = outputs[kIndex2];
    }
    MS_EXCEPTION_IF_NULL(dout_node);
    auto dout_cnode = dout_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dout_cnode);
    manager->SetEdge(cur_grad_fa_node, ops::FASGradInputIndex::kFASGradInputDyIndex + kIndex1,
                     dout_cnode->input(kIndex2));
    manager->SetEdge(cur_grad_fa_node, ops::FASGradInputIndex::kFASGradInputSoftmaxMaxIndex + kIndex1,
                     softmax_max_node);
    manager->SetEdge(cur_grad_fa_node, ops::FASGradInputIndex::kFASGradInputSoftmaxSumIndex + kIndex1,
                     softmax_sum_node);
    manager->SetEdge(cur_grad_fa_node, ops::FASGradInputIndex::kFASGradInputSoftmaxOutIndex + kIndex1,
                     cur_grad_fa_node->input(kIndex7));
    manager->SetEdge(cur_grad_fa_node, ops::FASGradInputIndex::kFASGradInputAttentionInIndex + kIndex1,
                     attention_out_node);
  }
}

void PrepareFAGradInput(const FuncGraphPtr &graph,
                        const std::map<std::string, AnfNodePtr, FaGradCompareMethod> &grad_fa_map,
                        const std::map<int64_t, AnfNodePtr> &attention_out_map,
                        const std::map<int64_t, AnfNodePtr> &softmax_max_map,
                        const std::map<int64_t, AnfNodePtr> &softmax_sum_map,
                        const std::map<int64_t, AnfNodePtr> &dout_map, int64_t rank, int64_t input_layout) {
  auto manager = graph->manager();
  AnfNodePtr cur_attn_out;
  AnfNodePtr cur_dout;
  AnfNodePtr cur_max;
  FAGradInputInfo full_info;
  FAGradInputInfo half_info;
  AnfNodePtr cur_sum;
  int64_t step = 0;
  int64_t last_fa_index = -1;
  for (auto it = grad_fa_map.rbegin(); it != grad_fa_map.rend(); ++it, ++step) {
    size_t underscore_pos = it->first.find('_');
    std::string str_fa_index = it->first.substr(0, underscore_pos);
    int fa_index = std::stoi(str_fa_index);
    if (fa_index != last_fa_index) {
      auto dout_element = dout_map.find(fa_index)->second;
      MS_EXCEPTION_IF_NULL(dout_element);
      auto dout_cnode = dout_element->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(dout_cnode);
      auto dout_node = dout_cnode->input(kIndex2);
      MS_EXCEPTION_IF_NULL(dout_node);
      auto filter_func = [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimSplit) || IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem) ||
                      IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimSplitWithSize);
        return std::make_pair(filter, 1);
      };
      full_info.dout = GetInputNodeWithFilter(dout_node, filter_func);
      MS_EXCEPTION_IF_NULL(full_info.dout);
      full_info.max = softmax_max_map.find(fa_index)->second;
      MS_EXCEPTION_IF_NULL(full_info.max);
      full_info.sum = softmax_sum_map.find(fa_index)->second;
      MS_EXCEPTION_IF_NULL(full_info.sum);
      full_info.attn_out = attention_out_map.find(fa_index)->second;
      MS_EXCEPTION_IF_NULL(full_info.attn_out);

      auto split_max = NewSplitNode(full_info.max, kDim2, kIndex2);
      half_info.max = NewTupleGetItemNode(split_max, kIndex1);
      auto split_sum = NewSplitNode(full_info.sum, kDim2, kIndex2);
      half_info.sum = NewTupleGetItemNode(split_sum, kIndex1);
      auto axis = input_layout == ops::FASInputLayoutMode::BSH ? kDim1 : kDim2;
      auto split_attn = NewSplitNode(full_info.attn_out, axis, kIndex2);
      half_info.attn_out = NewTupleGetItemNode(split_attn, kIndex1);
      half_info.dout = dout_node;
      last_fa_index = fa_index;
    }

    auto grad_fa_node = it->second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(grad_fa_node);
    auto sp_num = GetValue<int64_t>(grad_fa_node->GetPrimalAttr("sp_num"));
    step = step % sp_num;
    bool full_q = step >= sp_num - rank - 1;
    cur_attn_out = full_q ? full_info.attn_out : half_info.attn_out;
    cur_dout = full_q ? full_info.dout : half_info.dout;
    cur_max = full_q ? full_info.max : half_info.max;
    cur_sum = full_q ? full_info.sum : half_info.sum;
    MS_EXCEPTION_IF_NULL(cur_attn_out);
    MS_EXCEPTION_IF_NULL(cur_dout);
    MS_EXCEPTION_IF_NULL(cur_max);
    MS_EXCEPTION_IF_NULL(cur_sum);
    auto fwd_graph = cur_sum->func_graph();
    auto bck_graph = grad_fa_node->func_graph();
    if (fwd_graph != bck_graph) {
      std::vector<AnfNodePtr> outputs;
      CreateParameterWhenLazyInline(fwd_graph, bck_graph, {cur_max, cur_sum, cur_attn_out}, &outputs);
      cur_max = outputs[0];
      cur_sum = outputs[1];
      cur_attn_out = outputs[kIndex2];
    }
    manager->SetEdge(grad_fa_node, ops::FASGradInputIndex::kFASGradInputDyIndex + kIndex1, cur_dout);
    manager->SetEdge(grad_fa_node, ops::FASGradInputIndex::kFASGradInputSoftmaxMaxIndex + kIndex1, cur_max);
    manager->SetEdge(grad_fa_node, ops::FASGradInputIndex::kFASGradInputSoftmaxSumIndex + kIndex1, cur_sum);
    auto none_node = NewValueNode(std::make_shared<None>());
    none_node->set_abstract(std::make_shared<abstract::AbstractNone>());
    manager->SetEdge(grad_fa_node, ops::FASGradInputIndex::kFASGradInputSoftmaxOutIndex + kIndex1, none_node);
    manager->SetEdge(grad_fa_node, ops::FASGradInputIndex::kFASGradInputAttentionInIndex + kIndex1, cur_attn_out);
  }
}

void AdjustCommDepForCPGrad(const FuncGraphPtr &graph, const CNodePtr &pre_recv_kv_dkv_node,
                            const CNodePtr &pre_send_kv_dkv_node, const CNodePtr &pre_fa_grad, CNodePtr *grad_fa_node,
                            CNodePtr *dkv_recv_node, CNodePtr *dkv_send_node, int64_t pos) {
  auto manager = graph->manager();
  if (pos % kIndex2 == 0) {
    auto grad_send_input = (*dkv_send_node)->input(1);
    grad_send_input = CreateDepend(grad_send_input, pre_recv_kv_dkv_node, *dkv_send_node);
    grad_send_input = CreateDepend(grad_send_input, pre_fa_grad, *dkv_send_node);
    manager->SetEdge(*dkv_send_node, 1, grad_send_input);

    // ensure grad split which is output of receive(grad send) sort after fa
    manager->Replace(*dkv_send_node, CreateDepend(*dkv_send_node, *grad_fa_node, *dkv_send_node));

    auto grad_recv_input = (*dkv_recv_node)->input(1);
    manager->SetEdge(*dkv_recv_node, 1, CreateDepend(grad_recv_input, *dkv_send_node, *dkv_recv_node));

    // ensure fa sort after current grad send/recv
    auto grad_fa_input = (*grad_fa_node)->input(1);
    grad_fa_input = CreateDepend(grad_fa_input, pre_fa_grad, *grad_fa_node);
    grad_fa_input = CreateDepend(grad_fa_input, *dkv_recv_node, *grad_fa_node);
    grad_fa_input = CreateDepend(grad_fa_input, pre_recv_kv_dkv_node, *grad_fa_node);
    manager->SetEdge(*grad_fa_node, 1, grad_fa_input);

    manager->Replace(*grad_fa_node, CreateDepend(*grad_fa_node, *dkv_recv_node, *grad_fa_node));
  } else {
    auto grad_recv_input = (*dkv_recv_node)->input(1);
    grad_recv_input = CreateDepend(grad_recv_input, pre_send_kv_dkv_node, *dkv_recv_node);
    grad_recv_input = CreateDepend(grad_recv_input, pre_fa_grad, *dkv_recv_node);
    manager->SetEdge(*dkv_recv_node, 1, grad_recv_input);

    // ensure grad split which is output of receive(grad send) sort after fa
    manager->Replace(*dkv_send_node, CreateDepend(*dkv_send_node, *grad_fa_node, *dkv_send_node));

    auto grad_send_input = (*dkv_send_node)->input(1);
    manager->SetEdge(*dkv_send_node, 1, CreateDepend(grad_send_input, *dkv_recv_node, *dkv_send_node));

    // ensure fa sort after current grad send/recv
    auto grad_fa_input = (*grad_fa_node)->input(1);
    grad_fa_input = CreateDepend(grad_fa_input, pre_fa_grad, *grad_fa_node);
    grad_fa_input = CreateDepend(grad_fa_input, *dkv_send_node, *grad_fa_node);
    grad_fa_input = CreateDepend(grad_fa_input, pre_send_kv_dkv_node, *grad_fa_node);
    manager->SetEdge(*grad_fa_node, 1, grad_fa_input);

    manager->Replace(*grad_fa_node, CreateDepend(*grad_fa_node, *dkv_send_node, *grad_fa_node));
  }
}

void CreateCommForFirstStep(const FuncGraphPtr &graph, const std::map<std::string, AnfNodePtr> &fa_map,
                            const std::map<std::string, AnfNodePtr> &kv_recv_map, const string &fa_index,
                            CNodePtr *grad_fa_node, CNodePtr *kv_send_node, CNodePtr *kv_recv_node) {
  auto manager = graph->manager();
  auto first_fwd_fa = fa_map.begin()->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_fwd_fa);
  auto operator_info = first_fwd_fa->user_data<parallel::OperatorInfo>();
  int64_t pos = 0;
  int64_t recv_rank_id = 0;
  int64_t send_rank_id = 0;
  RankList rank_list;
  GetCommInfo(&send_rank_id, &recv_rank_id, &operator_info, &pos, &rank_list);
  MS_EXCEPTION_IF_NULL(operator_info);
  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  MS_EXCEPTION_IF_NULL(grad_fa_node);
  auto neigh_shape = kv_shape;
  if (neigh_shape[kIndex0] != -1) {
    neigh_shape[kIndex0] = neigh_shape[kIndex0] * kIndex2;
  }
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(*grad_fa_node, kIndex3);

  auto dout = (*grad_fa_node)->input(ops::FASGradInputIndex::kFASGradInputDyIndex + kIndex1);
  auto sp_num = GetValue<int64_t>((*grad_fa_node)->GetPrimalAttr("sp_num"));
  auto first_str = GetFirstStr(fa_index, sp_num - 1);  // last send/recv node in fwd graph
  auto kv_concat = kv_recv_map.at(first_str);
  auto duplicated_index = first_str + "duplicated";
  if (kv_recv_map.count(duplicated_index) != 0) {
    kv_concat = kv_recv_map.at(duplicated_index);
  }
  MS_EXCEPTION_IF_NULL(kv_concat);
  AnfNodePtr recv_kv;
  auto fwd_graph = kv_concat->func_graph();
  auto bck_graph = (*grad_fa_node)->func_graph();
  if (fwd_graph == bck_graph) {
    recv_kv = kv_concat;
  } else {
    std::vector<AnfNodePtr> outputs;
    CreateParameterWhenLazyInline(fwd_graph, bck_graph, {kv_concat}, &outputs);
    recv_kv = outputs[0];
  }
  MS_EXCEPTION_IF_NULL(recv_kv);
  recv_kv = CreateDepend(recv_kv, dout, *grad_fa_node);
  if (pos % kIndex2 == 1) {
    *kv_send_node =
      NewSendNode(recv_kv, 0, send_rank_id, neigh_shape, output_type_id, g_device_manager->world_group(), rank_list);
    *kv_recv_node = NewReceiveNode(*kv_send_node, 0, recv_rank_id, neigh_shape, output_type_id,
                                   g_device_manager->world_group(), rank_list);

    auto grad_fa_node_input = (*grad_fa_node)->input(1);
    grad_fa_node_input = CreateDepend(grad_fa_node_input, *kv_recv_node, *grad_fa_node);
    manager->SetEdge(*grad_fa_node, 1, grad_fa_node_input);
  } else {
    *kv_recv_node =
      NewReceiveNode(recv_kv, 0, recv_rank_id, neigh_shape, output_type_id, g_device_manager->world_group(), rank_list);
    *kv_send_node = NewSendNode(CreateDepend(recv_kv, *kv_recv_node, *kv_recv_node), 0, send_rank_id, neigh_shape,
                                output_type_id, g_device_manager->world_group(), rank_list);

    auto grad_fa_node_input = (*grad_fa_node)->input(1);
    grad_fa_node_input = CreateDepend(grad_fa_node_input, *kv_send_node, *grad_fa_node);
    manager->SetEdge(*grad_fa_node, 1, grad_fa_node_input);
  }
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(fa_index + "grad"), *kv_send_node);
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(fa_index + "grad"), *kv_recv_node);
}

void CreateCommForCPGrad(const FuncGraphPtr &graph, const std::map<std::string, AnfNodePtr> &fa_map,
                         const CNodePtr &pre_recv_kv_dkv_node, const CNodePtr &pre_send_kv_dkv_node,
                         const CNodePtr &pre_fa_grad, const string &fa_index, int64_t step, CNodePtr *grad_fa_node,
                         CNodePtr *dkv_recv_node, CNodePtr *dkv_send_node) {
  auto manager = graph->manager();
  auto first_fwd_fa = fa_map.begin()->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_fwd_fa);
  auto operator_info = first_fwd_fa->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto fa_info = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto input_layout = fa_info->input_layout();
  int64_t pos = 0;
  int64_t recv_rank_id = 0;
  int64_t send_rank_id = 0;
  RankList rank_list;
  GetCommInfo(&send_rank_id, &recv_rank_id, &operator_info, &pos, &rank_list);
  auto sp_num = GetValue<int64_t>((*grad_fa_node)->GetPrimalAttr("sp_num"));

  auto split_input = CreateDepend(pre_send_kv_dkv_node, pre_fa_grad, pre_send_kv_dkv_node);
  AnfNodePtr kv_node;
  AnfNodePtr kv_split;
  if (step == 1) {  // pre step just receive kv
    kv_node = pre_send_kv_dkv_node;
    kv_split = NewSplitNode(split_input, kDim0, kIndex2);
  } else {  // pre step receive kv and dkv
    auto kv_dkv_split = NewSplitNode(split_input, kDim0, kIndex2);
    kv_node = NewTupleGetItemNode(kv_dkv_split, kIndex0);
    kv_split = NewSplitNode(kv_node, kDim0, kIndex2);
  }
  auto key_node = NewTupleGetItemNode(kv_split, kIndex0);
  auto value_node = NewTupleGetItemNode(kv_split, kIndex1);
  if (step >= sp_num - pos - 1 && step != sp_num - 1) {  // split kv when step>sp_num-pos-1 or last step not casual
    auto axis = input_layout == ops::FASInputLayoutMode::BSH ? kDim1 : kDim2;
    auto split_k = NewSplitNode(key_node, axis, kIndex2);
    key_node = NewTupleGetItemNode(split_k, kIndex0);
    auto split_v = NewSplitNode(value_node, axis, kIndex2);
    value_node = NewTupleGetItemNode(split_v, kIndex0);
  }
  manager->SetEdge(*grad_fa_node, kIndex2, key_node);
  manager->SetEdge(*grad_fa_node, kIndex3, value_node);

  if (step != sp_num - 1) {  // merge kv and dkv, send tuple of kv and dkv if not last step
    auto dkv_recv_node_input = (*dkv_recv_node)->input(1);
    auto kv_dkv_tuple = NewMakeTupleNode({kv_node, dkv_recv_node_input});
    auto kv_dkv_concat = NewConcatNode(kv_dkv_tuple, 0);
    manager->SetEdge(*dkv_recv_node, 1, kv_dkv_concat);

    // update output shape infer
    MS_EXCEPTION_IF_NULL(operator_info);
    auto neigh_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
    auto output_type_id = common::AnfAlgo::GetOutputInferDataType(*grad_fa_node, kIndex3);
    neigh_shape[kIndex0] = neigh_shape[kIndex0] * kIndex4;
    common::AnfAlgo::SetOutputInferTypeAndShape({output_type_id}, {neigh_shape}, (*dkv_send_node).get());
    common::AnfAlgo::SetOutputInferTypeAndShape({output_type_id}, {neigh_shape}, (*dkv_recv_node).get());

    // replace input of receive output node
    auto kv_dkv_split = NewSplitNode(*dkv_send_node, kDim0, kIndex2);
    auto dkv_node = NewTupleGetItemNode(kv_dkv_split, kIndex1);  // receive dkv
    auto node_users_map = manager->node_users()[*dkv_send_node];
    for (auto &it : node_users_map) {
      auto user_node = it.first->cast<CNodePtr>();
      manager->SetEdge(user_node, it.second, dkv_node);
    }
  }
  AdjustCommDepForCPGrad(graph, pre_recv_kv_dkv_node, pre_send_kv_dkv_node, pre_fa_grad, grad_fa_node, dkv_recv_node,
                         dkv_send_node, pos);
}

void ReplaceDqAdd(const FuncGraphPtr &graph,
                  const std::map<std::string, AnfNodePtr, FaGradCompareMethod> &grad_fa_map) {
  auto manager = graph->manager();
  CNodePtr make_tuple_node;
  CNodePtr tmp_node;
  for (auto it = grad_fa_map.rbegin(); it != grad_fa_map.rend(); ++it) {
    auto grad_fa_node = it->second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(grad_fa_node);
    auto sp_num = GetValue<int64_t>(grad_fa_node->GetPrimalAttr("sp_num"));
    auto last_str = GetLastStr(GetValue<std::string>(grad_fa_node->GetPrimalAttr(RING_ATTENTION_INDEX)));
    auto last_fa_grad_node = grad_fa_map.at(last_str);
    auto last_fa_grad_node_users_map = manager->node_users()[last_fa_grad_node];
    for (auto &last_fa_grad_node_it : last_fa_grad_node_users_map) {
      auto user_node = last_fa_grad_node_it.first->cast<CNodePtr>();
      if (IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
        auto value_node = user_node->input(kIndex2)->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        if (GetValue<int64_t>(value_node->value()) == 0) {
          auto tuple_get_item_node_users_map = manager->node_users()[user_node];
          for (auto &maketuple_it : tuple_get_item_node_users_map) {
            make_tuple_node = maketuple_it.first->cast<CNodePtr>();
          }
        }
      }
    }
    if (make_tuple_node == nullptr) {
      return;
    }
    int second_number = GetSecondNumber(GetValue<std::string>(grad_fa_node->GetPrimalAttr(RING_ATTENTION_INDEX)));
    if (second_number < sp_num - 1) {
      if (second_number == sp_num - kRingStep2) {
        tmp_node =
          NewAddNode(make_tuple_node->input(second_number + 1), make_tuple_node->input(second_number + kRingStep2));
      } else {
        tmp_node = NewAddNode(make_tuple_node->input(second_number + 1), tmp_node);
      }
      if (second_number == 0) {
        auto make_tuple_node_users_map = manager->node_users()[make_tuple_node];
        for (auto &addn_it : make_tuple_node_users_map) {
          auto addn_node = addn_it.first->cast<CNodePtr>();
          if (IsPrimitiveCNode(addn_node, prim::kPrimAddN)) {
            manager->Replace(addn_node, tmp_node);
          }
        }
      } else {
        auto next_str = GetNextStr(GetValue<std::string>(grad_fa_node->GetPrimalAttr(RING_ATTENTION_INDEX)));
        auto next_grad_fa_node = grad_fa_map.at(next_str)->cast<CNodePtr>();
        auto next_grad_fa_input = next_grad_fa_node->input(1);
        next_grad_fa_input = CreateDepend(next_grad_fa_input, tmp_node, next_grad_fa_node);
        manager->SetEdge(next_grad_fa_node, 1, next_grad_fa_input);
      }
    }
  }
}

bool OverlapGradRingAttentionCP(
  const FuncGraphPtr &graph, const std::map<std::string, AnfNodePtr, FaGradCompareMethod> &grad_fa_map,
  const std::map<std::string, AnfNodePtr> &fa_map, const std::map<std::string, AnfNodePtr> &grad_send_map,
  const std::map<std::string, AnfNodePtr> &grad_recv_map, const std::map<std::string, AnfNodePtr> &kv_recv_map,
  const std::map<int64_t, AnfNodePtr> &attention_out_map, const std::map<int64_t, AnfNodePtr> &softmax_max_map,
  const std::map<int64_t, AnfNodePtr> &softmax_sum_map, const std::map<int64_t, AnfNodePtr> &dout_map) {
  auto first_fwd_fa = fa_map.begin()->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_fwd_fa);
  auto operator_info = first_fwd_fa->user_data<parallel::OperatorInfo>();
  auto fa_info = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  auto input_layout = fa_info->input_layout();
  int64_t pos;
  int64_t recv_rank_id;
  int64_t send_rank_id;
  RankList rank_list;
  GetCommInfo(&send_rank_id, &recv_rank_id, &operator_info, &pos, &rank_list);
  PrepareFAGradInput(graph, grad_fa_map, attention_out_map, softmax_max_map, softmax_sum_map, dout_map, pos,
                     input_layout);

  int64_t step = 0;
  CNodePtr pre_recv_kv_dkv_node;
  CNodePtr pre_send_kv_dkv_node;
  CNodePtr pre_fa_grad;

  ReplaceDqAdd(graph, grad_fa_map);
  for (auto it = grad_fa_map.rbegin(); it != grad_fa_map.rend(); ++it, ++step) {
    auto grad_fa_node = it->second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(grad_fa_node);
    auto sp_num = GetValue<int64_t>(grad_fa_node->GetPrimalAttr("sp_num"));
    step = step % sp_num;
    CNodePtr dkv_recv_node;
    CNodePtr dkv_send_node;
    CNodePtr kv_send_node;
    CNodePtr kv_recv_node;
    if (grad_recv_map.count(it->first) != 0) {
      dkv_recv_node = grad_recv_map.at(it->first)->cast<CNodePtr>();
    }
    if (grad_send_map.count(it->first) != 0) {
      dkv_send_node = grad_send_map.at(it->first)->cast<CNodePtr>();
    }
    if (step == 0) {
      pre_fa_grad = nullptr;
      pre_recv_kv_dkv_node = nullptr;
      pre_send_kv_dkv_node = nullptr;
      CreateCommForFirstStep(graph, fa_map, kv_recv_map, it->first, &grad_fa_node, &kv_send_node, &kv_recv_node);
      pre_recv_kv_dkv_node = kv_send_node;
      pre_send_kv_dkv_node = kv_recv_node;
    } else {
      CreateCommForCPGrad(graph, fa_map, pre_recv_kv_dkv_node, pre_send_kv_dkv_node, pre_fa_grad, it->first, step,
                          &grad_fa_node, &dkv_recv_node, &dkv_send_node);
      pre_recv_kv_dkv_node = dkv_recv_node;
      pre_send_kv_dkv_node = dkv_send_node;
    }
    pre_fa_grad = grad_fa_node;
  }
  return true;
}

bool StaticOverlapGradRingAttention(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  std::map<std::string, AnfNodePtr> fa_map, grad_send_map, grad_recv_map;
  std::map<std::string, AnfNodePtr, FaGradCompareMethod> grad_fa_map;
  std::map<int64_t, AnfNodePtr> attention_out_map, softmax_max_map, softmax_sum_map, dout_map;
  std::map<std::string, AnfNodePtr> kv_recv_map;
  auto ret = graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  CNodePtr loss_node;

  FindTargetNode(&origin_nodes_topological, &fa_map, &grad_fa_map, &grad_recv_map, &grad_send_map, &kv_recv_map,
                 &loss_node, &attention_out_map, &softmax_max_map, &softmax_sum_map, &dout_map);
  if (grad_fa_map.empty() || fa_map.empty() || grad_fa_map.size() != fa_map.size() || grad_recv_map.empty() ||
      grad_send_map.empty()) {
    return false;
  }
  MS_LOG(WARNING) << "Ring attention will be deprecated in subsequent versions. "
                     "It is recommended to use other sequence parallel methods as a replacement.";
  if (IsRingAttentionCP(fa_map.begin()->second)) {
    return OverlapGradRingAttentionCP(graph, grad_fa_map, fa_map, grad_send_map, grad_recv_map, kv_recv_map,
                                      attention_out_map, softmax_max_map, softmax_sum_map, dout_map);
  }

  auto operator_info = GetAttentionInfo(fa_map);
  int64_t pos, recv_rank_id, send_rank_id;
  RankList rank_list;
  GetCommInfo(&send_rank_id, &recv_rank_id, &operator_info, &pos, &rank_list);
  std::map<std::string, AnfNodePtr> grad_send_kv_map, grad_recv_kv_map;

  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto neigh_shape = kv_shape;
  if (neigh_shape[kIndex0] != -1) {
    neigh_shape[kIndex0] = neigh_shape[kIndex0] * kIndex2;
  }

  ChangeFAGradInput(&grad_fa_map, graph, &attention_out_map, &softmax_max_map, &softmax_sum_map, &dout_map);

  ReplaceDqAdd(graph, grad_fa_map);
  for (auto it = grad_fa_map.rbegin(); it != grad_fa_map.rend(); ++it) {
    auto grad_fa_node = it->second->cast<CNodePtr>();
    auto output_type_id = common::AnfAlgo::GetOutputInferDataType(grad_fa_node, kIndex3);
    auto new_str = GetPreviousStr(it->first);

    CNodePtr grad_recv_node;
    CNodePtr grad_send_node;
    if (grad_recv_map.find(it->first) != grad_recv_map.end() && grad_send_map.find(it->first) != grad_send_map.end()) {
      grad_recv_node = grad_recv_map.at(it->first)->cast<CNodePtr>();
      grad_send_node = grad_send_map.at(it->first)->cast<CNodePtr>();
    }
    if (grad_recv_node == nullptr || grad_send_node == nullptr) {
      GradFirstStepCommKV(it->first, &fa_map, graph, &grad_fa_node, pos, send_rank_id, recv_rank_id, neigh_shape,
                          output_type_id, &grad_send_kv_map, &grad_recv_kv_map);
      continue;
    }

    CNodePtr pre_grad_recv_kv_node;
    CNodePtr pre_grad_send_kv_node;
    GetPreNode(grad_recv_kv_map, grad_send_kv_map, new_str, &pre_grad_recv_kv_node, &pre_grad_send_kv_node);
    auto pre_fa_grad = grad_fa_map.at(new_str)->cast<CNodePtr>();
    AnfNodePtr send_node;
    AnfNodePtr recv_node;
    CreateCommForRAGrad(graph, pre_grad_recv_kv_node, pre_grad_send_kv_node, pre_fa_grad, loss_node, pos, send_rank_id,
                        recv_rank_id, neigh_shape, output_type_id, rank_list, &grad_fa_node, &grad_recv_node,
                        &grad_send_node, &send_node, &recv_node);
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(it->first + "grad"), send_node);
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(it->first + "grad"), recv_node);
    if (!IsFwdBeginningNode(it->first)) {
      grad_send_kv_map.insert({it->first, send_node});
      grad_recv_kv_map.insert({it->first, recv_node});
    }
  }
  return true;
}

bool OverlapGradRingAttention(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return false;
  }
  if (parallel::IsForwardDynamicShape()) {
    DynOverlapGradRingAttention(graph);
    return false;
  }
  return StaticOverlapGradRingAttention(graph);
}
}  // namespace parallel
}  // namespace mindspore
