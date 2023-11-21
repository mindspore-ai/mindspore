/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/came_parallel_handler.h"

#include <deque>
#include <algorithm>

#include "frontend/parallel/parameter_manager.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/hash_map.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/node_check.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "pipeline/jit/ps/pipeline.h"
#include "mindspore/core/utils/parallel_node_check.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/nn_ops.h"

namespace mindspore {
namespace parallel {
const std::string GetCNodeOpName(const CNodePtr &cnode) {
  // get the prim name of cnode
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(node_prim);
  return node_prim->name();
}

std::pair<bool, const CNodePtr> BackwardSearchCNode(const CNodePtr &bottom_node,
                                                    const std::vector<std::pair<std::string, size_t>> &bwd_calls,
                                                    const std::string &target_name) {
  CNodePtr target_node = bottom_node;
  for (const auto &call_param : bwd_calls) {
    const auto node_name = call_param.first;
    const auto idx = call_param.second;
    auto cnode_name = GetCNodeOpName(target_node);
    if (cnode_name != node_name) {
      MS_LOG(WARNING) << "[CAME] backward search failed, expect node name: " << node_name << " but got " << cnode_name;
      return {false, bottom_node};
    }
    const auto &param_node = target_node->input(idx + 1);
    if (!param_node) {
      MS_LOG(WARNING) << "[CAME] backward search failed, expect param at index: " << (idx + 1) << " but got null";
      return {false, bottom_node};
    }
    if (!param_node->isa<CNode>()) {
      MS_LOG(WARNING) << "[CAME] param node is not a cnode!";
      return {false, bottom_node};
    }
    auto param_cnode = param_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(param_cnode);
    target_node = param_cnode;
  }
  auto cnode_name = GetCNodeOpName(target_node);
  if (cnode_name != target_name) {
    MS_LOG(WARNING) << "[CAME] backward search failed, expect target node name: " << target_name << " but got "
                    << cnode_name;
    return {false, bottom_node};
  }
  return {true, target_node};
}

std::pair<bool, std::vector<CNodePtr>> ForwardSearchCNode(const CNodePtr &start_node,
                                                          const std::vector<std::string> &fwd_calls,
                                                          const NodeUsersMap &node_user_map) {
  if (!start_node) {
    MS_LOG(WARNING) << "[CAME] forward search start is null!";
    return {false, {}};
  }
  if (fwd_calls.empty()) {
    MS_LOG(WARNING) << "[CAME] gives empty forward calls!";
    return {false, {}};
  }
  std::vector<CNodePtr> candidates;
  std::deque<CNodePtr> visited;
  CNodePtr cur_node = nullptr;
  uint32_t depth = 0;

  visited.push_back(start_node);
  CNodePtr last_node = visited.back();
  while (!visited.empty()) {
    if (depth == fwd_calls.size() - 1) {
      std::copy(visited.begin(), visited.end(), std::back_inserter(candidates));
      break;
    }
    cur_node = visited.front();
    MS_LOG(INFO) << "[CAME] fwd current node: " << cur_node->DebugString();
    visited.pop_front();
    auto node_set = node_user_map.at(cur_node->cast<AnfNodePtr>());
    for (auto item : node_set) {
      auto user_node = item.first;
      if (!user_node->isa<CNode>()) {
        continue;
      }
      auto user_cnode = user_node->cast<CNodePtr>();
      if (GetCNodeOpName(user_cnode) == fwd_calls[depth + 1]) {
        visited.push_back(user_cnode);
      }
    }
    if (last_node == cur_node) {
      last_node = visited.back();
      depth++;
    }
  }

  if (candidates.empty()) {
    return {false, {}};
  } else {
    return {true, candidates};
  }
}

CameCommHandler::CameCommHandler(ParameterPtr origin, const std::vector<AnfNodePtr> &all_parameters,
                                 const NodeUsersMap &node_user_map)
    : origin(origin), all_parameters(all_parameters), node_user_map(node_user_map) {
  CheckGlobalDeviceManager();
  cur_rank = g_device_manager->global_rank();
  full_rank_list = g_device_manager->GetDeviceListInThisStage();

  tensor_layout = origin->user_data<TensorLayout>();
  MS_EXCEPTION_IF_NULL(tensor_layout);

  auto opt_shard_group_name = tensor_layout->opt_shard_group();
  if (!opt_shard_group_name.empty()) {
    is_opt_shard = true;
  }
  MS_LOG(DEBUG) << "CAME processing parameter";
  MS_LOG(DEBUG) << "tensor shape:" << tensor_layout->tensor_shape().ToString();
  MS_LOG(DEBUG) << "slice shape:" << tensor_layout->slice_shape().ToString();

  MS_LOG(DEBUG) << "opt shard slice shape:";
  for (const auto &item : tensor_layout->opt_shard_slice_shape()) {
    MS_LOG(DEBUG) << item;
  }
  MS_LOG(DEBUG) << "opt shard group:" << tensor_layout->opt_shard_group();
  MS_LOG(DEBUG) << "opt shard step:" << tensor_layout->opt_weight_shard_step();

  MS_LOG(DEBUG) << "device arrangement:" << tensor_layout->device_arrangement().ToString();
  MS_LOG(DEBUG) << "original device arrangement:" << tensor_layout->device_arrangement_origin().ToString();

  MS_LOG(DEBUG) << "tensor map:" << tensor_layout->tensor_map().ToString();
  MS_LOG(DEBUG) << "original tensor map:" << tensor_layout->origin_tensor_map().ToString();

  FindCameParams();
}

void CameCommHandler::FindCameParams() {
  const std::string origin_name = origin->name();
  const std::string exp_row_name = EXP_AVG_SQ_ROW + origin_name;
  const std::string exp_col_name = EXP_AVG_SQ_COL + origin_name;
  const std::string exp_insta_row_name = EXP_AVG_INSTA_ROW + origin_name;
  const std::string exp_insta_col_name = EXP_AVG_INSTA_COL + origin_name;
  const std::string exp_avg_name = std::string(EXP_AVG) + "." + origin_name;
  const size_t param_to_find_size = 5;
  size_t cur_found_param_count = 0;
  for (const auto &param_node : all_parameters) {
    auto param = param_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    const std::string param_name = param->name();
    if (param_name == exp_row_name) {
      MS_LOG(DEBUG) << "[CAME] found exp_avg_sq_row: " << param_name;
      exp_avg_sq_row = param;
      cur_found_param_count++;
    } else if (param_name == exp_col_name) {
      MS_LOG(DEBUG) << "[CAME] found exp_avg_sq_col: " << param_name;
      exp_avg_sq_col = param;
      cur_found_param_count++;
    } else if (param_name == exp_insta_row_name) {
      MS_LOG(DEBUG) << "[CAME] found exp_avg_insta_row: " << param_name;
      exp_avg_insta_row = param;
      cur_found_param_count++;
    } else if (param_name == exp_insta_col_name) {
      MS_LOG(DEBUG) << "[CAME] found exp_avg_insta_col: " << param_name;
      exp_avg_insta_col = param;
      cur_found_param_count++;
    } else if (param_name == exp_avg_name) {
      MS_LOG(DEBUG) << "[CAME] found exp_avg: " << param_name;
      exp_avg = param;
      cur_found_param_count++;
    }

    if (cur_found_param_count == param_to_find_size) {
      break;
    }
  }
  MS_LOG(INFO) << "[CAME] found params corresponding to origin param size: " << cur_found_param_count;
}

std::pair<Status, RankList> CameCommHandler::GetOptShardRankList(const int64_t rank) {
  DeviceMatrix temp_dev_matrix(rank, full_rank_list, tensor_layout->device_arrangement().array());
  RankList group_devices;
  Shape orig_tensor_map = tensor_layout->tensor_map().array();
  if (temp_dev_matrix.GetDevicesByTensorMap(orig_tensor_map, &group_devices) != SUCCESS) {
    return {FAILED, {}};
  }

  int64_t optimizer_weight_shard_size = ParallelContext::GetInstance()->optimizer_weight_shard_size();
  MS_EXCEPTION_IF_ZERO("optimizer_weight_shard_size", optimizer_weight_shard_size);

  int64_t index = std::find(group_devices.begin(), group_devices.end(), rank) - group_devices.begin();

  // eg: optimizer_weight_shard_size = 2, [0, 8, 16, 24] -> [0, 8], [16, 24]
  auto rank_list =
    RankList(group_devices.begin() + index / optimizer_weight_shard_size * optimizer_weight_shard_size,
             group_devices.begin() + (index / optimizer_weight_shard_size + 1) * optimizer_weight_shard_size);
  return std::make_pair(SUCCESS, rank_list);
}

std::pair<Status, RankList> CameCommHandler::GetDimRankList(const int64_t rank, const int64_t dim) {
  DeviceMatrix dev_matrix(rank, full_rank_list, tensor_layout->device_arrangement().array());
  int64_t device_reverse_dim = tensor_layout->tensor_map().GetDimByIdx(dim);
  if (device_reverse_dim == -1) {
    return {SUCCESS, {rank}};
  }
  int64_t device_dim = tensor_layout->device_arrangement().array().size() - 1 - device_reverse_dim;
  RankList rank_list;
  if (dev_matrix.GetDevicesAlongDim(LongToUlong(device_dim), &rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "Get devices along dim failed";
    return {FAILED, rank_list};
  }
  return {SUCCESS, rank_list};
}

RankList CameCommHandler::ExpandRankListWithOptShard(const RankList &rank_list) {
  if (!is_opt_shard) {
    return rank_list;
  }
  MS_LOG(INFO) << "opt shard yes, group name:" << tensor_layout->opt_shard_group();

  RankList opt_rank_list_find = g_device_manager->FindRankListByHashName(tensor_layout->opt_shard_group());
  for (const auto &opt_find_rank : opt_rank_list_find) {
    MS_LOG(INFO) << "group device member:" << opt_find_rank;
  }

  RankList expanded_list;
  for (const auto &rank : rank_list) {
    Status ret_state;
    RankList opt_shard_rank_list;
    std::tie(ret_state, opt_shard_rank_list) = GetOptShardRankList(rank);
    if (ret_state != SUCCESS) {
      MS_LOG(EXCEPTION) << "find opt shard rank list in adafactor failed";
    }
    MS_LOG(INFO) << "found opt shard rank list for rank " << rank;

    for (const auto &opt_rank : opt_shard_rank_list) {
      MS_LOG(INFO) << opt_rank;
    }
    expanded_list.insert(expanded_list.end(), opt_shard_rank_list.begin(), opt_shard_rank_list.end());
  }
  std::sort(expanded_list.begin(), expanded_list.end());
  MS_LOG(INFO) << "expand rank list with opt shard, before:";
  for (const auto &item : rank_list) {
    MS_LOG(INFO) << item;
  }
  MS_LOG(INFO) << "after:";
  for (const auto &item : expanded_list) {
    MS_LOG(INFO) << item;
  }
  return expanded_list;
}

RankList CameCommHandler::ExpandRankListWithDim(const RankList &rank_list, const int64_t dim) {
  RankList expanded_list;
  for (const auto &rank : rank_list) {
    Status ret_status;
    RankList dim_rank_list;
    std::tie(ret_status, dim_rank_list) = GetDimRankList(rank, dim);
    if (ret_status != SUCCESS) {
      MS_LOG(EXCEPTION) << "find dim rank list in adafactor failed";
    }
    expanded_list.insert(expanded_list.end(), dim_rank_list.begin(), dim_rank_list.end());
  }
  std::sort(expanded_list.begin(), expanded_list.end());
  return expanded_list;
}

CNodePtr CameCommHandler::FindReduceMean(size_t number) {
  if (reduce_mean_numbers.find(number) == reduce_mean_numbers.end()) {
    MS_LOG(INFO) << "[CAME] invalid reduce mean number: " << number;
  }

  if (number == kFirstCameReduceMean) {
    return FindReduceMean1256(exp_avg_sq_row);
  } else if (number == kSecondCameReduceMean) {
    return FindReduceMean1256(exp_avg_sq_col);
  } else if (number == kThirdCameReduceMean) {
    return FindReduceMean37(exp_avg_sq_row);
  } else if (number == kForthCameReduceMean) {
    return FindReduceMean4();
  } else if (number == kFifthCameReduceMean) {
    return FindReduceMean1256(exp_avg_insta_row);
  } else if (number == kSixthCameReduceMean) {
    return FindReduceMean1256(exp_avg_insta_col);
  } else if (number == kSeventhCameReduceMean) {
    return FindReduceMean37(exp_avg_insta_row);
  } else {
    return nullptr;
  }
}

CNodePtr CameCommHandler::FindReduceMean1256(const ParameterPtr &param) {
  if (!param) {
    return nullptr;
  }
  MS_LOG(INFO) << "[CAME] try find reduce_mean according to " << param->name() << " Assign:";
  auto param_user_set = node_user_map.at(param->cast<AnfNodePtr>());
  for (auto &param_pair : param_user_set) {
    auto user_cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    if (IsSomePrimitive(user_cnode, ASSIGN)) {
      MS_LOG(INFO) << "[CAME] found assign node";
      // assign 1 -> add 1 -> mul 0 -> reduce_mean
      auto res = BackwardSearchCNode(user_cnode, {{ASSIGN, 1}, {ADD, 1}, {MUL, 0}}, REDUCE_MEAN);
      if (res.first) {
        MS_LOG(INFO) << "[CAME] found reduce mean node: " << res.second->DebugString();
        return res.second;
      }
    }
  }
  return nullptr;
}

CNodePtr CameCommHandler::FindReduceMean37(const ParameterPtr &param) {
  if (!param) {
    return nullptr;
  }
  auto param_user_set = node_user_map.at(param->cast<AnfNodePtr>());
  MS_LOG(INFO) << "[CAME] user map size: " << param_user_set.size();
  size_t load_count = 0;
  for (auto &param_pair : param_user_set) {
    auto user_cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    if (IsSomePrimitive(user_cnode, LOAD)) {
      MS_LOG(INFO) << "[CAME] found load node";
      load_count++;
      // load -> reduce mean
      auto res = ForwardSearchCNode(user_cnode, {LOAD, REDUCE_MEAN}, node_user_map);
      if (res.first) {
        MS_LOG(INFO) << "[CAME] found reduce mean node size: " << res.second.size();
        return res.second[0];  // get the first one
      }
    }
  }
  MS_LOG(INFO) << "[CAME] found load count: " << load_count;
  return nullptr;
}

CNodePtr CameCommHandler::FindReduceMean4() {
  MS_LOG(INFO) << "[CAME] try find reduce_mean no.4 according to exp_avg Assign:";
  if (!exp_avg) {
    return nullptr;
  }
  auto exp_avg_user_set = node_user_map.at(exp_avg->cast<AnfNodePtr>());
  for (auto &param_pair : exp_avg_user_set) {
    auto user_cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    if (IsSomePrimitive(user_cnode, ASSIGN)) {
      MS_LOG(INFO) << "[CAME] found exp_avg's assign node";
      auto res = BackwardSearchCNode(
        user_cnode, {{ASSIGN, 1}, {ADD, 1}, {MUL, 0}, {REAL_DIV, 1}, {MAXIMUM, 0}, {REAL_DIV, 0}, {SQRT, 0}},
        REDUCE_MEAN);
      if (res.first) {
        MS_LOG(INFO) << "[CAME] found reduce mean node: " << res.second->DebugString();
        return res.second;
      }
    }
  }
  return nullptr;
}

void CameCommHandler::InsertAllReduceAndRealDivToReduceMeanInput(CNodePtr reduce_mean, const RankList &comm_rank_list) {
  // construct all reduce cnode and insert to the first input
  if (!reduce_mean) {
    return;
  }
  FuncGraphPtr func_graph = reduce_mean->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  CheckGlobalDeviceManager();

  MS_LOG(INFO) << "Insert All Reduce and RealDiv to node" << reduce_mean->DebugString();
  // insert all reduce
  OperatorName allreduce_op_name = ALL_REDUCE;
  OperatorAttrs all_reduce_op_attrs;
  ValuePtr allreduce_pyop_instance = CreateOpInstance(all_reduce_op_attrs, allreduce_op_name, "came_norm_allreduce");
  std::vector<AnfNodePtr> all_reduce_input = {NewValueNode(allreduce_pyop_instance), reduce_mean->input(1)};
  auto all_reduce_node = func_graph->NewCNode(all_reduce_input);
  auto all_reduce_prim = GetCNodePrimitive(all_reduce_node);
  auto all_reduce_attrs = all_reduce_prim->attrs();
  all_reduce_attrs["op"] = MakeValue<std::string>(REDUCE_OP_SUM);

  std::string group_name = CreateCommGroupFromRankList(comm_rank_list);
  MS_LOG(INFO) << "[CAME] came allreduce opt shard group: " << group_name;
  all_reduce_attrs["group"] = MakeValue<std::string>(group_name);
  int64_t fusion_id = 0;
  all_reduce_attrs["fusion"] = MakeValue(fusion_id);
  all_reduce_prim->SetAttrs(all_reduce_attrs);
  // insert real div
  OperatorName operator_name = REAL_DIV;
  OperatorAttrs operator_attrs;

  ValuePtr pyop_instance = CreateOpInstance(operator_attrs, operator_name, "came_norm_realdiv");
  MS_EXCEPTION_IF_NULL(pyop_instance);

  size_t group_rank_size = comm_rank_list.size();
  mindspore::tensor::TensorPtr tensor_ptr =
    std::make_shared<mindspore::tensor::Tensor>(static_cast<float>(group_rank_size));
  ValuePtr scale_value = MakeValue(tensor_ptr);

  std::vector<AnfNodePtr> real_div_input = {NewValueNode(pyop_instance), all_reduce_node->cast<AnfNodePtr>(),
                                            NewValueNode(scale_value)};
  auto real_div_node = func_graph->NewCNode(real_div_input);
  manager->SetEdge(reduce_mean, 1, real_div_node);
}

void CameCommHandler::Process() {
  auto reduce_mean_1 = FindReduceMean(1);
  auto reduce_mean_2 = FindReduceMean(2);
  auto reduce_mean_3 = FindReduceMean(3);
  auto reduce_mean_4 = FindReduceMean(4);
  auto reduce_mean_5 = FindReduceMean(5);
  auto reduce_mean_6 = FindReduceMean(6);
  auto reduce_mean_7 = FindReduceMean(7);
  MS_LOG(INFO) << "found all reduce mean for came/adafactor";

  auto shape_size = tensor_layout->slice_shape().array().size();
  if (shape_size == 1) {
    // for shape [A], mp and opt shard may overlay on dim A.
    Status ret_status;
    RankList comm_rank_list;
    std::tie(ret_status, comm_rank_list) = GetDimRankList(cur_rank, 0);
    if (ret_status != SUCCESS) {
      MS_LOG(ERROR) << "[CAME] shape size = 1, getting rank list along 0 failed";
    }
    comm_rank_list = ExpandRankListWithOptShard(comm_rank_list);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_4, comm_rank_list);
  } else {
    Status ret_status;
    RankList comm_rank_list_along_neg_1;
    RankList comm_rank_list_along_neg_2;
    RankList comm_rank_list_along_neg_12;
    int64_t actual_dim_of_neg_1 = shape_size - 1;
    int64_t actual_dim_of_neg_2 = shape_size - 2;
    std::tie(ret_status, comm_rank_list_along_neg_1) = GetDimRankList(cur_rank, actual_dim_of_neg_1);
    if (ret_status != SUCCESS) {
      MS_LOG(ERROR) << "[CAME] shape = 2, getting rank list along negative dim -1 failed";
    }
    std::tie(ret_status, comm_rank_list_along_neg_2) = GetDimRankList(cur_rank, actual_dim_of_neg_2);
    if (ret_status != SUCCESS) {
      MS_LOG(ERROR) << "[CAME] shape = 2, getting rank list along negative dim -2 failed";
    }
    if (shape_size == kParameterDimTwo) {
      comm_rank_list_along_neg_2 = ExpandRankListWithOptShard(comm_rank_list_along_neg_2);
    }
    comm_rank_list_along_neg_12 = ExpandRankListWithDim(comm_rank_list_along_neg_2, actual_dim_of_neg_1);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_1, comm_rank_list_along_neg_1);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_2, comm_rank_list_along_neg_2);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_3, comm_rank_list_along_neg_2);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_4, comm_rank_list_along_neg_12);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_5, comm_rank_list_along_neg_1);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_6, comm_rank_list_along_neg_2);
    InsertAllReduceAndRealDivToReduceMeanInput(reduce_mean_7, comm_rank_list_along_neg_2);
  }
}

std::string CameCommHandler::CreateCommGroupFromRankList(const RankList &rank_list) {
  Group comm_group;
  if (g_device_manager->CreateGroup(rank_list, &comm_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create comm group failed in came";
  }
  std::string group_name = comm_group.name();
  return group_name;
}

}  // namespace parallel
}  // namespace mindspore
