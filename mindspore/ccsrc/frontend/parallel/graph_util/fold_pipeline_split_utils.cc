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

#include "frontend/parallel/graph_util/fold_pipeline_split_utils.h"
#include <memory>
#include <list>
#include <set>
#include <queue>
#include <algorithm>

#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "ops/other_ops.h"
#include "ops/math_ops.h"
#include "ops/framework_ops.h"
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "ir/value.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {

namespace {
constexpr int kBackwardEnd = 1;
constexpr int kForwardStart = 2;
constexpr int kForwardEnd = 3;
}  // namespace

const std::set<PrimitivePtr> END_NODE_BLACK_LIST = {
  prim::kPrimDepend,    prim::kPrimTupleGetItem, prim::kPrimAdd,    prim::kPrimSoftmaxCrossEntropyWithLogits,
  prim::kPrimMakeTuple, prim::kPrimUpdateState,  prim::kPrimReshape};

int64_t GetSegmentMax(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &forward_end) {
  int64_t seg_max = 0;
  if (forward_end.empty()) {
    MS_LOG(EXCEPTION) << "Can not find the end node of pipeline, you are advised to use 'PipelineCell' to fix it.";
  } else {
    auto forward_end_cnode = forward_end.back()->cast<CNodePtr>();
    auto seg_size = forward_end_cnode->GetPrimalAttr(SEGMENT);
    MS_EXCEPTION_IF_NULL(seg_size);
    seg_max = GetValue<int64_t>(seg_size);
  }
  return seg_max;
}

std::vector<PipelinePair> GetSubStepPairs(const PipelinePair &fp_or_bp_pair, int64_t sub_step_num, int64_t seg_num,
                                          int64_t sub_micro_num, int64_t micro_num) {
  std::vector<PipelinePair> fp_or_bp_sub_pairs;
  for (int64_t s = 0; s < sub_step_num; s++) {
    std::vector<AnfNodePtr> temp_first;
    std::vector<AnfNodePtr> temp_second;
    for (int64_t sid = 0; sid < seg_num; sid++) {
      temp_first.insert(temp_first.end(), fp_or_bp_pair.first.begin() + s * sub_micro_num + sid * micro_num,
                        fp_or_bp_pair.first.begin() + (s + 1) * sub_micro_num + sid * micro_num);
      temp_second.insert(temp_second.end(), fp_or_bp_pair.second.begin() + s * sub_micro_num + sid * micro_num,
                         fp_or_bp_pair.second.begin() + (s + 1) * sub_micro_num + sid * micro_num);
    }
    fp_or_bp_sub_pairs.emplace_back(temp_first, temp_second);
  }
  return fp_or_bp_sub_pairs;
}

bool CompFuncBySegAscending(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  if (parallel_context->enable_fold_pipeline()) {
    auto get_value_func = [](const AnfNodePtr &node) {
      MS_EXCEPTION_IF_NULL(node);
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto seg = cnode->GetPrimalAttr(SEGMENT);
      MS_EXCEPTION_IF_NULL(seg);
      return GetValue<int64_t>(seg);
    };

    if (get_value_func(node1) != get_value_func(node2)) {
      return get_value_func(node1) < get_value_func(node2);
    }
  }
  return CompFunc(node1, node2);
}

bool CompFuncBySegDescending(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  if (parallel_context->enable_fold_pipeline()) {
    auto get_value_func = [](const AnfNodePtr &node) {
      MS_EXCEPTION_IF_NULL(node);
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto seg = cnode->GetPrimalAttr(SEGMENT);
      MS_EXCEPTION_IF_NULL(seg);
      return GetValue<int64_t>(seg);
    };

    if (get_value_func(node1) != get_value_func(node2)) {
      return get_value_func(node1) > get_value_func(node2);
    }
  }
  return CompFunc(node1, node2);
}

void InsertVirtualFoldPipelineEndNode(const AnfNodePtr &temp_node, const FuncGraphManagerPtr &manager) {
  auto end_node = GetPreNode(temp_node);
  MS_EXCEPTION_IF_NULL(end_node);
  auto end_cnode = end_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(end_cnode);
  auto end_prim = GetCNodePrimitive(end_node);
  OperatorAttrs attrs_;
  auto op = CreateOpInstance(attrs_, "_VirtualPipelineEnd", "end_node");
  auto value_node = NewValueNode(op);
  auto new_prim = GetValueNode(value_node)->cast<PrimitivePtr>();
  (void)new_prim->SetAttrs(end_prim->attrs());
  manager->SetEdge(end_node, 0, value_node);
  end_cnode->AddPrimalAttr(PIPELINE_END, end_cnode->GetPrimalAttr(MICRO));
  auto seg = ParallelContext::GetInstance()->pipeline_segment_split_num();
  end_cnode->AddPrimalAttr(SEGMENT, MakeValue(seg - 1));
}

AnfNodePtr FindNodeFirstUser(const FuncGraphPtr &root, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(root);
  auto node_users_map = root->manager()->node_users();
  auto users = node_users_map[node];
  for (auto &temp_user : users) {
    MS_LOG(INFO) << "Receive user: " << (temp_user.first)->ToString();
    return temp_user.first;
  }
  return nullptr;
}

static bool IsInEndNodeBlackListOrParallelBlackList(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return true;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (IsInParallelBlackList(prim)) {
    return true;
  }
  for (auto &prim_node : END_NODE_BLACK_LIST) {
    if (IsPrimitiveCNode(cnode, prim_node)) {
      return true;
    }
  }
  return false;
}

AnfNodePtr GetPreNode(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> node_queue = {node};
  while (!node_queue.empty()) {
    auto cur_node = (*node_queue.begin())->cast<CNodePtr>();
    (void)node_queue.erase(node_queue.begin());
    if (!cur_node) {
      continue;
    }
    if (!IsInEndNodeBlackListOrParallelBlackList(cur_node) && cur_node->HasPrimalAttr(NEED_GRAD)) {
      MS_LOG(INFO) << "Pipeline End node: " << cur_node->DebugString();
      return cur_node;
    }
    (void)node_queue.insert(node_queue.end(), cur_node->inputs().begin() + 1, cur_node->inputs().end());
  }
  MS_LOG(EXCEPTION) << "Get Pipeline End node failed.";
}

static bool ComputeLastSegForwardEndIdx(const PipelinePair &forward_start, size_t curr_idx, int64_t micro_max,
                                        int64_t stage_num, int64_t stage_id) {
  auto last_seg_idx = static_cast<size_t>(1 + micro_max + 1 - 2 * (stage_num - stage_id - 1) - 1);
  return curr_idx > forward_start.first.size() - last_seg_idx;
}

void ReorderForFoldPipelineForward(const std::vector<PipelinePair> &pair_vector, int64_t seg_max, int64_t micro_max,
                                   const FuncGraphPtr &root, AnfNodePtr *start_of_forward, AnfNodePtr *end_of_forward,
                                   bool enable_1f1b) {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto stage_num = g_device_manager->stage_num();
  auto stage_id = g_device_manager->stage_id();
  *start_of_forward = pair_vector[kForwardStart].first[0];
  for (size_t i = 1; i < pair_vector[kForwardStart].first.size(); ++i) {
    auto prior_node_begin = pair_vector[kForwardEnd].first[i - 1];
    auto prior_node_end = pair_vector[kForwardEnd].second[i - 1];
    auto post_node_begin = pair_vector[kForwardStart].first[i];
    auto post_node_end = pair_vector[kForwardStart].second[i];
    if (IsFirstStage() && (i > IntToSize(micro_max))) {
      auto receive_node = post_node_begin;
      post_node_begin = FindNodeFirstUser(root, post_node_begin);

      MS_EXCEPTION_IF_NULL(post_node_begin);
      auto insert_idx = i - (micro_max + 1) + (stage_num - 1);
      auto send_node_begin = pair_vector[3].first[insert_idx];
      auto send_node_end = pair_vector[3].second[insert_idx];
      InsertDepend(post_node_end, send_node_begin, manager, root);

      auto send_cnode = send_node_begin->cast<CNodePtr>();
      auto before_send_node = GetActualOp(send_cnode->input(1));

      InsertDepend(before_send_node, receive_node, manager, root);
    }
    if (enable_1f1b && ComputeLastSegForwardEndIdx(pair_vector[kForwardStart], i, micro_max, stage_num, stage_id)) {
      continue;
    }

    InsertDepend(prior_node_end, post_node_begin, manager, root);
    *end_of_forward = pair_vector[kForwardEnd].second[i];
  }
  (*end_of_forward)->cast<CNodePtr>()->AddPrimalAttr(FORWARD_END, MakeValue(true));
  (*end_of_forward)->cast<CNodePtr>()->AddPrimalAttr(SEGMENT_MAX, MakeValue(seg_max));
}

void ReorderForBackwardLastSeg(const std::vector<PipelinePair> &pair_vector, const FuncGraphPtr &root,
                               AnfNodePtr *start_of_backward, AnfNodePtr *end_of_backward, int64_t micro_max) {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto stage_num = g_device_manager->stage_num();
  auto stage_id = g_device_manager->stage_id();
  int64_t seg_max = GetSegmentMax(root, pair_vector[3].second);
  MS_LOG(INFO) << "Micro max:" << micro_max << "seg_max" << seg_max;
  int64_t last_seg_index = SizeToLong(pair_vector[2].first.size()) - 1 - micro_max;
  int64_t cur_stage_fwd_max_idx = 2 * (stage_num - stage_id - 1) + 1;
  if (!IsFirstStage() && (micro_max + 1 > cur_stage_fwd_max_idx)) {
    for (size_t i = LongToSize(cur_stage_fwd_max_idx); i < LongToSize(micro_max + 1); ++i) {
      auto forward_node_begin = pair_vector[2].first[LongToSize(last_seg_index) + i];
      auto forward_node_end = pair_vector[2].second[LongToSize(last_seg_index) + i];
      size_t insert_idx;
      if (i == LongToSize(cur_stage_fwd_max_idx)) {
        if (IsLastStage()) {
          continue;
        }
        insert_idx = LongToSize(last_seg_index) + i - 1;
        auto post_node = pair_vector[3].first[insert_idx];
        InsertDepend(forward_node_end, post_node, manager, root);

        auto prior_node = pair_vector[4].second[insert_idx];
        InsertDepend(prior_node, forward_node_begin, manager, root);
      } else {
        if (IsLastStage() && i == LongToSize(cur_stage_fwd_max_idx + 1)) {
          auto post_node0 = pair_vector[1].first[0];
          InsertDepend(forward_node_end, post_node0, manager, root);
          auto pre_prior_node = pair_vector[2].second[LongToSize(last_seg_index) + i - 1];
          InsertDepend(pre_prior_node, forward_node_begin, manager, root);
          auto pre_post_node = pair_vector[2].first[LongToSize(last_seg_index) + i - 1];
          auto prior_node0 = GetActualOp(pair_vector[1].first[0]->cast<CNodePtr>()->input(1));
          InsertDepend(prior_node0, pre_post_node, manager, root);
          continue;
        }
        insert_idx = i - LongToSize(cur_stage_fwd_max_idx) - 1;
        auto post_node1 = pair_vector[1].first[insert_idx];
        InsertDepend(forward_node_end, post_node1, manager, root);

        auto prior_cnode1 = post_node1->cast<CNodePtr>();
        auto before_prior_cnode = GetActualOp(prior_cnode1->input(1));
        InsertDepend(before_prior_cnode, forward_node_begin, manager, root);
      }
    }
  }

  if (micro_max + 1 > cur_stage_fwd_max_idx) {
    for (size_t i = LongToSize(cur_stage_fwd_max_idx); i < LongToSize(micro_max + 1); ++i) {
      if (!IsLastStage()) {
        auto prior_node1 = pair_vector[3].second[last_seg_index + i];
        auto post_node1 = pair_vector[0].first[LongToSize(SizeToLong(i) - cur_stage_fwd_max_idx + 1)];
        InsertDepend(prior_node1, post_node1, manager, root);
      }
      std::shared_ptr<AnfNode> post_node2;
      post_node2 = FindNodeFirstUser(root, pair_vector[kForwardStart].first[last_seg_index + i]);
      auto prior_node2 = pair_vector[1].second[LongToSize(SizeToLong(i) - cur_stage_fwd_max_idx)];
      InsertDepend(prior_node2, post_node2, manager, root);
    }

    for (size_t j = LongToSize(micro_max + 1 - 2 * (stage_num - stage_id - 1)); j < LongToSize(micro_max + 1); ++j) {
      auto prior_node3 = pair_vector[1].second[j - 1];
      auto post_node3 = pair_vector[0].first[j];
      InsertDepend(prior_node3, post_node3, manager, root);
    }
  } else {
    for (size_t j = 1; j < LongToSize(micro_max + 1); ++j) {
      auto prior_node4 = pair_vector[1].second[j - 1];
      auto post_node4 = pair_vector[0].first[j];
      InsertDepend(prior_node4, post_node4, manager, root);
    }
  }

  if (!IsLastStage()) {
    std::shared_ptr<AnfNode> prior_node5;
    if ((micro_max + 1 > cur_stage_fwd_max_idx)) {
      prior_node5 = pair_vector[kForwardEnd].second[LongToSize(last_seg_index + cur_stage_fwd_max_idx - 1)];
    } else {
      prior_node5 = pair_vector[kForwardEnd].second[LongToSize(last_seg_index + micro_max)];
    }
    auto post_node5 = pair_vector[0].first[0];
    InsertDepend(prior_node5, post_node5, manager, root);
  }

  for (size_t i = 0; i < pair_vector[0].first.size(); ++i) {
    pair_vector[0].first[i]->cast<CNodePtr>()->AddPrimalAttr(BACKWARD_MICRO_END, MakeValue(true));
    pair_vector[0].first[i]->cast<CNodePtr>()->AddPrimalAttr(SEGMENT_MAX, MakeValue(seg_max));
  }
  *start_of_backward = pair_vector[0].first[0];
  *end_of_backward = pair_vector[1].second.back();
  ReorderForBackwardOtherSeg(pair_vector[0], pair_vector[1], micro_max, stage_num, root);
}

void ReorderForBackwardOtherSeg(const PipelinePair &backward_start_pair, const PipelinePair &backward_end_pair,
                                int64_t micro_max, int64_t stage_num, const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  for (size_t i = micro_max + 1; i < backward_start_pair.first.size(); ++i) {
    auto prior_node_begin = backward_end_pair.first[i - 1];
    auto prior_node_end = backward_end_pair.second[i - 1];
    auto post_node_begin = backward_start_pair.first[i];
    auto post_node_end = backward_start_pair.second[i];

    if (IsLastStage() && (i > IntToSize(micro_max))) {
      auto receive_node = post_node_begin;
      post_node_begin = FindNodeFirstUser(root, post_node_begin);
      int64_t insert_idx = SizeToLong(i) - (micro_max + 1) + (stage_num - 1);
      auto send_node_begin = backward_end_pair.first[insert_idx];
      auto send_node_end = backward_end_pair.second[insert_idx];
      InsertDepend(post_node_end, send_node_begin, manager, root);

      auto send_cnode = send_node_begin->cast<CNodePtr>();
      auto before_send_node = GetActualOp(send_cnode->input(1));
      before_send_node = GetActualOp((before_send_node->cast<CNodePtr>())->input(1));

      InsertDepend(before_send_node, receive_node, manager, root);
    }

    InsertDepend(prior_node_end, post_node_begin, manager, root);
  }
}

PipelinePair Deduplicate(const std::vector<AnfNodePtr> &node_vector, const FuncGraphPtr &root, int64_t micro_max,
                         int64_t seg_max, bool is_train) {
  std::vector<AnfNodePtr> out_vec_begin;
  std::vector<AnfNodePtr> out_vec_end;
  for (int64_t h = 0; h <= seg_max; ++h) {
    CommonDeduplicate(node_vector, &out_vec_begin, &out_vec_end, root, micro_max, seg_max, h, is_train);
  }
  if (out_vec_begin.empty()) {
    return std::make_pair(node_vector, node_vector);
  }
  return std::make_pair(out_vec_begin, out_vec_end);
}

PipelinePair DeduplicateBySegAscending(const std::vector<AnfNodePtr> &node_vector, const FuncGraphPtr &root,
                                       int64_t micro_max, bool is_train, int64_t seg_max = 0) {
  std::vector<AnfNodePtr> out_vec_begin;
  std::vector<AnfNodePtr> out_vec_end;
  for (int64_t h = 0; h <= seg_max; ++h) {
    CommonDeduplicate(node_vector, &out_vec_begin, &out_vec_end, root, micro_max, seg_max, h, is_train);
  }
  if (out_vec_begin.empty()) {
    return std::make_pair(node_vector, node_vector);
  }
  return std::make_pair(out_vec_begin, out_vec_end);
}

PipelinePair DeduplicateBySegDescending(const std::vector<AnfNodePtr> &node_vector, const FuncGraphPtr &root,
                                        int64_t micro_max, bool is_train, int64_t seg_max = 0) {
  std::vector<AnfNodePtr> out_vec_begin;
  std::vector<AnfNodePtr> out_vec_end;
  for (int64_t h = seg_max; h >= 0; --h) {
    CommonDeduplicate(node_vector, &out_vec_begin, &out_vec_end, root, micro_max, seg_max, h, is_train);
  }
  if (out_vec_begin.empty()) {
    return std::make_pair(node_vector, node_vector);
  }
  return std::make_pair(out_vec_begin, out_vec_end);
}

void ReorderForFoldPipelineBackward(const std::vector<PipelinePair> &pair_vector, int64_t seg_max, int64_t micro_max,
                                    const FuncGraphPtr &root, AnfNodePtr *start_of_backward,
                                    AnfNodePtr *end_of_backward) {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto stage_num = g_device_manager->stage_num();

  bool first = true;
  for (size_t i = 0; i < pair_vector[0].first.size(); ++i) {
    pair_vector[0].first[i]->cast<CNodePtr>()->AddPrimalAttr(BACKWARD_MICRO_END, MakeValue(true));
    pair_vector[0].first[i]->cast<CNodePtr>()->AddPrimalAttr(SEGMENT_MAX, MakeValue(seg_max));
  }
  for (size_t i = 1; i < pair_vector[0].first.size(); ++i) {
    auto prior_node_begin = pair_vector[1].first[i - 1];
    auto prior_node_end = pair_vector[1].second[i - 1];
    auto post_node_begin = pair_vector[0].first[i];
    auto post_node_end = pair_vector[0].second[i];

    if (IsLastStage() && (i > IntToSize(micro_max))) {
      auto receive_node = post_node_begin;
      post_node_begin = FindNodeFirstUser(root, post_node_begin);
      auto insert_idx = i - (micro_max + 1) + (stage_num - 1);
      auto send_node_begin = pair_vector[1].first[insert_idx];
      auto send_node_end = pair_vector[1].second[insert_idx];

      InsertDepend(post_node_end, send_node_begin, manager, root);

      auto send_cnode = send_node_begin->cast<CNodePtr>();
      auto before_send_node = GetActualOp(send_cnode->input(1));
      before_send_node = GetActualOp((before_send_node->cast<CNodePtr>())->input(1));

      InsertDepend(before_send_node, receive_node, manager, root);
    }

    InsertDepend(prior_node_end, post_node_begin, manager, root);
    if (first) {
      *start_of_backward = pair_vector[0].first[i - 1];
      first = false;
    }
  }
  *end_of_backward = pair_vector[1].second.back();
}

PipelinePairVector UpdateSubPairs(int64_t sub_step_num, int64_t micro_num, std::vector<PipelinePair> pair_vector,
                                  int64_t sub_micro_num, int64_t seg_num) {
  PipelinePairVector sub_pair_vector;
  PipelinePairVector tmp_pair_vector;
  if (micro_num % sub_step_num != 0) {
    MS_LOG(EXCEPTION) << "Micro_num(" << micro_num << ")cannot be divisible by sub_step_num(" << sub_step_num << ").";
  }

  if (sub_micro_num < g_device_manager->stage_num()) {
    MS_LOG(EXCEPTION) << "Sub_micro_num(" << sub_micro_num << ") is less than stage_num("
                      << g_device_manager->stage_num() << ").";
  }
  MS_LOG(INFO) << "Micro_num=" << micro_num << ",sub_micro_num=" << sub_micro_num << ",seg_num = " << seg_num;

  std::transform(pair_vector.begin(), pair_vector.end(), std::back_inserter(tmp_pair_vector),
                 [&sub_step_num, &seg_num, &sub_micro_num, &micro_num](const auto &pipeline_pair) {
                   return GetSubStepPairs(pipeline_pair, sub_step_num, seg_num, sub_micro_num, micro_num);
                 });

  for (size_t i = 0; i < tmp_pair_vector.size(); i++) {
    std::vector<PipelinePair> sub_step1;
    std::vector<PipelinePair> sub_step2;
    if (!sub_pair_vector.empty()) {
      sub_pair_vector[0].push_back(sub_pair_vector[i][0]);
      sub_pair_vector[1].push_back(sub_pair_vector[i][1]);
    } else {
      sub_step1.push_back(sub_pair_vector[i][0]);
      sub_pair_vector.push_back(sub_step1);
      sub_step2.push_back(sub_pair_vector[i][1]);
      sub_pair_vector.push_back(sub_step2);
    }
  }
  return sub_pair_vector;
}

void FoldPipelineReorder(const FuncGraphPtr &root) {
  std::vector<AnfNodePtr> forward_start;
  std::vector<AnfNodePtr> forward_end;
  std::vector<AnfNodePtr> forward_params;
  std::vector<AnfNodePtr> backward_start;
  std::vector<AnfNodePtr> backward_end;
  std::vector<AnfNodePtr> backward_params;
  std::vector<AnfNodePtr> allreduce_params;

  SetParameterStartForCellShare(root);
  GetBorderNode(&forward_start, &forward_end, &backward_start, &backward_end, &forward_params, &backward_params,
                &allreduce_params, root);
  int64_t micro_max = GetMicroMax(root, forward_end);
  int64_t seg_max = GetSegmentMax(root, forward_end);
  std::vector<int64_t> seg_micro_max{micro_max, seg_max};

  auto backward_start_pair = DeduplicateBySegDescending(backward_start, root, micro_max, true, seg_max);
  auto backward_end_pair = DeduplicateBySegDescending(backward_end, root, micro_max, true, seg_max);
  auto forward_start_pair = DeduplicateBySegAscending(forward_start, root, micro_max, true, seg_max);
  auto forward_end_pair = DeduplicateBySegAscending(forward_end, root, micro_max, true, seg_max);
  auto forward_params_pair = Deduplicate(forward_params, root, micro_max, true, seg_max);
  auto backward_params_pair = Deduplicate(backward_params, root, micro_max, true, seg_max);
  CheckBorderNode(forward_start_pair, forward_end_pair, backward_start_pair, backward_end_pair, seg_micro_max);
  auto forward_end_before_pair = GetForwardEndBeforePair(forward_end_pair);
  std::vector<PipelinePair> pair_vector{backward_start_pair, backward_end_pair, forward_start_pair, forward_end_pair,
                                        forward_end_before_pair};
  AnfNodePtr start_of_forward;
  AnfNodePtr end_of_forward;
  AnfNodePtr start_of_backward;
  AnfNodePtr end_of_backward;
  AnfNodePtr pre_end_of_backward;

  bool enable_1f1b = false;
  if (common::GetEnv("FOLD_LAST_SEG_1F1B") != "") {
    enable_1f1b = true;
  }
  int64_t sub_step_num = 0;
  int64_t sub_micro_num = 0;
  if (common::GetEnv("FOLD_ACCUMULATION") != "") sub_step_num = std::stoi(common::GetEnv("FOLD_ACCUMULATION"));
  MS_LOG(INFO) << "Sub_step_num=" << sub_step_num;
  PipelinePairVector sub_pair_vector;
  if (sub_step_num > 0) {
    int64_t micro_num = micro_max + 1;
    int64_t seg_num = seg_max + 1;
    sub_micro_num = micro_num / sub_step_num;
    sub_pair_vector = UpdateSubPairs(sub_step_num, micro_num, pair_vector, sub_micro_num, seg_num);
  }

  if (enable_1f1b) {
    if (sub_step_num > 0) {
      for (int64_t s = 0; s < sub_step_num; s++) {
        ReorderForFoldPipelineForward(sub_pair_vector[s], seg_max, sub_micro_num - 1, root, &start_of_forward,
                                      &end_of_forward, enable_1f1b);
        ReorderForBackwardLastSeg(sub_pair_vector[s], root, &start_of_backward, &end_of_backward, sub_micro_num - 1);
        if (s > 0) {
          InsertDepend(pre_end_of_backward, start_of_forward, root->manager(), root);
        }
        pre_end_of_backward = end_of_backward;
        ReorderForParams(backward_params_pair, forward_params_pair, sub_pair_vector[kBackwardEnd][s],
                         sub_pair_vector[kForwardStart][s], root);
      }
    } else {
      ReorderForFoldPipelineForward(pair_vector, seg_max, micro_max, root, &start_of_forward, &end_of_forward,
                                    enable_1f1b);
      ReorderForBackwardLastSeg(pair_vector, root, &start_of_backward, &end_of_backward, micro_max);
      ReorderForParams(backward_params_pair, forward_params_pair, backward_end_pair, forward_start_pair, root);
    }
  } else {
    if (sub_step_num > 0) {
      for (int64_t s = 0; s < sub_step_num; s++) {
        ReorderForFoldPipelineForward(sub_pair_vector[s], seg_max, sub_micro_num - 1, root, &start_of_forward,
                                      &end_of_forward, enable_1f1b);

        ReorderForFoldPipelineBackward(sub_pair_vector[s], seg_max, sub_micro_num - 1, root, &start_of_backward,
                                       &end_of_backward);
        InsertDepend(end_of_forward, start_of_backward, root->manager(), root);
        if (s > 0) {
          InsertDepend(pre_end_of_backward, start_of_forward, root->manager(), root);
        }
        pre_end_of_backward = end_of_backward;
        ReorderForParams(backward_params_pair, forward_params_pair, sub_pair_vector[1][s], sub_pair_vector[2][s], root);
      }
    } else {
      ReorderForFoldPipelineForward(pair_vector, seg_max, micro_max, root, &start_of_forward, &end_of_forward,
                                    enable_1f1b);
      ReorderForFoldPipelineBackward(pair_vector, seg_max, micro_max, root, &start_of_backward, &end_of_backward);
      InsertDepend(end_of_forward, start_of_backward, root->manager(), root);
      ReorderForParams(backward_params_pair, forward_params_pair, backward_end_pair, forward_start_pair, root);
    }
  }
}

}  // namespace parallel
}  // namespace mindspore
