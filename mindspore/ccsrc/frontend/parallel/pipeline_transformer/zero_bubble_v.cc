/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pipeline_transformer/zero_bubble_v.h"

#include <algorithm>
#include <queue>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "include/utils/utils.h"
#include "mindspore/core/include/ir/core_ops_primitive.h"
#include "primitive/framework_ops.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace parallel {
enum class BorderType { kSend, kReceive, kCallForward, kCallBackward, kOther };
bool CompareBorderPair(const BorderPair &bp1, const BorderPair &bp2) {
  auto compare_border = [](const Border &b1, const Border &b2) {
    return b1.chunk == b2.chunk && b1.micro == b2.micro && b1.seq_chunk == b2.seq_chunk && b1.border == b2.border;
  };
  return compare_border(bp1.first, bp2.first) && compare_border(bp1.second, bp2.second);
}

bool CheckChunkAndMicro(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!cnode->HasPrimalAttr(CHUNK) || !cnode->HasPrimalAttr(MICRO)) {
    return false;
  }
  return true;
}

bool IsSendOrReceive(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return false;
  }
  return true;
}

void ZeroBubbleV::InsertCallControlOrder(const std::vector<BorderPair> &borders, const std::string &tags) {
  size_t size = borders.size();
  if (size < kIndexThree) {
    return;
  }
  for (size_t i = 0; i < size; i++) {
    if (i + kIndexTwo >= size) {
      continue;
    }
    const auto &prior = borders[i];
    const auto &last = borders[i + kIndexTwo];
    ControlOrder(prior.second, last.first, tags);
  }
}

void Add1b1fReceiveAttr(const BorderPair &recv, const std::string &tag, size_t index_1b1f) {
  if (IsPrimitiveCNode(recv.second.border)) {
    recv.second.border->AddAttr(tag, MakeValue<size_t>(index_1b1f));
  }
  if (IsPrimitiveCNode(recv.first.border)) {
    recv.first.border->AddAttr(tag, MakeValue<size_t>(index_1b1f));
  }
}

void ZeroBubbleV::InsertControlOrder(const std::vector<BorderPair> &borders, size_t start, size_t end,
                                     const std::string &tags) {
  while (start < end) {
    auto prior_border = borders[start].second;
    auto last_border = borders[start + 1].first;
    ControlOrder(prior_border, last_border, tags);
    start++;
  }
}

BorderType JudgeBorderType(const CNodePtr &border) {
  if (IsPrimitiveCNode(border, prim::kPrimSend)) {
    return BorderType::kSend;
  }
  if (IsPrimitiveCNode(border, prim::kPrimReceive)) {
    return BorderType::kReceive;
  }
  auto input = border->input(0);
  if (IsValueNode<FuncGraph>(input)) {
    return BorderType::kCallForward;
  }
  if (input->isa<CNode>()) {
    return BorderType::kCallBackward;
  }
  return BorderType::kOther;
}

void LabelFor1b1fOverlap(const std::vector<BorderPair> &borders, const std::pair<size_t, size_t> &border_step4,
                         const std::pair<size_t, size_t> &border_step5) {
  auto start_step4 = border_step4.first;
  auto end_step5 = border_step5.second;

  size_t index_1f1b = 0;
  size_t inter_recv_index = start_step4;
  size_t prior_cell_index = start_step4;
  for (size_t i = start_step4; i < end_step5; i++) {
    const auto &cur = borders[i];
    auto type = JudgeBorderType(cur.first.border);
    if (type == BorderType::kCallBackward) {
      prior_cell_index = i;
    }
    if (type == BorderType::kReceive) {
      inter_recv_index = i;
    }
    if (type == BorderType::kCallForward) {
      const auto &prior_cell_border = borders[prior_cell_index].second.border;
      MS_EXCEPTION_IF_NULL(prior_cell_border);
      const auto &next_cell_border = cur.first.border;
      MS_EXCEPTION_IF_NULL(next_cell_border);
      prior_cell_border->AddAttr(kCNodeAttr1f1bIndexBp, MakeValue<size_t>(index_1f1b));
      next_cell_border->AddAttr(kCNodeAttr1f1bIndexFp, MakeValue<size_t>(index_1f1b));
      auto prior_recv = borders[prior_cell_index - 1];
      const auto &prior_recv_border = prior_recv.second.border;
      MS_EXCEPTION_IF_NULL(prior_recv_border);
      // only label advanced recv
      if (JudgeBorderType(prior_recv_border) == BorderType::kReceive && index_1f1b > 0) {
        Add1b1fReceiveAttr(prior_recv, kCNodeAttr1f1bIndexRecv, index_1f1b);
      }

      auto next_recv = borders[inter_recv_index];
      const auto &next_recv_border = next_recv.second.border;
      MS_EXCEPTION_IF_NULL(next_recv_border);
      // only label advanced recv
      if (inter_recv_index > prior_cell_index && index_1f1b > 0) {
        Add1b1fReceiveAttr(next_recv, kCNodeAttr1f1bIndexInterRecv, index_1f1b);
      }
      index_1f1b++;
    }
  }
}

BorderPair GetTargetBorderPair(const std::vector<BorderPair> &borders, bool is_reverse, BorderType type) {
  std::vector<BorderPair> sorted_borders;
  if (is_reverse) {
    sorted_borders.insert(sorted_borders.end(), borders.rbegin(), borders.rend());
  } else {
    sorted_borders.insert(sorted_borders.end(), borders.begin(), borders.end());
  }
  BorderPair null_pair;
  auto iter = std::find_if(sorted_borders.begin(), sorted_borders.end(),
                           [type](const auto &border) { return JudgeBorderType(border.first.border) == type; });
  if (iter != sorted_borders.end()) {
    return *iter;
  }
  MS_LOG(EXCEPTION) << "Get target border pair failed.";
}

bool JudgeInsertControlEdge(const CNodePtr &pre, const CNodePtr &cur) {
  const auto &pre_type = JudgeBorderType(pre);
  const auto &cur_type = JudgeBorderType(cur);
  // call_send
  if (pre_type == BorderType::kCallBackward && cur_type == BorderType::kSend) {
    return false;
  }
  // call_recv
  if (pre_type == BorderType::kCallBackward && cur_type == BorderType::kReceive) {
    return false;
  }
  // send_call
  if (pre_type == BorderType::kSend && cur_type == BorderType::kCallForward) {
    return false;
  }
  if (pre_type == BorderType::kCallBackward && cur_type == BorderType::kCallForward) {
    return false;
  }
  return true;
}

void ZeroBubbleV::ReorderInnerOverlap(const std::vector<BorderPair> &borders,
                                      const std::vector<std::pair<size_t, size_t>> &overlap_border,
                                      const std::pair<size_t, size_t> &border_step4,
                                      const std::pair<size_t, size_t> &border_step5) {
  auto start_step4 = border_step4.first;
  auto end_step5 = border_step5.second;
  auto pre_index = start_step4;
  for (size_t i = 0; i < overlap_border.size(); i++) {
    const auto &index = overlap_border[i];
    auto start_index = index.first;
    auto end_index = index.second;
    InsertControlOrder(borders, pre_index, start_index);
    pre_index = end_index;
  }
  InsertControlOrder(borders, overlap_border.back().second, end_step5);
  for (size_t i = 0; i < overlap_border.size(); i++) {
    const auto &index = overlap_border[i];
    auto start_index = index.first;
    auto end_index = index.second;
    const auto &start = borders[start_index];
    const auto &end = borders[end_index];
    ControlOrder(start.second, end.first, kPrimalAttr1b1fCallCall);
    if (end_index - start_index <= 1) {
      continue;
    }
    for (size_t j = start_index + 1; j <= end_index; j++) {
      const auto &pre_cur = borders[j - 1];
      const auto &cur = borders[j];
      const auto &pre_cur_border = pre_cur.second.border;
      const auto &cur_border = cur.first.border;
      if (JudgeInsertControlEdge(pre_cur_border, cur_border)) {
        ControlOrder(pre_cur.second, cur.first, "inner_overlap");
      }
      const auto &cur_border_type = JudgeBorderType(cur_border);
      auto insert_depend = [this, &cur](const auto &next_users) {
        for (const auto &next_user : next_users) {
          if (IsPrimitiveCNode(next_user.first, prim::kPrimDepend)) {
            ControlOrder(cur.second, {next_user.first->template cast<CNodePtr>(), 0, 0}, "send_out_1f1b");
          }
        }
      };
      if (cur_border_type == BorderType::kSend) {
        auto next_users = GetOutputNodesWithFilter(
          end.first.border, [&](const AnfNodePtr &anode) { return IsPrimitiveCNode(anode, prim::kPrimTupleGetItem); });
        insert_depend(next_users);
      }
      if (cur_border_type != BorderType::kReceive) {
        continue;
      }
      const auto &cell_inputs = start.second.border->inputs();
      std::for_each(cell_inputs.begin(), cell_inputs.end(), [this, cur](const auto &cell_input) {
        if (IsPrimitiveCNode(cell_input, prim::kPrimDepend)) {
          ControlOrder({cell_input->template cast<CNodePtr>(), 0, 0}, cur.first, "input_recv_1f1b");
        }
      });
    }
  }
}

void MarkDualPipePhase(const std::vector<BorderPair> &orders, const std::string &tag, size_t start, size_t end,
                       size_t phase_id) {
  for (size_t i = start; i < end; i++) {
    const auto &order = orders[i];
    const auto &first_border = order.first.border;
    if (first_border != nullptr) {
      first_border->AddAttr(tag, MakeValue<size_t>(phase_id));
    }
    const auto &second_border = order.second.border;
    if (second_border != nullptr) {
      second_border->AddAttr(tag, MakeValue<size_t>(phase_id));
    }
  }
}

std::vector<BorderPair> FilterExecOrder(const std::vector<BorderPair> &orders) {
  std::vector<BorderPair> unique_exec_order;
  std::copy_if(orders.begin(), orders.end(), std::back_inserter(unique_exec_order),
               [](const auto &item) { return item.first.border != nullptr && item.second.border != nullptr; });
  auto iter = std::unique(unique_exec_order.begin(), unique_exec_order.end(), CompareBorderPair);
  unique_exec_order.erase(iter, unique_exec_order.end());
  return unique_exec_order;
}

std::pair<size_t, size_t> FindBorderIndex(const std::vector<BorderPair> &orders, const std::string &phase_tag,
                                          size_t phase_id) {
  std::pair<size_t, size_t> index;
  bool has_find = false;
  for (size_t i = 0; i < orders.size(); i++) {
    const auto &border = orders[i].first.border;
    MS_EXCEPTION_IF_NULL(border);
    if (border->HasAttr(phase_tag) && GetValue<size_t>(border->GetAttr(phase_tag)) == phase_id) {
      if (!has_find) {
        index.first = i;
        has_find = true;
      }
      index.second = i;
    }
  }
  if (!has_find) {
    MS_LOG(EXCEPTION) << "Has not find border index, phase_id: " << phase_id;
  }
  index.second++;
  return index;
}

bool ZeroBubbleV::IsDetachedBackward(int64_t chunk, int64_t micro) {
  return std::any_of(need_detach_info_.begin(), need_detach_info_.end(),
                     [chunk, micro](const auto &info) { return info.chunk == chunk && info.micro == micro; });
}

void ZeroBubbleV::GetBackwardBorder(const CNodePtr &cnode) {
  auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
  auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
  Border border = {cnode, chunk, micro};
  Border border_cell = {nullptr, chunk, micro};
  auto node_users_map = manager_->node_users();

  if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
    auto bwd_cell = GetCellBySend(cnode);
    MS_EXCEPTION_IF_NULL(bwd_cell);
    if ((stage_ == stage_num_ - 1 && chunk == 0) || (stage_ == 0 && chunk == 1)) {
      Border bwd_begin = {bwd_cell, chunk, micro};
      bwd_begin_.emplace_back(bwd_begin);
      border_cell.border = bwd_cell;
      bwd_cell_.emplace_back(border_cell);
    }
    bwd_end_.emplace_back(border);
  }
  if (cnode->HasPrimalAttr(PIPELINE_END)) {
    auto bwd_cell = GetCellByReceive(cnode, manager_);
    MS_EXCEPTION_IF_NULL(bwd_cell);
    if ((stage_ == 0 && chunk == 0) || (stage_ == stage_num_ - 1 && chunk == 1)) {
      Border bwd_end = {bwd_cell, chunk, micro};
      bwd_end_.emplace_back(bwd_end);
    }
    border_cell.border = bwd_cell;
    bwd_cell_.emplace_back(border_cell);
    bwd_begin_.emplace_back(border);
  }
  auto dw_node = GetDwBorder(border_cell, node_users_map);
  if (dw_node != nullptr) {
    auto dw_cnode = dw_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dw_cnode);
    if (dw_cnode->HasPrimalAttr(VISITED)) {
      return;
    }
    dw_cnode->AddPrimalAttr(VISITED, MakeValue(true));
    Border dw_border = {dw_cnode, chunk, micro};
    dw_border_.emplace_back(std::make_pair(dw_border, dw_border));
  }
  if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
    bwd_params_.emplace_back(border);
  }
}

AnfNodePtr ZeroBubbleV::GetDwBorder(const Border &bwd_cell, const NodeUsersMap &node_users_map) {
  auto cnode = bwd_cell.border;
  if (cnode == nullptr) {
    return nullptr;
  }
  auto tuple_get_item = cnode->input(0)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  auto closure_call = tuple_get_item->input(1);
  auto call_users = node_users_map.at(closure_call);
  for (const auto &call_user : call_users) {
    auto cuser = call_user.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cuser, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto index = GetTupleGetItemIndex(cuser);
    if (index != DW_INDEX) {
      continue;
    }
    auto dw_call = node_users_map.at(cuser).front();
    if (dw_call.second != 0) {
      MS_LOG(EXCEPTION) << "Can't find dw call node.";
    }
    return node_users_map.at(cuser).front().first;
  }
  return nullptr;
}

void ZeroBubbleV::GetBorderNode() {
  auto all_nodes = TopoSort(root_->get_return(), SuccDeeperSimple);
  GetChunkNumMicroSize(all_nodes);
  need_detach_info_ = InferNeedDetachInfo(stage_, micro_size_);
  if (chunk_num_ != kIndexTwo) {
    MS_LOG(EXCEPTION) << "Zero Bubble V scheduler only support chunk_num is 2, but got:" << chunk_num_;
  }
  for (const auto &node : all_nodes) {
    if (!IsSendOrReceive(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!CheckChunkAndMicro(cnode)) {
      continue;
    }
    if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      if (cnode->HasPrimalAttr(FREEZE)) {
        auto freeze_v = cnode->GetPrimalAttr(FREEZE);
        if (GetValue<bool>(freeze_v)) {
          continue;
        }
      }
      GetBackwardBorder(cnode);
      continue;
    }
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
    Border border = {cnode, chunk, micro};
    Border border_cell = {nullptr, chunk, micro};
    if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
      auto fwd_cell = GetCellByReceive(cnode, manager_);
      MS_EXCEPTION_IF_NULL(fwd_cell);
      if ((chunk == 1 && stage_ == 0) || (chunk == 0 && stage_ == stage_num_ - 1)) {
        Border fwd_end = {fwd_cell, chunk, micro};
        fwd_end_.emplace_back(fwd_end);
      }
      border_cell.border = fwd_cell;
      fwd_cell_.emplace_back(border_cell);
      fwd_begin_.emplace_back(border);
      continue;
    }

    if (cnode->HasPrimalAttr(PIPELINE_END)) {
      auto fwd_cell = GetCellBySend(cnode);
      MS_EXCEPTION_IF_NULL(fwd_cell);
      if ((stage_ == 0 && chunk == 0) || (stage_ == stage_num_ - 1 && chunk == 1)) {
        Border fwd_begin = {fwd_cell, chunk, micro};
        fwd_begin_.emplace_back(fwd_begin);
        border_cell.border = fwd_cell;
        fwd_cell_.emplace_back(border_cell);
      }
      fwd_end_.emplace_back(border);
      continue;
    }
    if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      fwd_params_.emplace_back(border);
      continue;
    }
  }
}

std::queue<BorderPair> ZeroBubbleV::GetTargetBorder(const std::vector<BorderPair> &ori_border, int64_t chunk) {
  std::queue<BorderPair> border_q;
  std::vector<BorderPair> border_v;
  std::copy_if(ori_border.begin(), ori_border.end(), std::back_inserter(border_v),
               [chunk](const auto &border) { return border.first.chunk == chunk; });
  std::sort(border_v.begin(), border_v.end(),
            [](BorderPair a, BorderPair b) -> bool { return a.first.micro < b.first.micro; });

  for (const auto &border : border_v) {
    border_q.push(border);
  }
  return border_q;
}

void ZeroBubbleV::ReorderShardedParam(const BorderVecPtr &exec_order) {
  if (!fwd_params_.empty()) {
    std::sort(fwd_params_.begin(), fwd_params_.end(), SortFuncInsideMicro);
    auto prior = fwd_params_.back();
    auto last = exec_order->front().first;
    ControlOrder(prior, last);
  }
  if (!bwd_params_.empty()) {
    std::sort(bwd_params_.begin(), bwd_params_.end(), SortFuncInsideMicro);
    auto prior2 = exec_order->back().second;
    auto last2 = bwd_params_.front();
    ControlOrder(prior2, last2);
  }
}

void ZeroBubbleV::ProcessStep1(const PipelineState &state, BorderVecPtr exec_order) {
  // step1: nF0
  const int64_t steps = (stage_num_ - stage_ - 1) * 2;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, fwd_b_ph0_);
    state.SafeAdd(exec_order, fwd_c_ph0_);
    state.SafeAdd(exec_order, fwd_e_ph0_);
  }
}

void ZeroBubbleV::ProcessStep2(const PipelineState &state, BorderVecPtr exec_order) {
  // step2: nF0F1
  const int64_t steps = stage_ + 1;
  state.SafeAdd(exec_order, fwd_b_ph0_);
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, fwd_c_ph0_);
    state.CondAdd(exec_order, fwd_b_ph0_, !state.is_first_stage);
    state.SafeAdd(exec_order, fwd_b_ph1_);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.CondAdd(exec_order, fwd_e_ph1_, !state.is_first_stage);
    state.CondAdd(exec_order, fwd_e_ph0_, !state.is_last_stage);
  }
}

void ZeroBubbleV::ProcessStep3(const PipelineState &state, BorderVecPtr exec_order) {
  // step3: nB1W1F1
  const int64_t steps = stage_num_ - stage_ - 1;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, bwd_b_ph1_);
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_b_ph1_);
    state.SafeAdd(exec_order, bwd_e_ph1_);
    state.SafeAdd(exec_order, dw_ph1_);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_e_ph1_);
  }
}

void ZeroBubbleV::ProcessStep4(const PipelineState &state, BorderVecPtr exec_order) {
  // step4: nB1F0B0F1
  const int64_t steps = micro_size_ - stage_num_ * 2 + stage_ + 1;
  for (int64_t i = 0; i < steps; ++i) {
    // B1F0
    if (i != 0) {
      state.CondAdd(exec_order, fwd_b_ph0_, !state.is_first_stage);
    } else {
      state.SafeAdd(exec_order, bwd_b_ph1_);
    }
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.CondAdd(exec_order, bwd_b_ph0_, !state.is_last_stage);
    state.CondAdd(exec_order, bwd_e_ph1_, !state.is_last_stage);
    state.SafeAdd(exec_order, fwd_c_ph0_);
    state.CondAdd(exec_order, fwd_b_ph1_, !state.is_last_stage);
    state.CondAdd(exec_order, fwd_e_ph0_, !state.is_last_stage);

    // B0F1
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.CondAdd(exec_order, bwd_e_ph0_, !state.is_first_stage);
    state.CondAdd(exec_order, bwd_b_ph1_, !state.is_first_stage);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_e_ph1_);
  }
}

void ZeroBubbleV::ProcessStep5(const PipelineState &state, BorderVecPtr exec_order) {
  // step5: nB1B0F1
  const int64_t steps = stage_num_ - stage_ - 1;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.SafeAdd(exec_order, bwd_b_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph1_);
    state.SafeAdd(exec_order, fwd_b_ph1_);
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph0_);
    state.CondAdd(exec_order, bwd_b_ph1_, !state.is_first_stage);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_e_ph1_);
  }
}

void ZeroBubbleV::ProcessStep6(const PipelineState &state, BorderVecPtr exec_order) {
  // step6: nB1B0
  const int64_t steps = stage_ + 1;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.CondAdd(exec_order, bwd_b_ph0_, !state.is_last_stage);
    state.SafeAdd(exec_order, bwd_e_ph1_);
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph0_);
    if (i != steps - 1) {
      state.SafeAdd(exec_order, bwd_b_ph1_);
    }
  }
}

void ZeroBubbleV::ProcessStep7(const PipelineState &state, BorderVecPtr exec_order) {
  // step7: nWB0
  const int64_t steps = stage_num_ - stage_ - 1;
  for (int64_t i = 0; i < steps; ++i) {
    if (i >= state.offset_int) {
      state.SafeAdd(exec_order, dw_ph0_);
    }
    state.SafeAdd(exec_order, bwd_b_ph0_);
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph0_);
  }
}

void ZeroBubbleV::ProcessStep8(const PipelineState &state, BorderVecPtr exec_order) {
  // step8: nW
  while (!dw_ph0_->empty()) {
    state.SafeAdd(exec_order, dw_ph0_);
  }
  while (!dw_ph1_->empty()) {
    state.SafeAdd(exec_order, dw_ph1_);
  }
}

void ZeroBubbleV::ReorderFor1b1fOverlap(const std::vector<BorderPair> borders,
                                        const std::pair<size_t, size_t> &border_step4,
                                        const std::pair<size_t, size_t> &border_step5) {
  // before: recv, fwd_cell
  // after: backward, send
  auto start_step4 = border_step4.first;
  auto end_step4 = border_step4.second;
  auto start_step5 = border_step5.first;
  auto end_step5 = border_step5.second;
  std::vector<BorderPair> before_overlap(borders.begin(), borders.begin() + start_step4);
  std::vector<BorderPair> after_overlap(borders.begin() + end_step5, borders.end());
  std::vector<BorderPair> after_step4(borders.begin() + start_step5, borders.end());
  auto pre_cell_step4 = GetTargetBorderPair(before_overlap, true, BorderType::kCallForward);

  std::vector<BorderPair> borders_step4(borders.begin() + start_step4, borders.begin() + end_step4);
  std::vector<BorderPair> borders_step5(borders.begin() + start_step5, borders.begin() + end_step5);

  BorderPair next_cell_step4 = GetTargetBorderPair(after_step4, false, BorderType::kCallBackward);
  BorderPair pre_cell_step5 = GetTargetBorderPair(borders_step4, true, BorderType::kCallForward);
  BorderPair next_cell_step5 = GetTargetBorderPair(after_overlap, false, BorderType::kCallBackward);

  std::vector<BorderPair> call_call_step4{pre_cell_step4};
  std::vector<BorderPair> call_call_step5{pre_cell_step5};

  bool is_step4 = true;

  auto call_call_control = [&call_call_step4, &call_call_step5](const BorderPair &cur, bool is_step4) {
    if (is_step4) {
      call_call_step4.push_back(cur);
    } else {
      call_call_step5.push_back(cur);
    }
  };
  size_t overlap_bp_index = 0;
  std::vector<std::pair<size_t, size_t>> overlap_border_index;
  MS_LOG(INFO) << "ZeroBubbleV::ReorderFor1b1fOverlap: start_step4: " << start_step4 << ", end_step5: " << end_step5;
  for (size_t i = start_step4; i < end_step5; i++) {
    if (i >= start_step5) {
      is_step4 = false;
    }
    const auto &cur = borders[i];
    auto type = JudgeBorderType(cur.first.border);
    if (type == BorderType::kCallBackward) {
      overlap_bp_index = i;
      call_call_control(cur, is_step4);
    }
    if (type == BorderType::kCallForward) {
      overlap_border_index.push_back(std::make_pair(overlap_bp_index, i));
      call_call_control(cur, is_step4);
    }
  }
  InsertControlOrder(borders, 0, start_step4, "before_overlap");
  ReorderInnerOverlap(borders, overlap_border_index, border_step4, border_step5);
  call_call_step4.push_back(next_cell_step4);
  call_call_step5.push_back(next_cell_step5);
  InsertCallControlOrder(call_call_step4, "call_call_1f1b");
  InsertCallControlOrder(call_call_step5, "call_call_1f1b");
  InsertControlOrder(borders, end_step5, borders.size() - 1, "after_overlap");
}

void ZeroBubbleV::Reorder() {
  auto fwd_begin = SortInsideMicro(fwd_begin_);
  auto fwd_end = SortInsideMicro(fwd_end_);
  auto fwd_cell = SortInsideMicro(fwd_cell_);
  auto bwd_begin = SortInsideMicro(bwd_begin_);
  auto bwd_end = SortInsideMicro(bwd_end_);
  auto bwd_cell = SortInsideMicro(bwd_cell_);

  fwd_b_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_begin, 0));
  fwd_c_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_cell, 0));
  fwd_e_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_end, 0));
  bwd_b_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_begin, 0));
  bwd_c_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_cell, 0));
  bwd_e_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_end, 0));
  dw_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(dw_border_, 0));

  fwd_b_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_begin, 1));
  fwd_c_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_cell, 1));
  fwd_e_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_end, 1));
  bwd_b_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_begin, 1));
  bwd_c_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_cell, 1));
  bwd_e_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_end, 1));
  dw_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(dw_border_, 1));
  PipelineState state(this);
  BorderVecPtr exec_order = std::make_shared<std::vector<BorderPair>>();

  size_t start_index = 0;
  const std::string phase_tag = "dual_pipe_phase";
  ProcessStep1(state, exec_order);
  size_t end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex1);

  start_index = exec_order->size();
  ProcessStep2(state, exec_order);
  end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex2);

  start_index = exec_order->size();
  ProcessStep3(state, exec_order);
  end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex3);

  start_index = exec_order->size();
  ProcessStep4(state, exec_order);
  end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex4);

  start_index = exec_order->size();
  ProcessStep5(state, exec_order);
  end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex5);

  start_index = exec_order->size();
  ProcessStep6(state, exec_order);
  end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex6);

  start_index = exec_order->size();
  ProcessStep7(state, exec_order);
  end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex7);

  start_index = exec_order->size();
  ProcessStep8(state, exec_order);
  ReorderShardedParam(exec_order);

  end_index = exec_order->size();
  MarkDualPipePhase(*exec_order, phase_tag, start_index, end_index, kIndex8);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<std::string>(MS_CTX_PP_1F1B_OVERLAP).empty()) {
    for (size_t i = 0; i < exec_order->size() - 1; ++i) {
      auto prior_border = exec_order->at(i).second;
      auto last_border = exec_order->at(i + 1).first;
      ControlOrder(prior_border, last_border);
    }
    return;
  }
  const auto &unique_exec_order = FilterExecOrder(*exec_order);
  std::pair<size_t, size_t> border_step4 = FindBorderIndex(unique_exec_order, phase_tag, kIndex4);
  std::pair<size_t, size_t> border_step5;
  bool is_last_stage = ((stage_num_ - 1) == stage_);
  if (is_last_stage) {
    border_step5.first = border_step4.second;
    border_step5.second = border_step4.second;
  } else {
    border_step5 = FindBorderIndex(unique_exec_order, phase_tag, kIndex5);
  }
  LabelFor1b1fOverlap(unique_exec_order, border_step4, border_step5);
  ReorderFor1b1fOverlap(unique_exec_order, border_step4, border_step5);
}

SchedulerRegisterAction PipelineSchedulerZeroBubbleV(parallel::kPipelineZeroBubbleV,
                                                     [](const FuncGraphManagerPtr &manager, const FuncGraphPtr &root,
                                                        int64_t stage, int64_t stage_num) {
                                                       return std::make_shared<parallel::ZeroBubbleV>(manager, root,
                                                                                                      stage, stage_num);
                                                     });
}  // namespace parallel
}  // namespace mindspore
