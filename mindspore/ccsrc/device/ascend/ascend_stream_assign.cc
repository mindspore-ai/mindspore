/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "device/ascend/ascend_stream_assign.h"

#include <algorithm>
#include <utility>

#include "ir/manager.h"
#include "utils/context/ms_context.h"
#include "common/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_adjust.h"
#include "predict/generator/utils/ir_model_util.h"
#include "pre_activate/common/helper.h"
#include "utils/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
const uint32_t kHcomMaxTask = 5;
const uint32_t kCommonMaxTask = 350;

void AscendStreamAssign::AssignStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (IsTaskSink()) {
    Reset();
    ReorderIndependentOrders(graph_ptr);
    AssignAllNodesStream(graph_ptr);
    UpdateAtomicAddrCleanStreamId(graph_ptr);
    InsertStreamActive(graph_ptr);
    InsertEventForHcomParallel(graph_ptr);
    InsertEventForIndependentParallel(graph_ptr);
    GetNeedActiveStreams(graph_ptr);
    graph_ptr->PrintGraphExecuteOrder();
    CheckResourceAssign(graph_ptr);
    MS_LOG(INFO) << "After finish stream assign";

    // Get info for D Model
    AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
    generator::IRModelUtil::GetInstance().set_event_num(resource_manager.get_cur_event_num());
    generator::IRModelUtil::GetInstance().set_stream_num(resource_manager.get_cur_stream_num());
    // Init to 1,temporarily
    generator::IRModelUtil::GetInstance().set_batch_num(1);
  }
}

// section 1
void AscendStreamAssign::ReorderIndependentOrders(const NotNull<KernelGraphPtr> &graph_ptr) {
  std::vector<CNodePtr> exe_orders;
  std::vector<CNodePtr> independents;
  std::vector<CNodePtr> others;

  auto cnode_ptr_list = graph_ptr->execution_order();
  MS_LOG(INFO) << "Before reorder, graph orders size:" << cnode_ptr_list.size();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    auto cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (IsIndependentNode(cur_cnode_ptr)) {
      independents.emplace_back(cur_cnode_ptr);
    } else {
      others.emplace_back(cur_cnode_ptr);
    }
  }

  if (others.empty() || independents.empty()) {
    MS_LOG(INFO) << "Independent or others is empty, no need reorder";
    return;
  }

  std::set<CNode *> processed;
  for (size_t i = 0; i < others.size(); i++) {
    auto begin = others.begin() + i;
    auto end = begin + 1;
    bool flag = false;
    for (size_t j = 0; j < independents.size(); j++) {
      auto cur_independent = independents[j];
      auto it = std::find(processed.begin(), processed.end(), cur_independent.get());
      if (it != processed.end()) {
        continue;
      }

      auto res = FindTargetOp(begin, end, cur_independent);
      if (res != end) {
        flag = true;
        exe_orders.emplace_back(cur_independent);
        exe_orders.emplace_back(*begin);
        processed.emplace(cur_independent.get());
        break;
      }
    }

    if (!flag) {
      exe_orders.emplace_back(*begin);
    }
  }

  MS_LOG(INFO) << "After reorder, graph orders size:" << exe_orders.size();
  if (processed.size() != independents.size()) {
    MS_LOG(WARNING) << "Processed independent nodes size is not equal to exiting independent nodes size";
    return;
  }

  graph_ptr->set_execution_order(exe_orders);
}

// section 2
void AscendStreamAssign::AssignAllNodesStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto cnode_ptr_list = graph_ptr->execution_order();
  bool exit_independent = false;
  bool exit_hcom = false;
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    // node has been assigned stream before
    if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
      continue;
    }

    if (IsHcom(cur_cnode_ptr)) {
      exit_hcom = true;
      continue;
    }

    if (IsIndependentNode(cur_cnode_ptr)) {
      exit_independent = true;
      continue;
    }

    AssignCommonStreamId(cur_cnode_ptr);
  }
  MS_LOG(INFO) << "Common start from 0, common stream nums:" << resource_manager.get_cur_stream_num();

  if (exit_hcom) {
    uint32_t first_hcom_stream_id = resource_manager.ApplyNewStream();
    for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
      CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
      // node has been assigned stream before
      if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
        continue;
      }

      if (IsHcom(cur_cnode_ptr)) {
        AssignHcomStreamId(cur_cnode_ptr);
      }
    }
    MS_LOG(INFO) << "Hcom start from :" << first_hcom_stream_id << ", hcom stream nums:" << hcom_stream_map_.size();
  }

  if (exit_independent) {
    uint32_t first_independ = resource_manager.ApplyNewStream();
    for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
      CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
      if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
        continue;
      }
      if (IsIndependentNode(cur_cnode_ptr)) {
        AssignIndependentStreamId(cur_cnode_ptr);
      }
    }
    MS_LOG(INFO) << "Independ start from:" << first_independ << ", stream nums:" << independent_stream_map_.size();
  }

  MS_LOG(INFO) << "After stream assign, total stream nums:" << resource_manager.get_cur_stream_num();
}

void AscendStreamAssign::AssignCommonStreamId(const CNodePtr &cur_cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  uint32_t cur_common_stream_id = 0;
  uint32_t cur_stream_num = resource_manager.get_cur_stream_num();
  if (cur_stream_num == 0) {
    cur_common_stream_id = resource_manager.ApplyNewStream();
  } else {
    cur_common_stream_id = resource_manager.GetCurAllocStreamId();
  }

  auto it = common_stream_map_.find(cur_common_stream_id);
  if (it == common_stream_map_.end()) {
    AnfAlgo::SetStreamId(cur_common_stream_id, cur_cnode_ptr.get());
    common_stream_map_.insert(std::make_pair(cur_common_stream_id, 1));
  } else {
    if (it->second < kCommonMaxTask) {
      AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
      it->second++;
    } else {
      cur_common_stream_id = resource_manager.ApplyNewStream();
      AnfAlgo::SetStreamId(cur_common_stream_id, cur_cnode_ptr.get());
      common_stream_map_.insert(std::make_pair(cur_common_stream_id, 1));
    }
  }
}

void AscendStreamAssign::AssignHcomStreamId(const CNodePtr &cur_cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  uint32_t cur_hcom_stream_id = resource_manager.GetCurAllocStreamId();
  auto it = hcom_stream_map_.find(cur_hcom_stream_id);
  if (it == hcom_stream_map_.end()) {
    AnfAlgo::SetStreamId(cur_hcom_stream_id, cur_cnode_ptr.get());
    hcom_stream_map_.insert(std::make_pair(cur_hcom_stream_id, 1));
  } else {
    if (it->second < kHcomMaxTask) {
      AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
      it->second++;
    } else {
      cur_hcom_stream_id = resource_manager.ApplyNewStream();
      AnfAlgo::SetStreamId(cur_hcom_stream_id, cur_cnode_ptr.get());
      hcom_stream_map_.insert(std::make_pair(cur_hcom_stream_id, 1));
    }
  }
}

void AscendStreamAssign::AssignIndependentStreamId(const CNodePtr &cur_cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  uint32_t cur_independent_id = resource_manager.GetCurAllocStreamId();
  auto it = independent_stream_map_.find(cur_independent_id);
  if (it == independent_stream_map_.end()) {
    AnfAlgo::SetStreamId(cur_independent_id, cur_cnode_ptr.get());
    independent_stream_map_.insert(std::make_pair(cur_independent_id, 1));
  } else {
    if (it->second < kCommonMaxTask) {
      AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
      it->second++;
    } else {
      cur_independent_id = resource_manager.ApplyNewStream();
      AnfAlgo::SetStreamId(cur_independent_id, cur_cnode_ptr.get());
      independent_stream_map_.insert(std::make_pair(cur_independent_id, 1));
    }
  }
}

bool AscendStreamAssign::IsIndependentNode(const CNodePtr &node_ptr) {
  MS_EXCEPTION_IF_NULL(node_ptr);
  if (AnfAlgo::GetKernelType(node_ptr) != AICPU_KERNEL) {
    return false;
  }

  if (AnfAlgo::GetCNodeName(node_ptr) == kGetNextOpName) {
    MS_LOG(INFO) << "GetNext should not be independent node";
    return false;
  }

  uint32_t input_nums = AnfAlgo::GetInputTensorNum(node_ptr);
  if (input_nums == 0) {
    MS_LOG(INFO) << "Node " << node_ptr->fullname_with_scope() << " is independent, as inputs nums is zero";
    return true;
  }

  auto inputs = node_ptr->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    if (!inputs[i]->isa<ValueNode>()) {
      return false;
    }
  }
  MS_LOG(INFO) << "Node " << node_ptr->fullname_with_scope() << " is independent, as inputs is all value node";
  return true;
}

// section 3:
void AscendStreamAssign::UpdateAtomicAddrCleanStreamId(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    // update AtomicAddrClean stream same witch the next node
    if (i > 0 && AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]) == kAtomicAddrCleanOpName) {
      AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(cur_cnode_ptr), cnode_ptr_list[i - 1].get());
    }
  }
  MS_LOG(INFO) << "End";
}

// section 4
void AscendStreamAssign::InsertStreamActive(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  GetProcessedStream(graph_ptr);
  std::vector<CNodePtr> update_cnode_list;
  CNodePtr cur_cnode_ptr = nullptr;
  CNodePtr pre_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;

  bool independent_flag = !(independent_stream_map_.empty());
  bool hcom_flag = !(hcom_stream_map_.empty());
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (IsIndependentNode(cur_cnode_ptr)) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (IsHcom(cur_cnode_ptr)) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    bool processed = IsProcessedStream(cur_stream_id);
    // 1)inner stream assign, need insert active op
    if (!processed) {
      MS_LOG(INFO) << "Common stream active info:" << pre_stream_id << "->active" << cur_stream_id;
      CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
      // 1.set stream id
      AnfAlgo::SetStreamId(pre_stream_id, active_ptr.get());
      // 2.set active stream ids
      std::vector<uint32_t> active_index_list{cur_stream_id};
      AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);
      update_cnode_list.emplace_back(active_ptr);
    }

    if ((independent_flag || hcom_flag) && (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName)) {
      MS_LOG(INFO) << "Insert StreamActive op after FP StreamSwitch for stream parallel";
      UpdateStreamSwitch(graph_ptr, cur_cnode_ptr, &update_cnode_list);
    } else {
      update_cnode_list.emplace_back(cur_cnode_ptr);
    }

    processed_streams_.emplace(cur_stream_id);
    pre_stream_id = cur_stream_id;
    pre_cnode_ptr = cur_cnode_ptr;
  }
  graph_ptr->set_execution_order(update_cnode_list);
  MS_LOG(INFO) << "End";
}

void AscendStreamAssign::GetProcessedStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  // 0 stream is activated at first
  processed_streams_.emplace(0);
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    auto cur_cnode_ptr = cnode_ptr_list[i];
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);

    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName) {
      auto primitive = AnfAlgo::GetCNodePrimitive(cur_cnode_ptr);
      MS_EXCEPTION_IF_NULL(primitive);
      auto true_stream_id = GetValue<uint32_t>(primitive->GetAttr(kAttrTrueBranchStream));
      processed_streams_.emplace(true_stream_id);

      auto value_ptr = primitive->GetAttr(kStreamNeedActivedFirst);
      if (value_ptr == nullptr) {
        continue;
      }
      auto need_active = GetValue<bool>(value_ptr);
      if (need_active) {
        processed_streams_.emplace(cur_stream_id);
      }
    }
  }
  for (const auto &item : processed_streams_) {
    MS_LOG(INFO) << "Before active:" << item << " is been processed";
  }
}

void AscendStreamAssign::UpdateStreamSwitch(const NotNull<KernelGraphPtr> &graph_ptr, const CNodePtr &switch_ptr,
                                            vector<CNodePtr> *orders) {
  orders->emplace_back(switch_ptr);
  auto primitive = AnfAlgo::GetCNodePrimitive(switch_ptr);
  MS_EXCEPTION_IF_NULL(primitive);
  auto value_ptr = primitive->GetAttr(kStreamNeedActivedFirst);
  if (value_ptr == nullptr) {
    return;
  }

  auto need_active = GetValue<bool>(value_ptr);
  if (!need_active) {
    return;
  }

  MS_EXCEPTION_IF_NULL(switch_ptr);
  auto true_stream_id = GetValue<uint32_t>(primitive->GetAttr(kAttrTrueBranchStream));
  MS_LOG(INFO) << "Streamswtich stream id:" << AnfAlgo::GetStreamId(switch_ptr)
               << "; active stream id:" << true_stream_id;

  CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
  AnfAlgo::SetStreamId(true_stream_id, active_ptr.get());
  vector<uint32_t> active_ids;
  // active indepdent stream
  for (const auto &item : independent_stream_map_) {
    active_ids.emplace_back(item.first);
  }
  // active hcom stream
  for (const auto &item : hcom_stream_map_) {
    active_ids.emplace_back(item.first);
  }
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_ids), active_ptr);

  // update processed stream
  independent_stream_activated_ = true;
  for (const auto &item : independent_stream_map_) {
    processed_streams_.emplace(item.first);
  }

  hcom_stream_activated_ = true;
  for (const auto &item : hcom_stream_map_) {
    processed_streams_.emplace(item.first);
  }

  orders->emplace_back(active_ptr);
}

bool AscendStreamAssign::IsProcessedStream(uint32_t stream_id) {
  auto it = std::find(processed_streams_.begin(), processed_streams_.end(), stream_id);
  if (it != processed_streams_.end()) {
    return true;
  }
  return false;
}

// section5
void AscendStreamAssign::InsertEventForHcomParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  InsertEventCommonDependHcom(graph_ptr);
  InsertEventHcomDependCommon(graph_ptr);
  InsertEventHcomDependHcom(graph_ptr);
  MS_LOG(INFO) << "End";
}

void AscendStreamAssign::InsertEventCommonDependHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  auto it = cnodes.begin();
  while (it != cnodes.end() && (it + 1) != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    MS_EXCEPTION_IF_NULL(*(it + 1));
    if (IsHcom(*it) && !IsHcom(*(it + 1))) {
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));
      it = cnodes.insert(it + 1, send_cnode_ptr);

      auto target = FindTargetOp(it, cnodes.end(), *(it - 1));
      if (target == cnodes.end()) {
        MS_LOG(WARNING) << "Hcom node:" << (*(it - 1))->fullname_with_scope()
                        << ", can't find target for insert recv op, no insert send/recv";
        it = cnodes.erase(it);
        continue;
      }

      if (IsHcom(*target)) {
        it = cnodes.erase(it);
        continue;
      }

      // deal recv op
      uint32_t stream_id = AnfAlgo::GetStreamId(*target);
      CNodePtr recv_cnode_ptr = CreateRecvApplyKernel(graph_ptr, cur_event_id, stream_id);
      (void)cnodes.insert(target, recv_cnode_ptr);
      cur_event_id = resource_manager.ApplyNewEvent();
    }
    ++it;
  }
  // one event allocated additional, should delete
  resource_manager.DeleteEvent();
  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After common depend hcom, total event nums:" << resource_manager.get_cur_event_num();
}

void AscendStreamAssign::InsertEventHcomDependCommon(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes;
  CNodePtr cur_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (i == 0) {
      cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (!IsHcom(cur_cnode_ptr)) {
      cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (cur_stream_id == pre_stream_id) {
      cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (!IsHcom(cnode_ptr_list[i - 1])) {
      uint32_t cur_event_id = resource_manager.ApplyNewEvent();
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, pre_stream_id);
      cnodes.emplace_back(send);
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_stream_id);
      cnodes.emplace_back(recv);
      cnodes.emplace_back(cur_cnode_ptr);
    }
    pre_stream_id = cur_stream_id;
  }

  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After hcom depend common, total event nums:" << resource_manager.get_cur_event_num();
}

void AscendStreamAssign::InsertEventHcomDependHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  uint32_t first_hcom_stream = kInvalidStreamId;
  uint32_t last_hcom_stream = kInvalidStreamId;
  // key: stream id, value:hcom index
  std::map<uint32_t, vector<size_t>> hcom_index;
  for (size_t i = 0; i < cnode_ptr_list.size(); i++) {
    auto cur_cnode = cnode_ptr_list[i];
    if (!IsHcom(cur_cnode)) {
      continue;
    }
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode);
    auto it = hcom_index.find(cur_stream_id);
    if (it != hcom_index.end()) {
      hcom_index[cur_stream_id].emplace_back(i);
    } else {
      hcom_index[cur_stream_id] = {i};
    }

    // record first hcom stream id
    if (first_hcom_stream == kInvalidStreamId) {
      first_hcom_stream = cur_stream_id;
    }

    // record last hcom stream id
    if (cur_stream_id != last_hcom_stream) {
      last_hcom_stream = cur_stream_id;
    }
  }

  if (hcom_index.size() < 2) {
    MS_LOG(INFO) << "Different stream hcom size is less than 2, no need insert event between them";
    return;
  }
  InsertEventBetweenHcom(graph_ptr, hcom_index, first_hcom_stream, last_hcom_stream);
  MS_LOG(INFO) << "After hcom depend hcom, total event nums:" << resource_manager.get_cur_event_num();
}

void AscendStreamAssign::InsertEventBetweenHcom(const NotNull<KernelGraphPtr> &graph_ptr,
                                                const map<uint32_t, vector<size_t>> &hcom_index,
                                                uint32_t first_hcom_stream, uint32_t last_hcom_stream) {
  vector<CNodePtr> orders;
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  size_t first_stream_last_index = hcom_index.at(first_hcom_stream).back();
  size_t last_stream_first_index = hcom_index.at(last_hcom_stream).front();
  std::copy(cnode_ptr_list.begin(), cnode_ptr_list.begin() + first_stream_last_index, std::back_inserter(orders));
  for (size_t i = first_stream_last_index; i <= last_stream_first_index; i++) {
    auto cur_cnode = cnode_ptr_list[i];
    if (!IsSatisfiedHcom(hcom_index, cur_cnode, i)) {
      orders.emplace_back(cur_cnode);
      continue;
    }
    auto cur_hcom_stream_id = AnfAlgo::GetStreamId(cur_cnode);
    if (i == first_stream_last_index) {
      // first fusion hcom
      orders.emplace_back(cur_cnode);
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(send);
    } else if (i == last_stream_first_index) {
      // last fusion hcom
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(recv);
      orders.emplace_back(cur_cnode);
    } else {
      auto cur_stream_hcom_size = hcom_index.at(cur_hcom_stream_id).size();
      if (cur_stream_hcom_size == 1) {
        auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
        orders.emplace_back(recv);
        cur_event_id = resource_manager.ApplyNewEvent();
        orders.emplace_back(cur_cnode);
        auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
        orders.emplace_back(send);
      } else {
        // current stream, first hcom:add recv op
        if (i == hcom_index.at(cur_hcom_stream_id).front()) {
          auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
          orders.emplace_back(recv);
          cur_event_id = resource_manager.ApplyNewEvent();
          orders.emplace_back(cur_cnode);
        } else if (i == hcom_index.at(cur_hcom_stream_id).back()) {
          // current stream, last hcom:add send op
          orders.emplace_back(cur_cnode);
          auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
          orders.emplace_back(send);
        } else {
          // current stream, not first and last op
          orders.emplace_back(cur_cnode);
        }
      }
    }
  }
  std::copy(cnode_ptr_list.begin() + last_stream_first_index + 1, cnode_ptr_list.end(), std::back_inserter(orders));
  graph_ptr->set_execution_order(orders);
}

bool AscendStreamAssign::IsSatisfiedHcom(const std::map<uint32_t, vector<size_t>> &hcom_index, const CNodePtr &node_ptr,
                                         size_t index) {
  MS_EXCEPTION_IF_NULL(node_ptr);
  auto cur_hcom_stream_id = AnfAlgo::GetStreamId(node_ptr);
  auto it = hcom_index.find(cur_hcom_stream_id);
  if (it == hcom_index.end()) {
    return false;
  }
  auto iter = std::find(hcom_index.at(cur_hcom_stream_id).begin(), hcom_index.at(cur_hcom_stream_id).end(), index);
  if (iter == hcom_index.at(cur_hcom_stream_id).end()) {
    return false;
  }
  return true;
}

// section6
void AscendStreamAssign::InsertEventForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  auto it = cnodes.begin();
  while (it != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    if (IsIndependentNode(*it)) {
      MS_LOG(INFO) << "Deal independent op[" << (*it)->DebugString() << "]";
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));
      it = cnodes.insert(it + 1, send_cnode_ptr);

      auto target = FindTargetOp(it, cnodes.end(), *(it - 1));
      if (target == cnodes.end()) {
        MS_LOG(DEBUG) << "Independ node[" << (*(it - 1))->fullname_with_scope()
                      << "] can't find target for insert recv op, no insert send/recv";
        it = cnodes.erase(it);
        continue;
      }

      // deal recv op
      uint32_t stream_id = AnfAlgo::GetStreamId(*target);
      CNodePtr recv_cnode_ptr = CreateRecvApplyKernel(graph_ptr, cur_event_id, stream_id);
      (void)cnodes.insert(target, recv_cnode_ptr);
      cur_event_id = resource_manager.ApplyNewEvent();
    }
    ++it;
  }
  // one event allocated additional, should delete
  resource_manager.DeleteEvent();
  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After independent parallel, total event nums:" << resource_manager.get_cur_event_num();
  MS_LOG(INFO) << "End";
}

// section7
void AscendStreamAssign::GetNeedActiveStreams(const NotNull<KernelGraphPtr> &graph_ptr) {
  CNodePtr cur_cnode_ptr = nullptr;
  auto cnode_ptr_list = graph_ptr->execution_order();
  // 1)first stream 0 should be actived first;
  need_first_active_streams_.emplace_back(0);

  // 2)stream witch kStreamNeedActivedFirst attr should be actived;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    auto primitive = AnfAlgo::GetCNodePrimitive(cur_cnode_ptr);
    MS_EXCEPTION_IF_NULL(primitive);
    auto value_ptr = primitive->GetAttr(kStreamNeedActivedFirst);
    if (value_ptr == nullptr) {
      continue;
    }

    auto need_active = GetValue<bool>(value_ptr);
    if (need_active) {
      auto stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
      MS_LOG(INFO) << "Stream id:" << stream_id << " is need actived at first";
      need_first_active_streams_.push_back(stream_id);
    }
  }

  // 3)independent stream:if has not been activate, push to need active vector
  if (!independent_stream_activated_) {
    for (auto &item : independent_stream_map_) {
      need_first_active_streams_.emplace_back(item.first);
    }
  }

  // 4)hcom stream:if has not been activate, push to need active vector
  if (!hcom_stream_activated_) {
    for (auto &item : hcom_stream_map_) {
      need_first_active_streams_.emplace_back(item.first);
    }
  }
}

// section8
void AscendStreamAssign::CheckResourceAssign(const NotNull<KernelGraphPtr> &graph_ptr) {
  CheckStreamAssign(graph_ptr);
  CheckEventAssign(graph_ptr);
}

void AscendStreamAssign::CheckStreamAssign(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  std::set<uint32_t> streams;
  uint32_t max_stream = 0;
  uint32_t min_stream = kInvalidStreamId;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (stream_id == kInvalidStreamId) {
      MS_LOG(EXCEPTION) << "Node:" << AnfAlgo::GetCNodeName(cur_cnode_ptr) << "had not been assigned stream";
    }

    (void)streams.emplace(stream_id);
    if (stream_id > max_stream) {
      max_stream = stream_id;
    }
    if (stream_id < min_stream) {
      min_stream = stream_id;
    }
  }

  // check stream assign
  if (!streams.empty()) {
    if (min_stream != 0) {
      MS_LOG(EXCEPTION) << "Stream should start from 0, now is from " << min_stream;
    }
    uint32_t assigned_stream_num = resource_manager.get_cur_stream_num();
    if ((max_stream != assigned_stream_num - 1) || (streams.size() != assigned_stream_num)) {
      MS_LOG(EXCEPTION) << "Stream should be consecutive, max stream id:" << max_stream
                        << "; alloc stream nums:" << assigned_stream_num << "; streams size:" << streams.size();
    }
  }
}

void AscendStreamAssign::CheckEventAssign(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  std::map<uint32_t, std::vector<CNodePtr>> event_map;
  uint32_t max_event_id = 0;
  uint32_t min_event_id = kInvalidEventId;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    auto name = AnfAlgo::GetCNodeName(cur_cnode_ptr);
    if (name == kSendOpName || name == kRecvOpName) {
      uint32_t event_id = AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrEventId);
      if (event_id > max_event_id) {
        max_event_id = event_id;
      }

      if (event_id < min_event_id) {
        min_event_id = event_id;
      }
      auto it = event_map.find(event_id);
      if (it == event_map.end()) {
        event_map[event_id] = {cur_cnode_ptr};
      } else {
        event_map[event_id].emplace_back(cur_cnode_ptr);
      }
    }
  }
  // check event assign
  if (!event_map.empty()) {
    if (min_event_id != 0) {
      MS_LOG(EXCEPTION) << "Event should start from 0, now is from " << min_event_id;
    }
    uint32_t assigned_event_num = resource_manager.get_cur_event_num();
    if ((max_event_id != assigned_event_num - 1) || (event_map.size() != assigned_event_num)) {
      MS_LOG(EXCEPTION) << "Event should be consecutive";
    }
    for (const auto &item : event_map) {
      if (item.second.size() != 2) {
        MS_LOG(EXCEPTION) << "Send/recv should be in pair and share one event id";
      }
      auto first_name = AnfAlgo::GetCNodeName(item.second[0]);
      auto second_name = AnfAlgo::GetCNodeName(item.second[1]);
      if (!(first_name == kSendOpName && second_name == kRecvOpName)) {
        MS_LOG(EXCEPTION) << "Send should be before recv";
      }
    }
  }
}

// section9
CNodePtr AscendStreamAssign::CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                   uint32_t stream_id) {
  auto send_op = std::make_shared<Primitive>(kSendOpName);
  MS_EXCEPTION_IF_NULL(send_op);
  auto send_apply = std::make_shared<ValueNode>(send_op);
  MS_EXCEPTION_IF_NULL(send_apply);
  std::vector<AnfNodePtr> send_input_list = {send_apply};
  CNodePtr send_node_ptr = graph_ptr->NewCNode(send_input_list);
  MS_EXCEPTION_IF_NULL(send_node_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), send_node_ptr.get());
  AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), send_node_ptr);
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract_none);
  send_node_ptr->set_abstract(abstract_none);
  AnfAlgo::SetStreamId(stream_id, send_node_ptr.get());
  return send_node_ptr;
}

CNodePtr AscendStreamAssign::CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                   uint32_t stream_id) {
  auto recv_op = std::make_shared<Primitive>(kRecvOpName);
  MS_EXCEPTION_IF_NULL(recv_op);
  auto recv_apply = std::make_shared<ValueNode>(recv_op);
  MS_EXCEPTION_IF_NULL(recv_apply);
  std::vector<AnfNodePtr> recv_input_list = {recv_apply};
  CNodePtr recv_node_ptr = graph_ptr->NewCNode(recv_input_list);
  MS_EXCEPTION_IF_NULL(recv_node_ptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), recv_node_ptr.get());
  AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), recv_node_ptr);
  AnfAlgo::SetStreamId(stream_id, recv_node_ptr.get());
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract_none);
  recv_node_ptr->set_abstract(abstract_none);
  return recv_node_ptr;
}

vector<CNodePtr>::iterator AscendStreamAssign::FindTargetOp(vector<CNodePtr>::iterator begin,
                                                            vector<CNodePtr>::iterator end, const CNodePtr &node) {
  while (begin != end) {
    auto inputs = (*begin)->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      auto input = inputs[i];
      if (opt::IsNopNode(input)) {
        CNodePtr cnode = input->cast<CNodePtr>();
        auto new_inputs = cnode->inputs();
        for (size_t j = 1; j < new_inputs.size(); j++) {
          auto new_real_input = AnfAlgo::VisitKernel(new_inputs[j], 0);
          if (node == new_real_input.first) {
            MS_LOG(INFO) << "Nop node find target op[" << (*begin)->DebugString() << "]";
            return begin;
          }
        }
      } else {
        auto real_input = AnfAlgo::VisitKernel(input, 0);
        if (node == real_input.first) {
          MS_LOG(INFO) << "Find target op[" << (*begin)->DebugString() << "]";
          return begin;
        }
      }
    }
    ++begin;
  }
  return end;
}

bool AscendStreamAssign::IsTaskSink() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->enable_task_sink()) {
    MS_LOG(INFO) << "Task sink mode is not enable";
    return false;
  } else {
    MS_LOG(INFO) << "Task sink mode is enable";
    return true;
  }
}

void AscendStreamAssign::GetWaitStreams(vector<uint32_t> *wait_active_stream_list) {
  MS_EXCEPTION_IF_NULL(wait_active_stream_list);
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  uint32_t total_stream_num = resource_manager.get_cur_stream_num();
  if (total_stream_num == 0) {
    MS_LOG(INFO) << "The total_common_stream_num is zero";
    return;
  }

  // common stream:active first common stream
  for (uint32_t i = 0; i < total_stream_num; i++) {
    auto it = std::find(need_first_active_streams_.begin(), need_first_active_streams_.end(), i);
    if (it == need_first_active_streams_.end()) {
      MS_LOG(INFO) << "Wait common stream id = " << i;
      wait_active_stream_list->push_back(i);
    }
  }
}

bool AscendStreamAssign::IsHcom(const CNodePtr &apply_kernel) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  return AnfAlgo::GetKernelType(apply_kernel) == HCCL_KERNEL;
}

void AscendStreamAssign::GetHcomStreams(std::vector<uint32_t> *streams) {
  MS_EXCEPTION_IF_NULL(streams);
  for (const auto &item : hcom_stream_map_) {
    streams->emplace_back(item.first);
  }
}

void AscendStreamAssign::Reset() {
  independent_stream_activated_ = false;
  hcom_stream_activated_ = false;
  independent_stream_map_.clear();
  hcom_stream_map_.clear();
  common_stream_map_.clear();
  processed_streams_.clear();
  need_first_active_streams_.clear();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
