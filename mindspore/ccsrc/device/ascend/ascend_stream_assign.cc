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
const uint32_t kIndependFirstStreamId = 1024;

bool AscendStreamAssign::IsHcom(const CNodePtr &apply_kernel) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  return AnfAlgo::GetKernelType(apply_kernel) == HCCL_KERNEL;
}

void AscendStreamAssign::ResetNew() {
  total_common_stream_num_ = 0;
  total_independ_stream_num_ = 0;
  total_event_num_ = 0;
  first_physic_id_ = UINT32_MAX;
  first_logic_id_ = UINT32_MAX;
  independent_id_ = kIndependFirstStreamId;
  logic_to_independent_map_.clear();
  processed_logic_id_.clear();
  logic_to_physic_map_.clear();
  independent_before_physic_id_.clear();
  inner_parallel_streams_.clear();
  processed_parallel_streams_.clear();
  hcom_stream_list_.clear();
  need_first_active_streams_.clear();
}

void AscendStreamAssign::AssignIndependentStreamId(const CNodePtr &cur_cnode_ptr, uint32_t processing_logic_id) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  auto it = logic_to_independent_map_.find(processing_logic_id);
  if (it == logic_to_independent_map_.end()) {
    (void)logic_to_independent_map_.insert(std::make_pair(processing_logic_id, independent_id_));
    AnfAlgo::SetStreamId(independent_id_, cur_cnode_ptr.get());
    independent_id_++;
  } else {
    AnfAlgo::SetStreamId(it->second, cur_cnode_ptr.get());
  }

  if (first_physic_id_ == UINT32_MAX) {
    auto res = std::find(independent_before_physic_id_.begin(), independent_before_physic_id_.end(),
                         AnfAlgo::GetStreamId(cur_cnode_ptr));
    if (res == independent_before_physic_id_.end()) {
      independent_before_physic_id_.push_back(AnfAlgo::GetStreamId(cur_cnode_ptr));
    }
  }
}

void AscendStreamAssign::AssignCommonStreamId(const CNodePtr &cur_cnode_ptr, CNodePtr *pre_cnode_ptr,
                                              uint32_t *cur_index, uint32_t *cur_stream_id) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  MS_EXCEPTION_IF_NULL(*pre_cnode_ptr);
  bool over_max_hcom_task = (IsHcom(cur_cnode_ptr) && (*cur_index) % kHcomMaxTask == 0);
  bool over_max_common_task = (!IsHcom(cur_cnode_ptr) && (*cur_index) % kCommonMaxTask == 0);
  bool pre_common_cur_hcom = (IsHcom(cur_cnode_ptr) && !IsHcom(*pre_cnode_ptr));
  bool pre_hcom_cur_common = (!IsHcom(cur_cnode_ptr) && IsHcom(*pre_cnode_ptr));
  if (over_max_hcom_task || over_max_common_task || pre_common_cur_hcom || pre_hcom_cur_common) {
    *cur_index = 0;
    ++(*cur_stream_id);
  }

  if (over_max_hcom_task || pre_common_cur_hcom) {
    hcom_stream_list_.emplace_back(*cur_stream_id);
  }
  ++(*cur_index);
  AnfAlgo::SetStreamId(*cur_stream_id, cur_cnode_ptr.get());
  *pre_cnode_ptr = cur_cnode_ptr;
}

bool AscendStreamAssign::IsProcessed(uint32_t logic_id) {
  auto it = std::find(processed_logic_id_.begin(), processed_logic_id_.end(), logic_id);
  if (it == processed_logic_id_.end()) {
    return false;
  }

  return true;
}

void AscendStreamAssign::RecordIdMap(uint32_t logic_id, uint32_t physic_id) {
  auto it = logic_to_physic_map_.find(logic_id);
  if (it == logic_to_physic_map_.end()) {
    MS_LOG(INFO) << "New logic_id[" << logic_id << "] to physic_id[" << physic_id << "]";
    (void)logic_to_physic_map_.insert(std::make_pair(logic_id, physic_id));
  }
}

void AscendStreamAssign::RecordFirstCommonOp(const CNodePtr &cur_cnode_ptr, uint32_t cur_node_logic_id,
                                             uint32_t cur_stream_id) {
  AnfAlgo::SetStreamId(cur_stream_id, cur_cnode_ptr.get());
  RecordIdMap(cur_node_logic_id, cur_stream_id);
  first_physic_id_ = cur_stream_id;
  first_logic_id_ = cur_node_logic_id;
}

uint32_t AscendStreamAssign::GetLogicId(const CNodePtr &cur_cnode_ptr) {
  uint32_t logic_id = AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get());
  if (logic_id == kInvalidDistincLabel) {
    MS_LOG(EXCEPTION) << "node[" << cur_cnode_ptr->DebugString() << "] logic id is invalid";
  }
  return logic_id;
}

void AscendStreamAssign::SetCommonStreamNum(uint32_t cur_stream_id) {
  if (first_physic_id_ == UINT32_MAX) {
    MS_LOG(INFO) << "cur common node size is zero";
    total_common_stream_num_ = 0;
  } else {
    total_common_stream_num_ = cur_stream_id + 1;
  }
}

void AscendStreamAssign::AssignAllNodesStream(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto cnode_ptr_list = graph_ptr->execution_order();
  CNodePtr pre_cnode_ptr = nullptr;
  uint32_t cur_index = 0;
  uint32_t cur_stream_id = 0;
  uint32_t processing_logic_id = UINT32_MAX;

  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    // get logic id
    uint32_t cur_node_logic_id = GetLogicId(cur_cnode_ptr);
    if (IsIndependentNode(cur_cnode_ptr)) {
      AssignIndependentStreamId(cur_cnode_ptr, cur_node_logic_id);
      continue;
    }
    if (pre_cnode_ptr == nullptr) {
      RecordFirstCommonOp(cur_cnode_ptr, cur_node_logic_id, cur_stream_id);
      processing_logic_id = cur_node_logic_id;
      ++cur_index;
      pre_cnode_ptr = cur_cnode_ptr;
      continue;
    }

    // 1.has been processed
    if (IsProcessed(cur_node_logic_id)) {
      continue;
    }

    if (cur_node_logic_id == processing_logic_id) {
      AssignCommonStreamId(cur_cnode_ptr, &pre_cnode_ptr, &cur_index, &cur_stream_id);
    } else {
      // 1.find other same logic id
      for (size_t j = i; j < cnode_ptr_list.size(); ++j) {
        CNodePtr cnode_ptr = cnode_ptr_list[j];
        MS_EXCEPTION_IF_NULL(cnode_ptr);
        uint32_t logic_id = AnfAlgo::GetStreamDistinctionLabel(cnode_ptr.get());
        if (logic_id == processing_logic_id) {
          AssignCommonStreamId(cnode_ptr, &pre_cnode_ptr, &cur_index, &cur_stream_id);
        }
      }
      // 2.after deal:
      processed_logic_id_.push_back(processing_logic_id);
      cur_cnode_ptr = cnode_ptr_list[i];
      // 3. new stream
      ++cur_stream_id;
      AnfAlgo::SetStreamId(cur_stream_id, cur_cnode_ptr.get());
      cur_index = 1;

      pre_cnode_ptr = cur_cnode_ptr;
      processing_logic_id = cur_node_logic_id;
      RecordIdMap(processing_logic_id, cur_stream_id);
    }
  }

  SetCommonStreamNum(cur_stream_id);
  total_independ_stream_num_ = independent_id_ - kIndependFirstStreamId;
  MS_LOG(INFO) << "stream nums:common:" << total_common_stream_num_ << ",independ:" << total_independ_stream_num_;
}

void AscendStreamAssign::TransLogicToPhysic(const vector<uint32_t> &logic_ids, vector<uint32_t> *physic_ids) {
  for (auto &id : logic_ids) {
    auto it = logic_to_physic_map_.find(id);
    if (it != logic_to_physic_map_.end()) {
      MS_LOG(INFO) << "logic id[" << id << "] to physic id[" << it->second << "]";
      (*physic_ids).push_back(it->second);
    } else {
      MS_LOG(EXCEPTION) << "logic id[" << id << "] has no correspond physic id";
    }

    auto it_independ = logic_to_independent_map_.find(id);
    if (it_independ != logic_to_independent_map_.end()) {
      MS_LOG(INFO) << "logic id[" << id << "] to independent id[" << it_independ->second << "]";
      (*physic_ids).push_back(it_independ->second);
    }
  }
}

void AscendStreamAssign::UpdateStreamActive(const CNodePtr &active_ptr) {
  MS_LOG(INFO) << "start update outter active op[" << active_ptr->DebugString() << "] ";
  MS_EXCEPTION_IF_NULL(active_ptr);
  auto primitive = AnfAlgo::GetCNodePrimitive(active_ptr);
  MS_EXCEPTION_IF_NULL(primitive);
  vector<uint32_t> active_logic_ids = GetValue<std::vector<uint32_t>>(primitive->GetAttr(kAttrActiveStreamList));
  // out StreamAcitve active physic stream is not parallel now, if parallel, should deal here.
  vector<uint32_t> active_physic_ids;
  TransLogicToPhysic(active_logic_ids, &active_physic_ids);
  ValuePtr active_physic_value = MakeValue<std::vector<uint32_t>>(active_physic_ids);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, active_physic_value, active_ptr);
}

void AscendStreamAssign::UpdateStreamSwitch(const CNodePtr &switch_ptr, const CNodePtr &active_ptr) {
  MS_LOG(INFO) << "start update switch op[" << switch_ptr->DebugString() << "]";
  MS_EXCEPTION_IF_NULL(switch_ptr);
  MS_EXCEPTION_IF_NULL(active_ptr);
  auto primitive = AnfAlgo::GetCNodePrimitive(switch_ptr);
  MS_EXCEPTION_IF_NULL(primitive);
  auto true_logic_id = GetValue<uint32_t>(primitive->GetAttr(kAttrTrueBranchStream));
  MS_LOG(INFO) << "streamswtich stream id[" << AnfAlgo::GetStreamId(switch_ptr) << "], true_logic_id[" << true_logic_id
               << "]";
  vector<uint32_t> logic_ids{true_logic_id};
  vector<uint32_t> physic_ids;
  TransLogicToPhysic(logic_ids, &physic_ids);
  if (physic_ids.empty()) {
    MS_LOG(EXCEPTION) << "stream switch true logic id[" << true_logic_id << "] has no physical id";
  }
  ValuePtr true_index = MakeValue<uint32_t>(physic_ids[0]);
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, true_index, switch_ptr);

  MS_LOG(INFO) << "start update StreamActive op[" << active_ptr->DebugString() << "]";
  AnfAlgo::SetStreamId(physic_ids[0], active_ptr.get());
  vector<uint32_t> active_ids;
  for (size_t i = 0; i < physic_ids.size(); i++) {
    if (i == 0) {
      MS_LOG(INFO) << "StreamActive op self stream id[" << physic_ids[i] << "]";
    } else {
      MS_LOG(INFO) << "StreamActive op active stream id[" << physic_ids[i] << "]";
      active_ids.emplace_back(physic_ids[i]);
    }
  }
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_ids), active_ptr);
}

void AscendStreamAssign::FindAllReduceParallel(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  CNodePtr cur_cnode_ptr = nullptr;
  CNodePtr pre_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (uint32_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (i == 0) {
      pre_cnode_ptr = cur_cnode_ptr;
      pre_stream_id = cur_stream_id;
      continue;
    }

    bool diff_stream = (pre_stream_id != cur_stream_id) && (pre_stream_id < cur_stream_id);
    bool pre_hcom = IsHcom(pre_cnode_ptr);
    if (diff_stream && pre_hcom) {
      inner_parallel_streams_.emplace_back(std::vector<uint32_t>{pre_stream_id, cur_stream_id});
    }

    pre_cnode_ptr = cur_cnode_ptr;
    pre_stream_id = cur_stream_id;
  }
}

void AscendStreamAssign::InsertSendRecvForDiffHcom(const shared_ptr<mindspore::session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<uint32_t> fusion_hcom_index;
  vector<CNodePtr> orders;
  for (size_t i = 0; i < cnode_ptr_list.size(); i++) {
    auto cur_cnode = cnode_ptr_list[i];
    if (IsHcom(cur_cnode)) {
      fusion_hcom_index.emplace_back(i);
    }
  }

  if (fusion_hcom_index.size() < 2) {
    MS_LOG(INFO) << "fusion hcom size is less than 2, no need insert event between them";
    return;
  }

  uint32_t first_index = fusion_hcom_index[0];
  uint32_t last_index = fusion_hcom_index[fusion_hcom_index.size() - 1];

  uint32_t cur_event_id = total_event_num_;
  uint32_t pre_hcom_stream_id = UINT32_MAX;
  std::copy(cnode_ptr_list.begin(), cnode_ptr_list.begin() + first_index, std::back_inserter(orders));
  for (size_t i = first_index; i <= last_index; i++) {
    auto cur_cnode = cnode_ptr_list[i];
    auto it = std::find(fusion_hcom_index.begin(), fusion_hcom_index.end(), i);
    if (it == fusion_hcom_index.end()) {
      orders.emplace_back(cur_cnode);
      continue;
    }

    auto cur_hcom_stream_id = AnfAlgo::GetStreamId(cur_cnode);
    if (cur_hcom_stream_id == pre_hcom_stream_id) {
      orders.emplace_back(cur_cnode);
      continue;
    }

    if (i == first_index) {
      // first fusion hcom
      orders.emplace_back(cur_cnode);
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(send);
    } else if (i == last_index) {
      // last fusion hcom
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(recv);
      orders.emplace_back(cur_cnode);
      cur_event_id++;
    } else {
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(recv);
      cur_event_id++;
      orders.emplace_back(cur_cnode);
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      orders.emplace_back(send);
    }

    pre_hcom_stream_id = cur_hcom_stream_id;
  }

  std::copy(cnode_ptr_list.begin() + last_index + 1, cnode_ptr_list.end(), std::back_inserter(orders));
  graph_ptr->set_execution_order(orders);
  total_event_num_ = cur_event_id;
  MS_LOG(INFO) << "after indsert between allreduce, total event nums[" << total_event_num_ << "]";
  MS_LOG(INFO) << "end";
}

void AscendStreamAssign::InsertSendRecvForHcomParallel(const shared_ptr<mindspore::session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = 0;
  auto it = cnodes.begin();
  while (it != cnodes.end() && (it + 1) != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    MS_EXCEPTION_IF_NULL(*(it + 1));
    if (IsHcom(*it) && !IsHcom(*(it + 1))) {
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));
      it = cnodes.insert(it + 1, send_cnode_ptr);

      auto target = FindTargetOp(it, cnodes.end(), *(it - 1));
      if (target == cnodes.end()) {
        MS_LOG(WARNING) << "hcom node[" << (*(it - 1))->fullname_with_scope()
                        << "] can't find target for insert recv op, no insert send/recv";
        it = cnodes.erase(it);
        continue;
      }

      // deal recv op
      uint32_t stream_id = AnfAlgo::GetStreamId(*target);
      CNodePtr recv_cnode_ptr = CreateRecvApplyKernel(graph_ptr, cur_event_id, stream_id);
      (void)cnodes.insert(target, recv_cnode_ptr);
      ++cur_event_id;
    }
    ++it;
  }
  graph_ptr->set_execution_order(cnodes);
  total_event_num_ = cur_event_id;
  MS_LOG(INFO) << "after insert send/recv for hcom parallel, total event nums[" << total_event_num_ << "]";

  // Insert Send/Recv between Hcom(such as:AllReduce1 Send1 Common Recv1 AllReduce2)
  InsertSendRecvForDiffHcom(graph_ptr);
  MS_LOG(INFO) << "end";
}

bool AscendStreamAssign::IsProcessedParallelStream(uint32_t stream_id) {
  auto it = std::find(processed_parallel_streams_.begin(), processed_parallel_streams_.end(), stream_id);
  if (it != processed_parallel_streams_.end()) {
    return true;
  }
  return false;
}

void AscendStreamAssign::GetParallelStream(uint32_t cur_stream_id, uint32_t stream_acitve_id,
                                           vector<uint32_t> *parallel_streams) {
  for (size_t i = 0; i < inner_parallel_streams_.size(); i++) {
    auto cur_parallel_streams = inner_parallel_streams_[i];
    auto it = std::find(cur_parallel_streams.begin(), cur_parallel_streams.end(), cur_stream_id);
    if (it != cur_parallel_streams.end()) {
      MS_LOG(INFO) << "stream id:" << cur_stream_id << " is parallel stream";
      for (size_t j = 0; j < cur_parallel_streams.size(); j++) {
        if (cur_parallel_streams[j] == stream_acitve_id) {
          MS_LOG(INFO) << "one of parallel stream id" << cur_parallel_streams[j]
                       << "is same with streamacvite stream id" << stream_acitve_id;
          continue;
        }
        (*parallel_streams).emplace_back(cur_parallel_streams[j]);
      }

      // record processed parallel streams
      (void)std::copy((*parallel_streams).begin(), (*parallel_streams).end(),
                      std::back_inserter(processed_parallel_streams_));
      return;
    }
  }

  (*parallel_streams).push_back(cur_stream_id);
}

void AscendStreamAssign::InsertActiveNew(const std::shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  std::vector<CNodePtr> update_cnode_list;
  CNodePtr cur_cnode_ptr = nullptr;
  CNodePtr pre_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;

  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (cur_stream_id >= kIndependFirstStreamId) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    bool inner_active = pre_stream_id != cur_stream_id && pre_stream_id < cur_stream_id &&
                        AnfAlgo::GetCNodeName(pre_cnode_ptr) != kStreamSwitchOpName &&
                        AnfAlgo::GetCNodeName(pre_cnode_ptr) != kStreamActiveOpName &&
                        AnfAlgo::GetCNodeName(pre_cnode_ptr) != kSendOpName;
    bool processed = IsProcessedParallelStream(cur_stream_id);
    // 1)inner stream assign, need insert active op
    if (inner_active && !processed) {
      MS_LOG(INFO) << "Inner insert active op, self stream id[" << pre_stream_id << "]";
      CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
      update_cnode_list.emplace_back(active_ptr);
      // 1.set stream id
      AnfAlgo::SetStreamId(pre_stream_id, active_ptr.get());
      // 2.set active stream ids
      std::vector<uint32_t> active_index_list;
      GetParallelStream(cur_stream_id, pre_stream_id, &active_index_list);
      AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);
    }
    // inner_active is not a if/else relationship with the next if/else. such as:StreamActive(S7)-->StreamActive(S8)
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamActiveOpName &&
        AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get()) != UINT32_MAX) {
      // 2)outter stream assign, update active op
      update_cnode_list.emplace_back(cur_cnode_ptr);
      UpdateStreamActive(cur_cnode_ptr);
    } else if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName) {
      // 3)update switch op
      MS_LOG(INFO) << "Insert active op after switch";
      CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
      update_cnode_list.emplace_back(cur_cnode_ptr);
      update_cnode_list.emplace_back(active_ptr);
      UpdateStreamSwitch(cur_cnode_ptr, active_ptr);
    } else {
      update_cnode_list.emplace_back(cur_cnode_ptr);
    }

    pre_stream_id = cur_stream_id;
    pre_cnode_ptr = cur_cnode_ptr;
  }
  graph_ptr->set_execution_order(update_cnode_list);
  MS_LOG(INFO) << "end";
}

void AscendStreamAssign::UpdateEventId(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  CNodePtr cur_cnode_ptr = nullptr;
  // key:virutal event id, value:real event id
  std::unordered_map<uint32_t, uint32_t> event_id_map;
  uint32_t event_id;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kSendOpName || AnfAlgo::GetCNodeName(cur_cnode_ptr) == kRecvOpName) {
      auto primitive = AnfAlgo::GetCNodePrimitive(cur_cnode_ptr);
      MS_EXCEPTION_IF_NULL(primitive);
      event_id = GetValue<uint32_t>(primitive->GetAttr(kAttrEventId));
      // before stream assign, send/recv event_id assign from kFirstEventId
      if (event_id < kFirstEventId) {
        continue;
      }
      auto it = event_id_map.find(event_id);
      if (it == event_id_map.end()) {
        event_id_map.insert(std::make_pair(event_id, total_event_num_));
        AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue<uint32_t>(total_event_num_), cur_cnode_ptr);
        total_event_num_++;
      } else {
        AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue<uint32_t>(it->second), cur_cnode_ptr);
      }
    }
  }
}

void AscendStreamAssign::UpdateStreamId(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  CNodePtr cur_cnode_ptr = nullptr;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (cur_stream_id < kIndependFirstStreamId) {
      if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamActiveOpName) {
        auto primitive = AnfAlgo::GetCNodePrimitive(cur_cnode_ptr);
        MS_EXCEPTION_IF_NULL(primitive);
        vector<uint32_t> active_ids = GetValue<std::vector<uint32_t>>(primitive->GetAttr(kAttrActiveStreamList));
        for (size_t j = 0; j < active_ids.size(); j++) {
          if (active_ids[j] >= kIndependFirstStreamId) {
            active_ids[j] = active_ids[j] - kIndependFirstStreamId + total_common_stream_num_;
          }
        }
        ValuePtr active_value = MakeValue<std::vector<uint32_t>>(active_ids);
        AnfAlgo::SetNodeAttr(kAttrActiveStreamList, active_value, cur_cnode_ptr);
      }
    } else {
      uint32_t update_id = cur_stream_id - kIndependFirstStreamId + total_common_stream_num_;
      AnfAlgo::SetStreamId(update_id, cur_cnode_ptr.get());
    }

    // update AtomicAddrClean stream same witch the next node
    if (i > 0 && AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]) == "AtomicAddrClean") {
      MS_LOG(INFO) << "update AtomicAddrClean stream id from[" << AnfAlgo::GetStreamId(cnode_ptr_list[i - 1])
                   << "] to [" << AnfAlgo::GetStreamId(cur_cnode_ptr) << "]";
      AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(cur_cnode_ptr), cnode_ptr_list[i - 1].get());
    }
  }

  // update logic_to_independent_map_
  for (auto &indep : logic_to_independent_map_) {
    if (indep.second >= kIndependFirstStreamId) {
      indep.second = indep.second - kIndependFirstStreamId + total_common_stream_num_;
    }
  }

  // update independent_before_physic_id_
  for (auto &id : independent_before_physic_id_) {
    if (id >= kIndependFirstStreamId) {
      id = id - kIndependFirstStreamId + total_common_stream_num_;
    }
  }

  // update independent_id_
  independent_id_ = independent_id_ - kIndependFirstStreamId + total_common_stream_num_;
  MS_LOG(INFO) << "end";
}

void AscendStreamAssign::GetNeedActiveStreams(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  CNodePtr cur_cnode_ptr = nullptr;
  auto cnode_ptr_list = graph_ptr->execution_order();
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
      MS_LOG(INFO) << "stream id:" << stream_id << " is need actived at first";
      need_first_active_streams_.push_back(stream_id);
    }
  }
}

void AscendStreamAssign::AssignStreamNew(const shared_ptr<session::KernelGraph> &graph_ptr) {
  if (IsTaskSink()) {
    ResetNew();
    ReorderIndependentOrders(graph_ptr);
    AssignAllNodesStream(graph_ptr);
    FindAllReduceParallel(graph_ptr);
    InsertActiveNew(graph_ptr);
    InsertSendRecvForHcomParallel(graph_ptr);
    InsertSendRecvForIndependent(graph_ptr);
    UpdateStreamId(graph_ptr);
    UpdateEventId(graph_ptr);
    GetNeedActiveStreams(graph_ptr);

    MS_LOG(INFO) << "after finish stream assign";
    PrintGraphExeOrders(graph_ptr);

    // Get info for D Model
    generator::IRModelUtil::GetInstance().set_event_num(total_event_num());
    generator::IRModelUtil::GetInstance().set_stream_num(total_common_stream_num() + total_independ_stream_num());
    // Init to 1,temporarily
    generator::IRModelUtil::GetInstance().set_batch_num(1);
  }
}

CNodePtr AscendStreamAssign::CreateSendApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                                   uint32_t event_id, uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
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

CNodePtr AscendStreamAssign::CreateRecvApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                                   uint32_t event_id, uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
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
          MS_LOG(INFO) << "find target op[" << (*begin)->DebugString() << "]";
          return begin;
        }
      }
    }
    ++begin;
  }
  return end;
}  // namespace ascend

void AscendStreamAssign::InsertSendRecvForIndependent(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = total_event_num_;
  auto it = cnodes.begin();
  while (it != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    if (IsIndependentNode(*it)) {
      MS_LOG(INFO) << "deal independent op[" << (*it)->DebugString() << "]";
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));
      it = cnodes.insert(it + 1, send_cnode_ptr);

      auto target = FindTargetOp(it, cnodes.end(), *(it - 1));
      if (target == cnodes.end()) {
        MS_LOG(DEBUG) << "independ node[" << (*(it - 1))->fullname_with_scope()
                      << "] can't find target for insert recv op, no insert send/recv";
        it = cnodes.erase(it);
        continue;
      }

      // deal recv op
      uint32_t stream_id = AnfAlgo::GetStreamId(*target);
      CNodePtr recv_cnode_ptr = CreateRecvApplyKernel(graph_ptr, cur_event_id, stream_id);
      (void)cnodes.insert(target, recv_cnode_ptr);
      ++cur_event_id;
    }
    ++it;
  }
  graph_ptr->set_execution_order(cnodes);
  total_event_num_ = cur_event_id;
  MS_LOG(INFO) << "total event nums[" << total_event_num_ << "]";
  MS_LOG(INFO) << "end";
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
    MS_LOG(INFO) << "node " << node_ptr->fullname_with_scope() << " is independent, as inputs nums is zero";
    return true;
  }

  auto inputs = node_ptr->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    if (!inputs[i]->isa<ValueNode>()) {
      return false;
    }
  }
  MS_LOG(INFO) << "node " << node_ptr->fullname_with_scope() << " is independent, as inputs is all value node";
  return true;
}

bool AscendStreamAssign::IsTaskSink() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->enable_task_sink()) {
    MS_LOG(INFO) << "task sink mode is not enable";
    return false;
  } else {
    MS_LOG(INFO) << "task sink mode is enable";
    return true;
  }
}

void AscendStreamAssign::GetWaitStreams(vector<uint32_t> *wait_active_stream_list) {
  if (total_common_stream_num_ == 0) {
    MS_LOG(INFO) << "total_common_stream_num is zero";
    return;
  }

  // common stream:active first common stream
  MS_LOG(INFO) << "active physic id[" << first_physic_id_ << "]";
  for (uint32_t i = first_physic_id_ + 1; i < total_common_stream_num_; i++) {
    auto it = std::find(need_first_active_streams_.begin(), need_first_active_streams_.end(), i);
    if (it == need_first_active_streams_.end()) {
      MS_LOG(INFO) << "wait common stream id = " << i;
      (*wait_active_stream_list).push_back(i);
    }
  }

  // all independ stream id before first physical stream id should be actived
  auto it = logic_to_independent_map_.find(first_logic_id_);
  if (it != logic_to_independent_map_.end()) {
    uint32_t independent_id = it->second;
    auto res = std::find(independent_before_physic_id_.begin(), independent_before_physic_id_.end(), independent_id);
    if (res == independent_before_physic_id_.end()) {
      // first physical to independ id may be not in independent_before_physic_id_
      independent_before_physic_id_.push_back(independent_id);
    }
    MS_LOG(INFO) << "active independent id[" << independent_id << "]";
  }

  uint32_t max_before_physic = 0;
  for (size_t i = 0; i < independent_before_physic_id_.size(); i++) {
    if (independent_before_physic_id_[i] > max_before_physic) {
      max_before_physic = independent_before_physic_id_[i];
    }
    MS_LOG(INFO) << "independent id[" << independent_before_physic_id_[i] << "] before first physic is active";
  }

  for (uint32_t i = 0; i < total_independ_stream_num_; i++) {
    if (i + total_common_stream_num_ <= max_before_physic) {
      continue;
    }
    // all wait streams should not in need_first_active_streams_
    auto iter =
      std::find(need_first_active_streams_.begin(), need_first_active_streams_.end(), i + total_common_stream_num_);
    if (iter == need_first_active_streams_.end()) {
      MS_LOG(INFO) << "wait independent stream id:" << i + total_common_stream_num_;
      (*wait_active_stream_list).push_back(i + total_common_stream_num_);
    }
  }
}

uint32_t AscendStreamAssign::GetTotalStreamNum() const { return total_common_stream_num_ + total_independ_stream_num_; }
void AscendStreamAssign::ReorderIndependentOrders(const shared_ptr<mindspore::session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  CNodePtr cur_cnode_ptr = nullptr;
  std::vector<CNodePtr> exe_orders;
  std::vector<CNodePtr> independents;
  std::vector<CNodePtr> others;

  auto cnode_ptr_list = graph_ptr->execution_order();
  MS_LOG(INFO) << "before reorder, graph orders size:" << cnode_ptr_list.size();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (IsIndependentNode(cur_cnode_ptr)) {
      independents.emplace_back(cur_cnode_ptr);
    } else {
      others.emplace_back(cur_cnode_ptr);
    }
  }

  if (others.empty() || independents.empty()) {
    MS_LOG(INFO) << "independent or others is empty, no need reorder";
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

  MS_LOG(INFO) << "after reorder, graph orders size:" << exe_orders.size();
  if (processed.size() != independents.size()) {
    MS_LOG(WARNING) << "processed independent nodes size is not equal to exiting independent nodes size";
    return;
  }

  graph_ptr->set_execution_order(exe_orders);
}

void AscendStreamAssign::PrintGraphExeOrders(const shared_ptr<mindspore::session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kSendOpName || AnfAlgo::GetCNodeName(cur_cnode_ptr) == kRecvOpName) {
      auto primitive = AnfAlgo::GetCNodePrimitive(cur_cnode_ptr);
      MS_LOG(INFO) << "node name[" << AnfAlgo::GetCNodeName(cur_cnode_ptr) << "], logic id["
                   << AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get()) << "], stream id["
                   << AnfAlgo::GetStreamId(cur_cnode_ptr) << "], event_id["
                   << GetValue<uint32_t>(primitive->GetAttr(kAttrEventId)) << "]";
    } else {
      MS_LOG(INFO) << "node name[" << cur_cnode_ptr->fullname_with_scope() << "], logic id["
                   << AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get()) << "], stream id["
                   << AnfAlgo::GetStreamId(cur_cnode_ptr) << "]";
    }
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
