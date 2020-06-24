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

void AscendStreamAssign::AssignStream(const shared_ptr<session::KernelGraph> &graph_ptr) {
  if (IsTaskSink()) {
    Reset();
    ReorderIndependentOrders(graph_ptr);
    AssignAllNodesStream(graph_ptr);
    UpdateAtomicAddrCleanStreamId(graph_ptr);
    FindHcomParallelStreams(graph_ptr);
    InsertStreamActive(graph_ptr);
    InsertSendRecvForHcomParallel(graph_ptr);
    InsertSendRecvForIndependent(graph_ptr);
    UpdateEventId(graph_ptr);
    GetNeedActiveStreams(graph_ptr);
    graph_ptr->PrintGraphExecuteOrder();
    CheckStreamAssign(graph_ptr);
    MS_LOG(INFO) << "after finish stream assign";

    // Get info for D Model
    AscendStreamMng &stream_manager = AscendStreamMng::GetInstance();
    generator::IRModelUtil::GetInstance().set_event_num(total_event_num());
    generator::IRModelUtil::GetInstance().set_stream_num(stream_manager.GetCurAllocStreamNum());
    // Init to 1,temporarily
    generator::IRModelUtil::GetInstance().set_batch_num(1);
  }
}

// section 0
void AscendStreamAssign::CheckStreamAssign(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  std::set<uint32_t> streams;
  uint32_t max_stream = 0;
  uint32_t min_stream = kInvalidStreamId;
  const std::vector<CNodePtr> &cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (stream_id == kInvalidStreamId) {
      MS_LOG(EXCEPTION) << "node [" << AnfAlgo::GetCNodeName(cur_cnode_ptr) << "] had not been assigned streams";
    }

    streams.emplace(stream_id);
    if (stream_id > max_stream) {
      max_stream = stream_id;
    }
    if (stream_id < min_stream) {
      min_stream = stream_id;
    }
  }

  if (!streams.empty()) {
    if (min_stream != 0) {
      MS_LOG(EXCEPTION) << "before stream assign, assigned stream should start from 0, now is from " << min_stream;
    }
    if (max_stream != (streams.size() - 1)) {
      MS_LOG(EXCEPTION) << "before stream assign, assigned stream should be consecutive";
    }
  }
}

// section 1
void AscendStreamAssign::AssignAllNodesStream(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto cnode_ptr_list = graph_ptr->execution_order();
  CNodePtr pre_cnode_ptr = nullptr;
  uint32_t cur_index = 0;
  uint32_t cur_stream_id = 0;

  bool exit_independent = false;
  AscendStreamMng &stream_manager = AscendStreamMng::GetInstance();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
      continue;
    }
    if (IsIndependentNode(cur_cnode_ptr)) {
      exit_independent = true;
      continue;
    }
    // first common node, only exe one time
    if (pre_cnode_ptr == nullptr) {
      uint32_t cur_stream_num = stream_manager.GetCurAllocStreamNum();
      if (cur_stream_num == 0) {
        cur_stream_id = stream_manager.ApplyNewStream();
      } else {
        cur_stream_id = stream_manager.GetCurAllocStream();
      }
      ++cur_index;
      pre_cnode_ptr = cur_cnode_ptr;
      AnfAlgo::SetStreamId(cur_stream_id, cur_cnode_ptr.get());
      if (IsHcom(cur_cnode_ptr)) {
        hcom_stream_list_.emplace(cur_stream_id);
      }
      continue;
    }

    AssignCommonStreamId(cur_cnode_ptr, &pre_cnode_ptr, &cur_index, &cur_stream_id);
  }

  if (exit_independent) {
    uint32_t first_independent_stream_id = stream_manager.ApplyNewStream();
    for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
      CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
      MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
      if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
        continue;
      }
      if (IsIndependentNode(cur_cnode_ptr)) {
        AssignIndependentStreamId(cur_cnode_ptr);
      }
    }
    MS_LOG(INFO) << "independent start from :" << first_independent_stream_id;
  }

  MS_LOG(INFO) << "total stream nums:" << stream_manager.GetCurAllocStreamNum();
}

void AscendStreamAssign::AssignIndependentStreamId(const CNodePtr &cur_cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  AscendStreamMng &stream_manager = AscendStreamMng::GetInstance();
  uint32_t cur_independent_id = stream_manager.GetCurAllocStream();
  auto it = independent_stream_map_.find(cur_independent_id);
  if (it == independent_stream_map_.end()) {
    AnfAlgo::SetStreamId(cur_independent_id, cur_cnode_ptr.get());
    independent_stream_map_.emplace(cur_independent_id, 1);
  } else {
    if (it->second < kCommonMaxTask) {
      AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
      it->second++;
    } else {
      cur_independent_id = stream_manager.ApplyNewStream();
      AnfAlgo::SetStreamId(cur_independent_id, cur_cnode_ptr.get());
      independent_stream_map_.emplace(cur_independent_id, 1);
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
    MS_LOG(INFO) << "node " << node_ptr->fullname_with_scope() << " is independent, as inputs nums is zero";
    return true;
  }

  const std::vector<AnfNodePtr> &inputs = node_ptr->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    if (!inputs[i]->isa<ValueNode>()) {
      return false;
    }
  }
  MS_LOG(INFO) << "node " << node_ptr->fullname_with_scope() << " is independent, as inputs is all value node";
  return true;
}

void AscendStreamAssign::AssignCommonStreamId(const CNodePtr &cur_cnode_ptr, CNodePtr *pre_cnode_ptr,
                                              uint32_t *cur_index, uint32_t *cur_stream_id) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  MS_EXCEPTION_IF_NULL(pre_cnode_ptr);
  MS_EXCEPTION_IF_NULL(*pre_cnode_ptr);
  AscendStreamMng &stream_manager = AscendStreamMng::GetInstance();
  bool over_max_hcom_task = (IsHcom(cur_cnode_ptr) && (*cur_index) % kHcomMaxTask == 0);
  bool over_max_common_task = (!IsHcom(cur_cnode_ptr) && (*cur_index) % kCommonMaxTask == 0);
  bool pre_common_cur_hcom = (IsHcom(cur_cnode_ptr) && !IsHcom(*pre_cnode_ptr));
  bool pre_hcom_cur_common = (!IsHcom(cur_cnode_ptr) && IsHcom(*pre_cnode_ptr));
  if (over_max_hcom_task || over_max_common_task || pre_common_cur_hcom || pre_hcom_cur_common) {
    *cur_index = 0;
    *cur_stream_id = stream_manager.ApplyNewStream();
  }

  ++(*cur_index);
  AnfAlgo::SetStreamId(*cur_stream_id, cur_cnode_ptr.get());
  *pre_cnode_ptr = cur_cnode_ptr;

  // record ll hcom streams as hcom stream has different stream flag
  if (IsHcom(cur_cnode_ptr)) {
    auto it = std::find(hcom_stream_list_.begin(), hcom_stream_list_.end(), *cur_stream_id);
    if (it == hcom_stream_list_.end()) {
      MS_LOG(INFO) << "hcom stream id:" << *cur_stream_id;
      hcom_stream_list_.emplace(*cur_stream_id);
    }
  }
}

// section 2:
void AscendStreamAssign::UpdateAtomicAddrCleanStreamId(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  const std::vector<CNodePtr> &cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    // update AtomicAddrClean stream same witch the next node
    if (i > 0 && AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]) == kAtomicAddrCleanOpName) {
      MS_LOG(INFO) << "update AtomicAddrClean stream id from[" << AnfAlgo::GetStreamId(cnode_ptr_list[i - 1])
                   << "] to [" << AnfAlgo::GetStreamId(cur_cnode_ptr) << "]";
      AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(cur_cnode_ptr), cnode_ptr_list[i - 1].get());
    }
  }
  MS_LOG(INFO) << "end";
}

// section 3
void AscendStreamAssign::FindHcomParallelStreams(const shared_ptr<session::KernelGraph> &graph_ptr) {
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

    bool pre_fusion_hcom = IsFusionHcom(pre_cnode_ptr);
    bool diff_stream = (pre_stream_id != cur_stream_id);
    if (diff_stream && pre_fusion_hcom) {
      inner_parallel_streams_.emplace_back(std::vector<uint32_t>{pre_stream_id, cur_stream_id});
    }

    pre_cnode_ptr = cur_cnode_ptr;
    pre_stream_id = cur_stream_id;
  }
}

// section 4
void AscendStreamAssign::UpdateStreamSwitch(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                            const CNodePtr &switch_ptr, const vector<uint32_t> &independent_stream,
                                            vector<CNodePtr> *orders) {
  MS_EXCEPTION_IF_NULL(orders);
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

  MS_LOG(INFO) << "start update switch op[" << switch_ptr->DebugString() << "]";
  MS_EXCEPTION_IF_NULL(switch_ptr);
  auto true_stream_id = GetValue<uint32_t>(primitive->GetAttr(kAttrTrueBranchStream));
  MS_LOG(INFO) << "streamswtich stream id[" << AnfAlgo::GetStreamId(switch_ptr) << "], true_logic_id[" << true_stream_id
               << "]";

  CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
  MS_LOG(INFO) << "start update StreamActive op[" << active_ptr->DebugString() << "]";
  AnfAlgo::SetStreamId(true_stream_id, active_ptr.get());
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(independent_stream), active_ptr);
  independent_stream_activated_ = true;

  // update processed stream
  for (auto &item : independent_stream) {
    processed_streams_.emplace(item);
  }

  orders->emplace_back(active_ptr);
}  // namespace ascend

void AscendStreamAssign::InsertStreamActive(const std::shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  std::vector<CNodePtr> update_cnode_list;
  CNodePtr cur_cnode_ptr = nullptr;
  CNodePtr pre_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;
  std::vector<uint32_t> independent_stream;
  MS_LOG(INFO) << "independent stream size:" << independent_stream_map_.size();
  for (auto item : independent_stream_map_) {
    independent_stream.emplace_back(item.first);
  }

  bool independent_flag = !(independent_stream.empty());

  const std::vector<CNodePtr> &cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (IsIndependentNode(cur_cnode_ptr)) {
      update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    bool inner_active = false;
    if (pre_cnode_ptr != nullptr) {
      inner_active = pre_stream_id != cur_stream_id && AnfAlgo::GetCNodeName(pre_cnode_ptr) != kStreamSwitchOpName &&
                     AnfAlgo::GetCNodeName(pre_cnode_ptr) != kSendOpName;
    }

    bool processed = IsProcessedStream(cur_stream_id);
    // 1)inner stream assign, need insert active op
    if (inner_active && !processed) {
      MS_LOG(INFO) << "Inner insert active op, self stream id[" << pre_stream_id << "]";
      CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
      // 1.set stream id
      AnfAlgo::SetStreamId(pre_stream_id, active_ptr.get());
      // 2.set active stream ids
      std::vector<uint32_t> active_index_list;
      GetParallelStream(cur_stream_id, pre_stream_id, &active_index_list);
      AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);
      update_cnode_list.emplace_back(active_ptr);
    }

    if (independent_flag && (AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName)) {
      MS_LOG(INFO) << "Insert StreamActive op after FP StreamSwitch for stream parallel";
      UpdateStreamSwitch(graph_ptr, cur_cnode_ptr, independent_stream, &update_cnode_list);
    } else {
      update_cnode_list.emplace_back(cur_cnode_ptr);
    }

    processed_streams_.emplace(cur_stream_id);
    pre_stream_id = cur_stream_id;
    pre_cnode_ptr = cur_cnode_ptr;
  }
  graph_ptr->set_execution_order(update_cnode_list);
  MS_LOG(INFO) << "end";
}

bool AscendStreamAssign::IsProcessedStream(uint32_t stream_id) {
  auto it = std::find(processed_streams_.begin(), processed_streams_.end(), stream_id);
  if (it != processed_streams_.end()) {
    return true;
  }
  return false;
}

void AscendStreamAssign::GetParallelStream(uint32_t cur_stream_id, uint32_t stream_acitve_id,
                                           vector<uint32_t> *parallel_streams) {
  MS_EXCEPTION_IF_NULL(parallel_streams);
  for (size_t i = 0; i < inner_parallel_streams_.size(); i++) {
    const auto &cur_parallel_streams = inner_parallel_streams_[i];
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
        processed_streams_.emplace(cur_parallel_streams[j]);
      }
      return;
    }
  }

  processed_streams_.emplace(cur_stream_id);
  (*parallel_streams).push_back(cur_stream_id);
}

// section5
void AscendStreamAssign::InsertSendRecvForDiffHcom(const shared_ptr<mindspore::session::KernelGraph> &graph_ptr) {
  MS_LOG(INFO) << "start";
  MS_EXCEPTION_IF_NULL(graph_ptr);
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<uint32_t> fusion_hcom_index;
  vector<CNodePtr> orders;
  for (size_t i = 0; i < cnode_ptr_list.size(); i++) {
    auto cur_cnode = cnode_ptr_list[i];
    if (IsFusionHcom(cur_cnode)) {
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
  uint32_t pre_hcom_stream_id = kInvalidStreamId;
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
  MS_LOG(INFO) << "after indsert between allreduce, total event nums[" << total_event_num_ << "]\n end";
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
      bool is_fusion = IsFusionHcom(*it);
      if (!is_fusion) {
        ++it;
        continue;
      }
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

void AscendStreamAssign::GetNeedActiveStreams(const shared_ptr<session::KernelGraph> &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  CNodePtr cur_cnode_ptr = nullptr;
  auto cnode_ptr_list = graph_ptr->execution_order();
  // 1)stream witch kStreamNeedActivedFirst attr should be actived;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    ValuePtr value_ptr = nullptr;
    auto primitive = AnfAlgo::GetCNodePrimitive(cur_cnode_ptr);
    if (primitive != nullptr) {
      value_ptr = primitive->GetAttr(kStreamNeedActivedFirst);
    } else {
      auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(cur_cnode_ptr);
      MS_EXCEPTION_IF_NULL(func_graph);
      value_ptr = func_graph->get_attr(kStreamNeedActivedFirst);
    }
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

  // 2)first stream 0 should be actived first;
  need_first_active_streams_.emplace_back(0);

  // 3)independent stream:if has not been activate, push to need active vector
  if (!independent_stream_activated_) {
    for (auto &item : independent_stream_map_) {
      need_first_active_streams_.emplace_back(item.first);
    }
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
  MS_EXCEPTION_IF_NULL(wait_active_stream_list);
  AscendStreamMng &stream_manager = AscendStreamMng::GetInstance();
  uint32_t total_stream_num = stream_manager.GetCurAllocStreamNum();
  if (total_stream_num == 0) {
    MS_LOG(INFO) << "total_common_stream_num is zero";
    return;
  }

  // common stream:active first common stream
  for (uint32_t i = 0; i < total_stream_num; i++) {
    auto it = std::find(need_first_active_streams_.begin(), need_first_active_streams_.end(), i);
    if (it == need_first_active_streams_.end()) {
      MS_LOG(INFO) << "wait common stream id = " << i;
      (*wait_active_stream_list).push_back(i);
    }
  }
}

bool AscendStreamAssign::IsHcom(const CNodePtr &apply_kernel) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  return AnfAlgo::GetKernelType(apply_kernel) == HCCL_KERNEL;
}

bool AscendStreamAssign::IsFusionHcom(const CNodePtr &cur_cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  bool is_hcom = IsHcom(cur_cnode_ptr);
  if (!is_hcom) {
    return false;
  }

  if (!AnfAlgo::HasNodeAttr(kAttrFusion, cur_cnode_ptr)) {
    return false;
  }

  if (AnfAlgo::GetNodeAttr<int>(cur_cnode_ptr, kAttrFusion) == 0) {
    return false;
  }

  return true;
}

void AscendStreamAssign::GetHcomStreams(std::vector<uint32_t> *streams) {
  MS_EXCEPTION_IF_NULL(streams);
  for (const auto &stream : hcom_stream_list_) {
    (*streams).emplace_back(stream);
  }
}

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

void AscendStreamAssign::Reset() {
  total_event_num_ = 0;
  independent_stream_activated_ = false;
  independent_stream_map_.clear();
  processed_streams_.clear();
  hcom_stream_list_.clear();
  need_first_active_streams_.clear();
  inner_parallel_streams_.clear();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
