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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_

#include <functional>
#include <unordered_map>
#include <map>
#include <set>
#include <string>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_set>
#include <utility>
#include "runtime/base.h"
#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "backend/session/kernel_graph.h"
#include "utils/contract.h"

namespace mindspore {
namespace device {
namespace ascend {
using std::map;
using std::queue;
using std::shared_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using CNodeKey = void *;
const uint32_t kInvalidStreamId = UINT32_MAX;
const uint32_t kInvalidEventId = UINT32_MAX;
class AscendResourceMng {
 public:
  static AscendResourceMng &GetInstance() {
    static AscendResourceMng instance;
    return instance;
  }

  void ResetResource() {
    cur_stream_num_ = 0;
    cur_event_num_ = 0;
  }
  uint32_t ApplyNewStream() {
    if (!cur_stream_num_) {
      uint32_t cur_stream_id = cur_stream_num_;
      cur_stream_num_++;
      return cur_stream_id;
    }
    uint32_t cur_stream_id = cur_stream_num_;
    cur_stream_num_++;
    return cur_stream_id;
  }
  uint32_t ApplyNewEvent() {
    if (!cur_event_num_) {
      uint32_t cur_event_id = cur_event_num_;
      cur_event_num_++;
      return cur_event_id;
    }
    uint32_t cur_event_id = cur_event_num_;
    cur_event_num_++;
    return cur_event_id;
  }

  void DeleteEvent() {
    if (!cur_event_num_) {
      MS_LOG(WARNING) << "total event num is 0, no event to delete";
    } else {
      --cur_event_num_;
    }
  }
  uint32_t get_cur_stream_num() { return cur_stream_num_; }
  uint32_t GetCurAllocStreamId() {
    if (!cur_stream_num_) {
      MS_LOG(EXCEPTION) << "stream nums is 0, no stream id should be get";
    }
    return cur_stream_num_ - 1;
  }
  uint32_t get_cur_event_num() { return cur_event_num_; }

 private:
  uint32_t cur_stream_num_{0};
  uint32_t cur_event_num_{0};
};

enum StreamActiveKind { kInvalid = 0, kHead, kMiddle, kTail };
class AscendStreamAssign {
 public:
  static AscendStreamAssign &GetInstance() {
    static AscendStreamAssign instance;  // Guaranteed to be destroyed.
    return instance;
  }

  AscendStreamAssign(const AscendStreamAssign &) = delete;
  AscendStreamAssign &operator=(const AscendStreamAssign &) = delete;

  void AssignStream(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetHcomStreams(std::vector<uint32_t> *streams);
  void GetWaitStreams(vector<uint32_t> *wait_active_stream_list);
  const std::vector<std::vector<uint32_t>> &get_stream_group() const { return stream_groups_; }
  const std::map<CNodePtr, CNodePtr> &get_event_map() const { return event_map_; }

 private:
  AscendStreamAssign() = default;
  ~AscendStreamAssign() = default;
  void Reset();
  CNodePtr CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t stream_id);
  CNodePtr CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t stream_id);
  void CheckResourceAssign(const NotNull<KernelGraphPtr> &graph_ptr);
  void CheckStreamAssign(const NotNull<KernelGraphPtr> &graph_ptr);
  void CheckEventAssign(const NotNull<KernelGraphPtr> &graph_ptr);
  void AssignAllNodesStream(const NotNull<KernelGraphPtr> &graph_ptr);
  void AssignCommonStreamId(const CNodePtr &cur_cnode_ptr);
  void AssignHcom(const NotNull<KernelGraphPtr> &graph_ptr);
  uint32_t AssignHcomStreamId(const CNodePtr &cur_cnode_ptr, bool new_graph);
  void AssignIndependent(const NotNull<KernelGraphPtr> &graph_ptr);
  uint32_t AssignIndependentStreamId(const CNodePtr &cur_cnode_ptr, bool new_graph);
  void UpdateAtomicAddrCleanStreamId(const NotNull<KernelGraphPtr> &graph_ptr);
  void FindHcomParallelStreams(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActive(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActiveForCommon(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActiveForIndependent(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActiveForParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void ActiveRootGraphHcom(const NotNull<KernelGraphPtr> &graph_ptr, const std::set<uint32_t> &hcom_streams);
  void ActiveRootGraphIndependent(const NotNull<KernelGraphPtr> &graph_ptr, std::set<uint32_t> independent_streams);
  void ActiveOtherGraphParallel(const NotNull<KernelGraphPtr> &graph_ptr,
                                std::map<uint32_t, std::set<uint32_t>> other_graph);
  void UpdateStreamSwitch(const NotNull<KernelGraphPtr> &graph_ptr, const CNodePtr &switch_ptr,
                          vector<CNodePtr> *orders);
  bool CheckStreamSwitch(const CNodePtr &switch_ptr);
  void InsertEventForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertCtrlForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventForHcomParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventCommonDependHcom(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventHcomDependCommon(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventHcomDependCommonBak(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventHcomDependHcom(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventBetweenHcom(const NotNull<KernelGraphPtr> &graph_ptr,
                              const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index);

  void AdjustAtomicAddrCleanOrder(const NotNull<KernelGraphPtr> &graph_ptr);
  vector<CNodePtr> GetLastInputCnode(const NotNull<KernelGraphPtr> &graph_ptr, const CNodePtr &cur_cnode_ptr);
  bool IsSatisfiedHcom(const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index, const CNodePtr &node_ptr,
                       size_t index);

  void GetProcessedStream(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetNeedActiveStreams(const NotNull<KernelGraphPtr> &graph_ptr);
  void ReorderIndependentOrders(const NotNull<KernelGraphPtr> &graph_ptr);

  void CheckScenario(const NotNull<KernelGraphPtr> &graph_ptr, vector<CNodePtr> *last_grad_and_status);
  CNodePtr GetCNodesNeededMoved(vector<CNodePtr> *moved_backward_cnodes, vector<CNodePtr> *moved_forward_cnodes,
                                const vector<CNodePtr> &last_grad_and_status, const NotNull<KernelGraphPtr> &graph_ptr);
  CNodePtr GetTargetOutputNode(const vector<CNodePtr> &moved_backward_cnodes, const CNodePtr first_node,
                               const NotNull<KernelGraphPtr> &graph_ptr);
  bool FinetuneSubgraphExecOrder(vector<CNodePtr> *cnodes);
  void TrailingTimeOptimizationByReorder(const NotNull<KernelGraphPtr> &graph_ptr);

  uint32_t GetMaxIndexTarget(const NotNull<KernelGraphPtr> &graph_ptr);
  uint32_t GetIndexByKey(const NotNull<KernelGraphPtr> &graph_ptr, const CNodeKey &key);
  uint32_t GetIndependentStreamSwitchStreamId(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetIndependentMaxTarget(const NotNull<KernelGraphPtr> &graph_ptr);

  bool IsTaskSink();
  bool IsHcom(const CNodePtr &cur_cnode_ptr);
  bool IsIndependentNode(const CNodePtr &node_ptr);
  bool IsProcessedStream(uint32_t stream_id);
  vector<CNodePtr>::iterator FindTargetOp(vector<CNodePtr>::iterator begin, vector<CNodePtr>::iterator end,
                                          const CNodePtr &node, bool exclude_hcom);
  void GetParallelStream(uint32_t cur_stream_id, uint32_t stream_acitve_id, std::vector<uint32_t> *parallel_streams);
  void SetLoopSink();

  // function for memory reuse
  void GetStreamRelations();
  void DFS(uint32_t start, std::vector<uint32_t> *group);
  bool IsVecExist(const std::vector<uint32_t> &group);
  void FindStreamRelations(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetStreamSwitchStreamRelation(const CNodePtr &node_ptr);
  void GetStreamActiveStreamRelation(const NotNull<KernelGraphPtr> &graph_ptr, size_t index);
  StreamActiveKind GetStreamActiveKind(const NotNull<KernelGraphPtr> &graph_ptr, size_t index);
  uint32_t GetStreamByActivedStream(uint32_t actived_stream_id);
  void PrintStreamRelations();
  void PrintStreamGroups();
  void FindEventRelations(const NotNull<KernelGraphPtr> &graph_ptr);
  bool IsSatisfiedEvent(uint32_t send_stream_id, uint32_t recv_stream_id) const;
  vector<CNodePtr> GetInputKernels(const CNodePtr &node);

  bool independent_stream_activated_{false};
  bool hcom_stream_activated_{false};
  bool loop_sink_{false};

  // key:stream id, value:node number
  std::map<uint32_t, uint32_t> common_stream_map_{};
  // key:stream id, value:node number
  std::map<uint32_t, uint32_t> independent_stream_map_{};
  // key:stream id, value:task number
  std::map<uint32_t, uint32_t> hcom_stream_map_{};

  std::set<uint32_t> processed_streams_{};
  std::vector<uint32_t> need_first_active_streams_{};
  std::set<CNodeKey> independent_targets_;

  // key:group name, value:key1:graph id, value1:stream id
  std::map<std::string, std::map<uint32_t, std::set<uint32_t>>> group_hcom_graph_map_;
  // key:graph id, value:stream set
  std::map<uint32_t, std::set<uint32_t>> independent_graph_map_;

  // attr for memory copy reuse
  std::map<uint32_t, std::vector<uint32_t>> stream_relations_{};
  std::vector<std::vector<uint32_t>> stream_groups_{};
  std::map<CNodePtr, CNodePtr> event_map_{};
  std::set<uint32_t> middle_active_streams_{};
  // new policy end
  bool IsAllOutGraphOut(const KernelGraphPtr &graph, const CNodePtr &cnode);
  vector<CNodePtr>::iterator FindGraphEnd(vector<CNodePtr>::iterator begin, vector<CNodePtr>::iterator end);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
