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

#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>
#include "runtime/base.h"
#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "session/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
using std::map;
using std::shared_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;

class AscendStreamAssign {
 public:
  static AscendStreamAssign &GetInstance() {
    static AscendStreamAssign instance;  // Guaranteed to be destroyed.
    return instance;
  }

  AscendStreamAssign(const AscendStreamAssign &) = delete;
  AscendStreamAssign &operator=(const AscendStreamAssign &) = delete;

  uint32_t GetTotalStreamNum() const;
  // new stream policy
  uint32_t total_common_stream_num() const { return total_common_stream_num_; }
  uint32_t total_independ_stream_num() const { return total_independ_stream_num_; }
  uint32_t total_event_num() const { return total_event_num_; }

  void InsertActiveNew(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void AssignAllNodesStream(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void ResetNew();
  void AssignStreamNew(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  bool IsIndependentNode(const CNodePtr &node_ptr);
  const std::unordered_map<uint32_t, uint32_t> &logic_to_independent_map() { return logic_to_independent_map_; }
  const std::unordered_map<uint32_t, uint32_t> &logic_to_physic_map() { return logic_to_physic_map_; }
  const std::vector<std::vector<uint32_t>> &inner_parallel_streams() { return inner_parallel_streams_; }
  void GetWaitStreams(vector<uint32_t> *wait_active_stream_list);
  const std::vector<uint32_t> &hcom_streams() { return hcom_stream_list_; }
  CNodePtr CreateSendApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr, uint32_t event_id,
                                 uint32_t stream_id);
  CNodePtr CreateRecvApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr, uint32_t event_id,
                                 uint32_t stream_id);

 private:
  AscendStreamAssign() = default;
  ~AscendStreamAssign() = default;

  vector<CNodePtr>::iterator FindTargetOp(vector<CNodePtr>::iterator begin, vector<CNodePtr>::iterator end,
                                          const CNodePtr &node);

  bool IsHcom(const CNodePtr &apply_kernel);
  bool IsProcessed(uint32_t logic_id);
  void TransLogicToPhysic(const vector<uint32_t> &logic_ids, vector<uint32_t> *physic_ids);
  void AssignCommonStreamId(const CNodePtr &cur_cnode_ptr, CNodePtr *pre_cnode_ptr, uint32_t *cur_index,
                            uint32_t *cur_stream_id);
  void RecordIdMap(uint32_t logic_id, uint32_t physic_id);
  void UpdateStreamActive(const CNodePtr &active_ptr);
  void UpdateStreamSwitch(const CNodePtr &switch_ptr, const CNodePtr &active_ptr);
  bool IsTaskSink();
  void AssignIndependentStreamId(const CNodePtr &cur_cnode_ptr, uint32_t deal_logic_id);
  void UpdateStreamId(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void UpdateEventId(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void PrintGraphExeOrders(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void RecordFirstCommonOp(const CNodePtr &cur_cnode_ptr, uint32_t cur_node_logic_id, uint32_t cur_stream_id);
  uint32_t GetLogicId(const CNodePtr &cur_cnode_ptr);
  void SetCommonStreamNum(uint32_t cur_stream_id);
  void FindAllReduceParallel(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  bool IsProcessedParallelStream(uint32_t stream_id);
  void GetParallelStream(uint32_t cur_stream_id, uint32_t stream_acitve_id, std::vector<uint32_t> *parallel_streams);
  void InsertSendRecvForIndependent(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void InsertSendRecvForHcomParallel(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void GetNeedActiveStreams(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void ReorderIndependentOrders(const std::shared_ptr<session::KernelGraph> &graph_ptr);

  uint32_t total_common_stream_num_{0};
  uint32_t total_independ_stream_num_{0};
  uint32_t total_event_num_{0};

  uint32_t first_physic_id_{UINT32_MAX};
  uint32_t first_logic_id_{UINT32_MAX};
  uint32_t independent_id_{UINT32_MAX};
  vector<uint32_t> processed_logic_id_{};
  std::unordered_map<uint32_t, uint32_t> logic_to_physic_map_{};       // key:logic id, value: first physic id
  std::unordered_map<uint32_t, uint32_t> logic_to_independent_map_{};  // key:logic id, value: dependent id
  std::vector<uint32_t> independent_before_physic_id_{};               // record independent id before first physic id
  std::vector<std::vector<uint32_t>> inner_parallel_streams_{};
  std::vector<uint32_t> processed_parallel_streams_{};
  std::vector<uint32_t> hcom_stream_list_{};
  std::vector<uint32_t> need_first_active_streams_{};
  // new policy end
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
