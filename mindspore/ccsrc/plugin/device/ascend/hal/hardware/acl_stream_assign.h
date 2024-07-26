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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ACL_STREAM_ASSIGN_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ACL_STREAM_ASSIGN_H_

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
#include "include/backend/kernel_graph.h"
#include "include/common/utils/contract.h"

namespace mindspore {
namespace device {
namespace ascend {
struct NodeExecInfo {
  CNodePtr node;
  uint32_t stream_id;
  size_t execution_order_index;
};
using NodeExecInfoPtr = std::shared_ptr<NodeExecInfo>;

struct NodeIoExecInfo {
  NodeExecInfoPtr node_exec_info;
  std::vector<NodeExecInfoPtr> inputs;
  std::vector<NodeExecInfoPtr> outputs;
};
using NodeIoExecInfoPtr = std::shared_ptr<NodeIoExecInfo>;

class AclStreamAssign {
 public:
  static AclStreamAssign &GetInstance() {
    static AclStreamAssign instance;  // Guaranteed to be destroyed.
    return instance;
  }

  AclStreamAssign(const AclStreamAssign &) = delete;
  AclStreamAssign &operator=(const AclStreamAssign &) = delete;

  void AssignStream(const NotNull<KernelGraphPtr> &kernel_graph,
                    const std::vector<std::pair<CNodePtr, CNodePtr>> &sched_events);

 private:
  AclStreamAssign() = default;
  ~AclStreamAssign() = default;

  void GenKernelIoExecInfoMap(const NotNull<KernelGraphPtr> &kernel_graph,
                              mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> *kernel_io_exec_info_map) const;

  void UpdateEventsToExecutionOrder(const NotNull<KernelGraphPtr> &kernel_graph,
                                    const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &send_after_node,
                                    const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &recv_before_node,
                                    const mindspore::HashMap<AnfNodePtr, std::set<size_t>> &producer_streams);

  void GenEventsForParallelOp(const NotNull<KernelGraphPtr> &kernel_graph,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                              mindspore::HashMap<AnfNodePtr, std::set<size_t>> *producer_streams);

  void InsertEventForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph,
                                 const std::vector<std::pair<CNodePtr, CNodePtr>> &sched_events);

  void ProcessStreamForInputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                              const NodeIoExecInfoPtr &io_exec_info,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                              mindspore::HashMap<AnfNodePtr, std::set<size_t>> *producer_streams);

  void InsertEventsForOutputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                              const NodeIoExecInfoPtr &io_exec_info,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv);

  void InsertEvents(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &parallel_cnode,
                    const AnfNodePtr &node_before_send,
                    mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                    mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                    const AnfNodePtr &node_after_recv);

  CNodePtr CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t stream_id,
                                 uint32_t event_generate_id);
  CNodePtr CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t record_stream_id,
                                 uint32_t stream_id, uint32_t event_generate_id);
  void AddBoundarySendRecvKernel(const NotNull<KernelGraphPtr> &kernel_graph, uint32_t record_stream_id,
                                 uint32_t wait_stream_id, std::vector<CNodePtr> *exec_order,
                                 std::map<size_t, std::set<size_t>> *no_event_streams, CNodePtr pre_cnode = nullptr,
                                 CNodePtr next_cnode = nullptr);
  void ProcessSideEffect(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr kernel, size_t process_stream_id,
                         const CNodePtr last_kernel, std::vector<AnfNodePtr> *real_inputs,
                         std::map<AnfNodePtr, std::set<size_t>> *side_effect_map,
                         std::map<size_t, std::set<size_t>> *no_event_streams, std::vector<CNodePtr> *new_exec_orders);
  CNodePtr GetTargetRecvKernel(const std::vector<CNodePtr> &recv_ops, int64_t target_micro, int64_t target_chunk,
                               bool target_phase);
  void InsertEventsForSendOp(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                             const std::vector<CNodePtr> &recv_ops, int64_t chunk_num, int64_t micro_size,
                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv);
  std::atomic<uint32_t> event_generate_id_ = 0;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ACL_STREAM_ASSIGN_H_
