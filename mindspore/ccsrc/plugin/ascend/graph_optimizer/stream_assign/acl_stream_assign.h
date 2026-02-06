/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include <tuple>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/utils/contract.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace device {
namespace ascend {
struct NodeExecInfo {
  CNodePtr node;
  uint32_t stream_id;
  size_t execution_order_index;
};
using NodeExecInfoPtr = std::shared_ptr<NodeExecInfo>;

struct StreamInfo {
  std::set<size_t> streams_set;
  std::set<size_t> streams_usr_set;
  std::map<size_t, std::set<size_t>> no_event_streams;
};

struct NodeIoExecInfo {
  NodeExecInfoPtr node_exec_info;
  std::vector<NodeExecInfoPtr> inputs;
  std::vector<NodeExecInfoPtr> outputs;
};
using NodeIoExecInfoPtr = std::shared_ptr<NodeIoExecInfo>;

struct ResLimitInfo {
  uint32_t cube_num;
  uint32_t vector_num;
  bool cube_num_modify_flag;
  bool vector_num_modify_flag;
};
using ResLimitInfoPtr = std::shared_ptr<ResLimitInfo>;

class AclStreamAssign {
 public:
  static AclStreamAssign &GetInstance() {
    static AclStreamAssign instance;  // Guaranteed to be destroyed.
    return instance;
  }

  AclStreamAssign(const AclStreamAssign &) = delete;
  AclStreamAssign &operator=(const AclStreamAssign &) = delete;

  void AssignStream(const NotNull<KernelGraphPtr> &kernel_graph, DeviceResManager *device_res_manager);
  void CreateEvent(const NotNull<KernelGraphPtr> &kernel_graph);
  std::pair<CNodePtr, CNodePtr> CreateSendReceive(const NotNull<KernelGraphPtr> &kernel_graph,
                                                  uint32_t record_stream_id, uint32_t wait_stream_id);

 private:
  AclStreamAssign() = default;
  ~AclStreamAssign() = default;

  void GenKernelIoExecInfoMap(const NotNull<KernelGraphPtr> &kernel_graph,
                              mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> *kernel_io_exec_info_map) const;

  std::pair<CNodePtr, CNodePtr> CreateSendRecvEventsPair(const NotNull<KernelGraphPtr> &kernel_graph,
                                                         size_t send_stream_id, size_t wait_stream_id);

  void UpdateEventsToExecutionOrder(const NotNull<KernelGraphPtr> &kernel_graph,
                                    const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &send_after_node,
                                    const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &recv_before_node,
                                    const mindspore::HashMap<AnfNodePtr, std::set<size_t>> &producer_streams);

  void UpdateGPTOEventsToExecutionOrder(
    const NotNull<KernelGraphPtr> &kernel_graph,
    const std::vector<std::pair<CNodePtr, std::tuple<char, size_t, size_t, size_t>>> &mock_exec_order);

  void GenEventsForParallelOp(const NotNull<KernelGraphPtr> &kernel_graph,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                              mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                              mindspore::HashMap<AnfNodePtr, std::set<size_t>> *producer_streams);

  void InsertEventForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph);

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
  StreamInfo AddInitialBoundarySync(const NotNull<KernelGraphPtr> &kernel_graph,
                                    std::vector<CNodePtr> *new_exec_orders);
  void AddFinalBoundarySync(const NotNull<KernelGraphPtr> &kernel_graph, const std::set<size_t> &streams_set,
                            const std::set<size_t> &streams_usr_set, std::vector<CNodePtr> *new_exec_orders,
                            std::map<size_t, std::set<size_t>> *no_event_streams);
  CNodePtr CreateLimitApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr,
                                  const mindspore::HashMap<std::string, uint32_t> &res_limit_map);
  void InsertResLimitForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph, DeviceResManager *device_res_manager);
  void InsertResLimit(const NotNull<KernelGraphPtr> &kernel_graph, DeviceResManager *device_res_manager,
                      const mindspore::HashMap<size_t, ResLimitInfoPtr> &stream_res_limit_map, bool is_dyn_graph);
  CNodePtr CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t stream_id,
                                 uint32_t event_generate_id);
  CNodePtr CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t record_stream_id,
                                 uint32_t stream_id, uint32_t event_generate_id);
  void AddBoundarySendRecvKernel(const NotNull<KernelGraphPtr> &kernel_graph, uint32_t record_stream_id,
                                 uint32_t wait_stream_id, std::vector<CNodePtr> *exec_order,
                                 std::map<size_t, std::set<size_t>> *no_event_streams, CNodePtr pre_cnode = nullptr,
                                 CNodePtr next_cnode = nullptr);
  void AddDelayedSendRecvKernel(const NotNull<mindspore::KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                size_t exec_idx, uint32_t record_stream_id,
                                const std::vector<CNodePtr> &origin_exec_order, std::vector<CNodePtr> *exec_order,
                                mindspore::HashMap<size_t, std::vector<CNodePtr>> *delayed_recv_nodes);
  void ProcessSideEffect(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr kernel, size_t process_stream_id,
                         const CNodePtr last_kernel, std::vector<AnfNodePtr> *real_inputs,
                         std::map<AnfNodePtr, std::set<size_t>> *side_effect_map,
                         std::map<size_t, std::set<size_t>> *no_event_streams, std::vector<CNodePtr> *new_exec_orders);
  std::atomic<uint32_t> event_generate_id_ = 0;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ACL_STREAM_ASSIGN_H_
