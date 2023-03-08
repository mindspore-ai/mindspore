/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"

#include <algorithm>
#include <utility>
#include <unordered_set>

#include "ir/manager.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/common/utils/parallel_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/device/kernel_adjust.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "backend/common/optimizer/helper.h"
#include "kernel/oplib/oplib.h"
#include "include/common/utils/utils.h"

#ifdef ENABLE_DUMP_IR
#include "debug/rdr/stream_exec_order_recorder.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr uint32_t kDeviceNumOfServer = 8;
constexpr uint32_t kDeviceNumThreshold = 1024;
const char kDefaultGroup[] = "__default_group";
constexpr auto kAttrStreamID = "stream_id";

constexpr uint32_t kHcomSecondaryStreamNum = 3;
constexpr uint32_t kMaxCommonNodeNumPerStream = 350;

constexpr uint32_t kTaskNumPerHcomNode = 300;
constexpr uint32_t kTaskNumPerWorldHcomNode = 350;
constexpr uint32_t kTaskNumPerSameServerHcomNode = 125;
constexpr uint32_t kTaskNumPerHcomSendRecvNode = 15;
constexpr uint32_t kTaskNumPerCommonNode = 3;

constexpr size_t kHcomNum = 2;
constexpr size_t kLastGradHcomOffset = 2;
constexpr size_t kLastGradAndStatusNum = 2;

const std::unordered_set<std::string> kDropoutGenMaskOps = {kDropoutGenMaskOpName, kDropoutGenMaskV3OpName,
                                                            kStatelessDropOutGenMaskOpName};

bool IsSameServer(const std::vector<uint32_t> &rank_ids) {
  auto min_iter = min_element(rank_ids.begin(), rank_ids.end());
  uint32_t min = (min_iter != rank_ids.end()) ? *min_iter : 0;
  auto max_iter = max_element(rank_ids.begin(), rank_ids.end());
  uint32_t max = (max_iter != rank_ids.end()) ? *max_iter : 0;
  return ((max - min < kDeviceNumOfServer) && (min / kDeviceNumOfServer == max / kDeviceNumOfServer));
}

string DoGetHcomGroup(const string &original_group, const std::vector<uint32_t> &rank_ids) {
  string communi_parallel_mode = parallel::ParallelContext::GetInstance()->communi_parallel_mode();
  if (communi_parallel_mode == parallel::kAllGroupParallel) {
    return original_group;
  }

  if (communi_parallel_mode == parallel::kNoGroupParallel) {
    return kDefaultGroup;
  }

  if (rank_ids.empty() || original_group == kHcclWorldGroup) {
    return kDefaultGroup;
  }

  if (IsSameServer(rank_ids)) {
    return original_group;
  }

  return kDefaultGroup;
}

string GetHcomGroup(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    MS_LOG(EXCEPTION) << "Hcom node " << cnode->fullname_with_scope() << " has no group attribute.";
  }

  auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  auto rank_ids = common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, cnode)
                    ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrGroupRankIds)
                    : std::vector<uint32_t>();
  auto new_group = DoGetHcomGroup(group_name, rank_ids);
  MS_LOG(INFO) << "hcom node: " << cnode->fullname_with_scope() << ", old group: " << group_name
               << ", new group: " << new_group;

  return new_group;
}

CNodePtr GetHcomAndOverflowMarker(const NotNull<KernelGraphPtr> &graph_ptr, vector<CNodePtr> *hcom_nodes) {
  MS_EXCEPTION_IF_NULL(hcom_nodes);
  auto cnode_ptr_list = graph_ptr->execution_order();
  CNodePtr overflow_marker = nullptr;
  std::string kNPUGetFloatStatusOpName = "NPUGetFloatStatus";
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    auto cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (common::AnfAlgo::GetCNodeName(cur_cnode_ptr) == kNPUGetFloatStatusOpName) {
      overflow_marker = cur_cnode_ptr;
    } else if (AnfAlgo::GetKernelType(cur_cnode_ptr) == HCCL_KERNEL) {
      (void)hcom_nodes->emplace_back(cur_cnode_ptr);
    } else if (i > 0 && common::AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]) == kAtomicAddrCleanOpName) {
      auto graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
      AnfAlgo::SetGraphId(graph_id, cnode_ptr_list[i - 1].get());
    }
  }
  return overflow_marker;
}

bool HasRefNodes(const vector<CNodePtr> &moved_backward_cnodes) {
  for (auto &cnode : moved_backward_cnodes) {
    std::string op_name = common::AnfAlgo::GetCNodeName(cnode);
    auto op_info = mindspore::kernel::OpLib::FindOp(op_name, kernel::kImplyTBE);
    if (op_info != nullptr && op_info->is_ref()) {
      MS_LOG(INFO) << "Find RefNode: " << op_name << ", full name: " << cnode->fullname_with_scope();
      return true;
    }
  }
  return false;
}

StreamActiveKind GetStreamKind(uint32_t cur_stream_id, uint32_t pre_stream_id, uint32_t next_stream_id) {
  // pre_stream_id equal to UINT32_MAX means no node active current StreamActive
  // next_stream_id equal to UINT32_MAX means current StreamActive active no node
  if (pre_stream_id == UINT32_MAX || next_stream_id == UINT32_MAX) {
    return kInvalid;
  }

  if (cur_stream_id == pre_stream_id && cur_stream_id == next_stream_id) {
    return kMiddle;
  }

  if (cur_stream_id == pre_stream_id) {
    return kTail;
  }

  if (cur_stream_id == next_stream_id) {
    return kHead;
  }

  return kInvalid;
}

void SetNodeStreamIDAttr(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto exec_orders = graph_ptr->execution_order();
  for (const auto &node : exec_orders) {
    common::AnfAlgo::SetNodeAttr(kAttrStreamID, MakeValue<uint32_t>(AnfAlgo::GetStreamId(node)), node);
  }
}

inline bool IsCommSubGraph(const KernelGraphPtr &root_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(node);
  uint32_t id = AnfAlgo::GetGraphId(node.get());
  return root_graph->IsCommSubGraph(id);
}

uint32_t GetGotoLabelId(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::GetCNodeName(node) != kLabelGotoOpName) {
    return kInvalidLabelId;
  }

  return common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
}

void AssignStreamWithDefaultStream(const KernelGraphPtr &graph) {
  auto nodes = graph->execution_order();
  for (auto &node : nodes) {
    AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
  }
}
}  // namespace

uint32_t AscendStreamAssign::GetHcomTaskNum(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    MS_LOG(EXCEPTION) << "Hcom node " << cnode->fullname_with_scope() << " has no group attribute.";
  }

  auto rank_ids = common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, cnode)
                    ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrGroupRankIds)
                    : std::vector<uint32_t>();
  if (rank_ids.empty()) {
    return kTaskNumPerHcomNode;
  }

  uint32_t device_num = 0;
  if (!CommManager::GetInstance().GetRankSize(kHcclWorldGroup, &device_num)) {
    MS_LOG(EXCEPTION) << "Get rank size failed.";
  }
  auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  if (group_name == kHcclWorldGroup) {
    if (device_num >= kDeviceNumThreshold) {
      return kTaskNumPerWorldHcomNode;
    }

    return kTaskNumPerHcomNode;
  }

  if (IsSameServer(rank_ids)) {
    return kTaskNumPerSameServerHcomNode;
  } else if (rank_ids.size() == static_cast<size_t>(device_num) && device_num >= kDeviceNumThreshold) {
    return kTaskNumPerWorldHcomNode;
  }
  return kTaskNumPerHcomNode;
}

void AscendStreamAssign::GetMaxStreamTaskNum() {
  auto ret = rtGetMaxStreamAndTask(RT_NORMAL_STREAM, &max_stream_count_, &max_task_count_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "call rtGetMaxStreamAndTask failed.";
  }
  MS_LOG(INFO) << "AscendStreamAssign::max_stream_count_: " << max_stream_count_;
  MS_LOG(INFO) << "AscendStreamAssign::max_task_count_: " << max_task_count_;
}

void AscendStreamAssign::AssignStreamForNonTaskSink(const std::vector<CNodePtr> &kernels) {
  if (kernels.empty()) {
    return;
  }
  for (const auto &node : kernels) {
    if (AnfAlgo::IsKernelSelectBackoffOp(node)) {
      continue;
    }
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      AnfAlgo::SetStreamId(kWorldGroupStreamIndex, node.get());
    } else {
      AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
    }
  }
  for (size_t i = 1; i < kernels.size(); ++i) {
    if (common::AnfAlgo::GetCNodeName(kernels[i - 1]) == kAtomicAddrCleanOpName) {
      auto stream_id = AnfAlgo::GetStreamId(kernels[i]);
      AnfAlgo::SetStreamId(stream_id, kernels[i - 1].get());
    }
  }
}

void GenKernelIoExecInfoMap(const NotNull<KernelGraphPtr> &kernel_graph,
                            mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> *kernel_io_exec_info_map) {
  auto &exec_kernels = kernel_graph->execution_order();
  for (size_t i = 0; i < exec_kernels.size(); ++i) {
    auto &process_kernel = exec_kernels[i];
    MS_EXCEPTION_IF_NULL(process_kernel);
    auto process_exec_info = std::make_shared<NodeExecInfo>();
    MS_EXCEPTION_IF_NULL(process_exec_info);
    process_exec_info->node = process_kernel;
    process_exec_info->stream_id = AnfAlgo::GetStreamId(process_kernel);
    process_exec_info->execution_order_index = i;
    auto process_io_exec_info = std::make_shared<NodeIoExecInfo>();
    MS_EXCEPTION_IF_NULL(process_io_exec_info);
    process_io_exec_info->node_exec_info = process_exec_info;
    process_io_exec_info->inputs = {};
    process_io_exec_info->outputs = {};
    (*kernel_io_exec_info_map)[process_kernel] = process_io_exec_info;
  }

  for (auto &process_kernel : exec_kernels) {
    MS_EXCEPTION_IF_NULL(process_kernel);
    auto process_iter = kernel_io_exec_info_map->find(process_kernel);
    if (process_iter == kernel_io_exec_info_map->end()) {
      MS_LOG(INFO) << "Can't get kernel io execution info for  " << process_kernel->fullname_with_scope();
      continue;
    }
    auto process_io_exec_info = process_iter->second;
    MS_EXCEPTION_IF_NULL(process_io_exec_info);
    auto process_exec_info = process_iter->second->node_exec_info;
    MS_EXCEPTION_IF_NULL(process_exec_info);
    auto inputs = process_kernel->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      auto input_node = common::AnfAlgo::VisitKernelWithReturnType(inputs[i], 0).first;
      MS_EXCEPTION_IF_NULL(input_node);
      if (AnfUtils::IsRealCNodeKernel(input_node)) {
        auto input_kernel = input_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(input_kernel);
        auto iter = kernel_io_exec_info_map->find(input_kernel);
        if (iter == kernel_io_exec_info_map->end()) {
          MS_LOG(INFO) << "Can't get kernel io execution info for " << process_kernel->fullname_with_scope()
                       << "'s input node " << input_kernel->fullname_with_scope();
          continue;
        }
        auto input_io_exec_info = iter->second;
        auto input_exec_info = iter->second->node_exec_info;
        MS_EXCEPTION_IF_NULL(input_io_exec_info);
        process_io_exec_info->inputs.push_back(input_exec_info);
        input_io_exec_info->outputs.push_back(process_exec_info);
      }
    }
  }
}

void AscendStreamAssign::InsertEventsForInputs(
  const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel, const NodeIoExecInfoPtr &io_exec_info,
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv) const {
  MS_EXCEPTION_IF_NULL(io_exec_info);
  auto process_stream_id = AnfAlgo::GetStreamId(kernel);
  auto input_exec_info_list = io_exec_info->inputs;
  mindspore::HashMap<uint32_t, NodeExecInfoPtr> stream_max_exec_node_map;

  for (auto &input : input_exec_info_list) {
    MS_EXCEPTION_IF_NULL(input);
    auto input_stream_id = input->stream_id;
    auto iter = stream_max_exec_node_map.find(input_stream_id);
    if (iter == stream_max_exec_node_map.end()) {
      stream_max_exec_node_map[input_stream_id] = input;
    } else {
      MS_EXCEPTION_IF_NULL(iter->second);
      if (input->execution_order_index > iter->second->execution_order_index) {
        iter->second = input;
      }
    }
  }

  for (auto input_exec : stream_max_exec_node_map) {
    MS_EXCEPTION_IF_NULL(input_exec.second);
    if (input_exec.second->stream_id == process_stream_id) {
      continue;
    }
    InsertEvents(kernel_graph, kernel, input_exec.second->node, kernel_send, kernel_recv, kernel);
  }
}

void AscendStreamAssign::InsertEventsForOutputs(
  const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel, const NodeIoExecInfoPtr &io_exec_info,
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv) const {
  MS_EXCEPTION_IF_NULL(io_exec_info);
  auto process_stream_id = AnfAlgo::GetStreamId(kernel);
  auto output_exec_info_list = io_exec_info->outputs;
  mindspore::HashMap<uint32_t, NodeExecInfoPtr> stream_min_exec_node_map;
  for (auto &output : output_exec_info_list) {
    MS_EXCEPTION_IF_NULL(output);
    auto output_stream_id = output->stream_id;
    auto iter = stream_min_exec_node_map.find(output_stream_id);
    if (iter == stream_min_exec_node_map.end()) {
      stream_min_exec_node_map[output_stream_id] = output;
    } else {
      MS_EXCEPTION_IF_NULL(iter->second);
      if (output->execution_order_index < iter->second->execution_order_index) {
        iter->second = output;
      }
    }
  }

  for (auto output_exec : stream_min_exec_node_map) {
    MS_EXCEPTION_IF_NULL(output_exec.second);
    if (output_exec.second->stream_id == process_stream_id) {
      continue;
    }
    InsertEvents(kernel_graph, kernel, kernel, kernel_send, kernel_recv, output_exec.second->node);
  }

  // parallel op has output tensor, and it didn't connect to other kernel, it's output is graph output, sync it.
  if (output_exec_info_list.empty() && (AnfAlgo::GetOutputTensorNum(kernel) != 0)) {
    InsertEvents(kernel_graph, kernel, kernel, kernel_send, kernel_recv, kernel_graph->output());
  }
}

void AscendStreamAssign::InsertEvents(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &parallel_cnode,
                                      const AnfNodePtr &node_before_send,
                                      mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                      mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                                      const AnfNodePtr &node_after_recv) const {
  MS_EXCEPTION_IF_NULL(kernel_send);
  MS_EXCEPTION_IF_NULL(kernel_recv);
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  uint32_t event_id = resource_manager.ApplyNewEvent();
  auto event = resource_manager.ApplyRtEvent();
  auto send_cnode = CreateSendApplyKernel(kernel_graph, event_id, AnfAlgo::GetStreamId(node_before_send));
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_cnode);
  auto send_iter = kernel_send->find(node_before_send);
  if (send_iter == kernel_send->end()) {
    (*kernel_send)[node_before_send] = {send_cnode};
  } else {
    send_iter->second.push_back(send_cnode);
  }

  CNodePtr recv_cnode = CreateRecvApplyKernel(kernel_graph, event_id, AnfAlgo::GetStreamId(node_after_recv));
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_cnode);
  auto process_iter = kernel_recv->find(node_after_recv);
  if (process_iter == kernel_recv->end()) {
    (*kernel_recv)[node_after_recv] = {recv_cnode};
  } else {
    process_iter->second.push_back(recv_cnode);
  }

  if (parallel_cnode == node_before_send) {
    kernel_graph->InsertSendRecvPairForParallelOpOutputs(parallel_cnode, std::make_pair(send_cnode, recv_cnode));
    MS_LOG(INFO) << "Generate send/recv for parallel op " << parallel_cnode->fullname_with_scope() << "'s output."
                 << "Send node " << send_cnode->fullname_with_scope() << " after "
                 << node_before_send->fullname_with_scope() << ", recv node " << recv_cnode->fullname_with_scope()
                 << " before " << node_after_recv->fullname_with_scope();
  } else {
    kernel_graph->InsertSendRecvPairForParallelOpInputs(parallel_cnode, std::make_pair(send_cnode, recv_cnode));
    MS_LOG(INFO) << "Generate send/recv for parallel op " << parallel_cnode->fullname_with_scope() << "'s input."
                 << "Send node " << send_cnode->fullname_with_scope() << " after "
                 << node_before_send->fullname_with_scope() << ", recv node " << recv_cnode->fullname_with_scope()
                 << " before " << node_after_recv->fullname_with_scope();
  }
}

void AscendStreamAssign::GenEventsForParallelOp(
  const NotNull<KernelGraphPtr> &kernel_graph, mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv) const {
  MS_LOG(DEBUG) << "Start GenEventsForParallelOp...";
  auto exec_kernels = kernel_graph->execution_order();
  mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> kernel_io_exec_info_map;
  GenKernelIoExecInfoMap(kernel_graph, &kernel_io_exec_info_map);
  for (auto &process_kernel : exec_kernels) {
    if (AnfAlgo::IsKernelSelectBackoffOp(process_kernel)) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(process_kernel);
    auto process_stream_id = AnfAlgo::GetStreamId(process_kernel);
    if (process_stream_id == kDefaultStreamIndex) {
      continue;
    }
    MS_LOG(DEBUG) << "Start GenEvents For ParallelOp " << process_kernel->fullname_with_scope();
    auto process_iter = kernel_io_exec_info_map.find(process_kernel);
    if (process_iter == kernel_io_exec_info_map.end()) {
      MS_LOG(INFO) << "Can't get node io execution info for  " << process_kernel->fullname_with_scope();
      continue;
    }
    auto process_io_exec_info = process_iter->second;
    InsertEventsForInputs(kernel_graph, process_kernel, process_io_exec_info, kernel_send, kernel_recv);
    InsertEventsForOutputs(kernel_graph, process_kernel, process_io_exec_info, kernel_send, kernel_recv);
  }
  MS_LOG(DEBUG) << "Finish GenEventsForParallelOp.";
}

void AscendStreamAssign::UpdateEventsToExecutionOrder(
  const NotNull<KernelGraphPtr> &kernel_graph,
  const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &send_after_node,
  const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &recv_before_node) const {
  MS_LOG(DEBUG) << "Start UpdateEventsToExecutionOrder...";
  auto exec_kernels = kernel_graph->execution_order();
  std::vector<CNodePtr> new_exec_orders;
  for (auto &kernel : exec_kernels) {
    auto before_iter = recv_before_node.find(kernel);
    if (before_iter != recv_before_node.end()) {
      (void)std::copy(before_iter->second.begin(), before_iter->second.end(), std::back_inserter(new_exec_orders));
    }
    new_exec_orders.push_back(kernel);
    auto after_iter = send_after_node.find(kernel);
    if (after_iter != send_after_node.end()) {
      (void)std::copy(after_iter->second.begin(), after_iter->second.end(), std::back_inserter(new_exec_orders));
    }
  }
  auto graph_output = kernel_graph->output();
  auto graph_output_iter = recv_before_node.find(graph_output);
  if (graph_output_iter != recv_before_node.end()) {
    (void)std::copy(graph_output_iter->second.begin(), graph_output_iter->second.end(),
                    std::back_inserter(new_exec_orders));
  }

  kernel_graph->set_execution_order(new_exec_orders);
  MS_LOG(DEBUG) << "Finish UpdateEventsToExecutionOrder.";
}

void AscendStreamAssign::InsertEventForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph) const {
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_send;
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_recv;
  AnfAlgo::SetStreamId(kDefaultStreamIndex, kernel_graph->output().get());
  GenEventsForParallelOp(kernel_graph, &kernel_send, &kernel_recv);
  UpdateEventsToExecutionOrder(kernel_graph, kernel_send, kernel_recv);
  InsertEventForMicroBatchIndependent(kernel_graph);
}

void AscendStreamAssign::AssignStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  const auto default_stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(default_stream);
  const auto comm_stream = AscendStreamMng::GetInstance().GetStream(kWorldGroupStreamIndex);
  MS_EXCEPTION_IF_NULL(comm_stream);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool enable_mem_offload =
    ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD) && !graph_ptr->is_dynamic_shape();
  if (enable_mem_offload || (graph_ptr->has_flag(kFlagPyNativeRunInGraph) && !graph_ptr->is_graph_run_mode())) {
    AssignStreamWithDefaultStream(graph_ptr.get());
    return;
  }
  if (graph_ptr->is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic shape do not need to assign stream.";
    return;
  }

  MS_LOG(INFO) << "Status record: start assign stream. graph id: " << graph_ptr->graph_id()
               << ", sink mode: " << IsTaskSink();
  PROF_START(assign_stream);
  if (!IsTaskSink()) {
    auto kernels = graph_ptr->execution_order();
    AssignStreamForNonTaskSink(kernels);
    InsertEventForNonTaskSink(graph_ptr);
    MS_LOG(INFO) << "After finish stream assign";
    graph_ptr->PrintGraphExecuteOrder();
    PROF_END(assign_stream);
    MS_LOG(INFO) << "Status record: end assign stream. graph id: " << graph_ptr->graph_id();
    return;
  }
  MS_LOG(INFO) << "Communication parallel mode: " << parallel::ParallelContext::GetInstance()->communi_parallel_mode()
               << ".";

  Reset();
  SetLoopSink();
  GetMaxStreamTaskNum();
  ReorderIndependentOrders(graph_ptr);
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  static const auto gen_mask_not_fusion = (common::GetEnv("GENMASK_NOT_FUSION") == "1");
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    TrailingTimeOptimizationByReorder(graph_ptr);
  }
  AssignAllNodesStream(graph_ptr);
  UpdateAtomicAddrCleanStreamId(graph_ptr);
  InsertStreamActive(graph_ptr);
  InsertEventForHcomParallel(graph_ptr);
  InsertEventForIndependentParallel(graph_ptr);
  if (gen_mask_not_fusion) {
    InsertEventForMicroBatchIndependent(graph_ptr);
  }
  GetIndependentMaxTarget(graph_ptr);
  InsertCtrlForIndependentParallel(graph_ptr);
  AdjustAtomicAddrCleanOrder(graph_ptr);
  GetNeedActiveStreams(graph_ptr);

  MS_LOG(INFO) << "After finish stream assign and before check resource assign:";
  graph_ptr->PrintGraphExecuteOrder();
  CheckResourceAssign(graph_ptr);

#ifdef ENABLE_DUMP_IR
  SubModuleId module = SubModuleId::SM_SESSION;
  std::string name = "assign_stream." + std::to_string(graph_ptr->graph_id());
  const std::vector<CNodePtr> &exec_order = graph_ptr->execution_order();
  (void)mindspore::RDR::RecordStreamExecOrder(module, name, exec_order);
#endif

  SetNodeStreamIDAttr(graph_ptr);
  FindStreamRelations(graph_ptr);
  PrintStreamRelations();
  GetStreamRelations();
  PrintStreamGroups();
  FindEventRelations(graph_ptr);
  PROF_END(assign_stream);
  MS_LOG(INFO) << "Status record: end assign stream. graph id: " << graph_ptr->graph_id();
}

void AscendStreamAssign::SetLoopSink() { loop_sink_ = KernelAdjust::NeedLoopSink(); }

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
    if (AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      (void)independents.emplace_back(cur_cnode_ptr);
    } else {
      (void)others.emplace_back(cur_cnode_ptr);
    }
  }

  if (others.empty() || independents.empty()) {
    MS_LOG(INFO) << "Independent or others is empty, no need reorder";
    return;
  }

  std::set<CNode *> processed;
  for (size_t i = 0; i < others.size(); i++) {
    auto begin = others.begin() + static_cast<std::vector<CNodePtr>::difference_type>(i);
    auto end = begin + 1;
    bool flag = false;
    for (size_t j = 0; j < independents.size(); j++) {
      auto cur_independent = independents[j];
      auto it = std::find(processed.cbegin(), processed.cend(), cur_independent.get());
      if (it != processed.end()) {
        continue;
      }

      auto res = FindFirstUserInExecutionOrder(begin, end, cur_independent, false);
      if (res != end) {
        flag = true;
        (void)exe_orders.emplace_back(cur_independent);
        (void)exe_orders.emplace_back(*begin);
        (void)processed.emplace(cur_independent.get());
        break;
      }
    }

    if (!flag) {
      (void)exe_orders.emplace_back(*begin);
    }
  }

  MS_LOG(INFO) << "After reorder, graph orders size:" << exe_orders.size();
  if (processed.size() != independents.size()) {
    MS_LOG(WARNING) << "Processed independent nodes size is not equal to exiting independent nodes size";
    return;
  }

  graph_ptr->set_execution_order(exe_orders);
}

void AscendStreamAssign::CheckScenario(const NotNull<KernelGraphPtr> &graph_ptr,
                                       vector<CNodePtr> *last_grad_and_status) const {
  MS_EXCEPTION_IF_NULL(last_grad_and_status);
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> hcom_nodes;
  auto overflow_marker = GetHcomAndOverflowMarker(graph_ptr, &hcom_nodes);
  if (hcom_nodes.size() < kHcomNum || overflow_marker == nullptr) {
    MS_LOG(INFO) << "Current model isn't in distribute or mix-precision mode, no optimization needed";
    last_grad_and_status->clear();
    return;
  }

  // if boost scene, disable reorder allreduce.
  for (const auto hcom_node : hcom_nodes) {
    if (common::AnfAlgo::HasNodeAttr(kAttrReuseCommunication, hcom_node)) {
      MS_LOG(INFO) << "Current graph has reuse hccl, no optimization needed!";
      last_grad_and_status->clear();
      return;
    }
  }

  auto overflow_marker_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), overflow_marker);
  auto last_hcom_ptr = hcom_nodes[hcom_nodes.size() - 1];
  auto last_hcom_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_hcom_ptr);
  auto last_grad_hcom_ptr = hcom_nodes[hcom_nodes.size() - kLastGradHcomOffset];
  auto last_grad_hcom_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_grad_hcom_ptr);
  if (last_grad_hcom_pos > overflow_marker_pos || last_hcom_pos < overflow_marker_pos) {
    MS_LOG(INFO) << "Grads average done after overflow judgement or status aren't allgathered, no optimization needed";
    last_grad_and_status->clear();
    return;
  }

  auto last_inputs = GetLastInputCnode(graph_ptr, last_grad_hcom_ptr);
  if (last_inputs.empty() || last_inputs.size() > 1 || IsHcom(last_inputs[0])) {
    MS_LOG(INFO) << "Inputs of last gradients allreduce is empty or include other allreduce, no optimization needed";
    last_grad_and_status->clear();
    return;
  }
  auto last_grad_ptr = last_inputs[0];
  MS_EXCEPTION_IF_NULL(last_grad_ptr);
  MS_LOG(DEBUG) << "Last Hcom: " << last_grad_hcom_ptr->fullname_with_scope()
                << "; last input: " << last_grad_ptr->fullname_with_scope();
  auto last_grad_hcom_graph_id = AnfAlgo::GetGraphId(last_grad_hcom_ptr.get());
  auto last_grad_graph_id = AnfAlgo::GetGraphId(last_grad_ptr.get());
  auto overflow_marker_graph_id = AnfAlgo::GetGraphId(overflow_marker.get());
  if (last_grad_graph_id != last_grad_hcom_graph_id || last_grad_graph_id != overflow_marker_graph_id) {
    MS_LOG(INFO) << "The grads and grad_hcom or overflow marker were not on the same subgraph, no optimization needed";
    last_grad_and_status->clear();
    return;
  }

  auto label_switch_pos = find_if(last_grad_hcom_pos, cnode_ptr_list.end(), [](const CNodePtr &node) -> bool {
    return common::AnfAlgo::GetCNodeName(node) == "LabelSwitch";
  });
  if (label_switch_pos == cnode_ptr_list.end()) {
    MS_LOG(INFO) << "No branches after getting overflow status, no optimization needed";
    last_grad_and_status->clear();
    return;
  }
  (void)last_grad_and_status->emplace_back(last_grad_ptr);
  (void)last_grad_and_status->emplace_back(overflow_marker);
  return;
}

CNodePtr AscendStreamAssign::GetCNodesNeededMoved(vector<CNodePtr> *moved_backward_cnodes,
                                                  vector<CNodePtr> *moved_forward_cnodes,
                                                  const vector<CNodePtr> &last_grad_and_status,
                                                  const NotNull<KernelGraphPtr> &graph_ptr) const {
  MS_EXCEPTION_IF_NULL(moved_backward_cnodes);
  MS_EXCEPTION_IF_NULL(moved_forward_cnodes);
  auto cnode_ptr_list = graph_ptr->execution_order();
  if (last_grad_and_status.size() != kLastGradAndStatusNum) {
    return nullptr;
  }
  auto last_grad_ptr = last_grad_and_status[0];
  auto float_status_ptr = last_grad_and_status[1];
  auto last_grad_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_grad_ptr);
  auto float_status_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), float_status_ptr);
  if (last_grad_pos == cnode_ptr_list.end() || float_status_pos == cnode_ptr_list.end()) {
    return nullptr;
  }
  auto graph_id = AnfAlgo::GetGraphId(last_grad_ptr.get());
  (void)moved_backward_cnodes->insert(moved_backward_cnodes->cend(), last_grad_pos + 1, float_status_pos);

  auto it = float_status_pos;
  while (AnfAlgo::GetGraphId((*it).get()) == graph_id && it < cnode_ptr_list.end()) {
    if (common::AnfAlgo::GetCNodeName(*it) == kAtomicAddrCleanOpName) {
      ++it;
      continue;
    }
    auto inputs = GetInputKernels(*it);
    bool is_independent = true;
    for (auto &input : inputs) {
      if (find(moved_backward_cnodes->begin(), moved_backward_cnodes->end(), input) != moved_backward_cnodes->end()) {
        is_independent = false;
        break;
      }
    }
    if (is_independent) {
      if (common::AnfAlgo::GetCNodeName(*(it - 1)) == kAtomicAddrCleanOpName) {
        (void)moved_forward_cnodes->emplace_back(*(it - 1));
      }
      (void)moved_forward_cnodes->emplace_back(*it);
    } else {
      if (common::AnfAlgo::GetCNodeName(*(it - 1)) == kAtomicAddrCleanOpName) {
        (void)moved_backward_cnodes->emplace_back(*(it - 1));
      }
      (void)moved_backward_cnodes->emplace_back(*it);
    }
    ++it;
  }

  size_t total_moved_size = LongToSize(it - last_grad_pos - 1);
  if (HasRefNodes(*moved_backward_cnodes) ||
      moved_backward_cnodes->size() + moved_forward_cnodes->size() != total_moved_size) {
    MS_LOG(INFO) << "Ref node was found or invalid number of moved nodes, give up optimization";
    return nullptr;
  }
  return GetTargetOutputNode(*moved_backward_cnodes, *it, graph_ptr);
}

CNodePtr AscendStreamAssign::GetTargetOutputNode(const vector<CNodePtr> &moved_backward_cnodes,
                                                 const CNodePtr first_node,
                                                 const NotNull<KernelGraphPtr> &graph_ptr) const {
  auto cnode_ptr_list = graph_ptr->execution_order();
  if (moved_backward_cnodes.empty() || !first_node) {
    return nullptr;
  }
  uint32_t subgraph_id = 0;
  bool get_subgraph_id = false;
  auto it = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), first_node);
  CNodePtr first_output_node_ptr = nullptr;
  while (!get_subgraph_id && it < cnode_ptr_list.end()) {
    auto inputs = GetInputKernels(*it);
    for (auto &input : inputs) {
      if (find(moved_backward_cnodes.begin(), moved_backward_cnodes.end(), input) != moved_backward_cnodes.end()) {
        get_subgraph_id = true;
        subgraph_id = AnfAlgo::GetGraphId((*it).get());
        first_output_node_ptr = *it;
        break;
      }
    }
    ++it;
  }
  if (subgraph_id == 0) {
    MS_LOG(INFO) << "The nodes moved backward were not used by any other nodes, no need moved";
    return nullptr;
  }

  for (; it < cnode_ptr_list.end(); ++it) {
    auto inputs = GetInputKernels(*it);
    for (auto &input : inputs) {
      if (find(moved_backward_cnodes.begin(), moved_backward_cnodes.end(), input) != moved_backward_cnodes.end() &&
          AnfAlgo::GetGraphId((*it).get()) != subgraph_id) {
        MS_LOG(INFO) << "The nodes moved backward were used by nodes on different subgraphs, no need moved";
        return nullptr;
      }
    }
  }
  return first_output_node_ptr;
}

bool AscendStreamAssign::FinetuneSubgraphExecOrder(vector<CNodePtr> *cnodes) const {
  MS_EXCEPTION_IF_NULL(cnodes);
  auto hcom_pos = find_if(cnodes->begin(), cnodes->end(), [](const CNodePtr &node_ptr) -> bool {
    return common::AnfAlgo::GetCNodeName(node_ptr) == "AllReduce";
  });
  if (hcom_pos == cnodes->end()) {
    return false;
  }
  CNodePtr hcom_ptr = *hcom_pos;

  vector<CNodePtr> ori_cnodes(cnodes->begin(), cnodes->end());
  cnodes->clear();
  vector<CNodePtr> atomic_addr_clean;
  for (auto iter = ori_cnodes.begin(); iter < ori_cnodes.end(); ++iter) {
    if (common::AnfAlgo::GetCNodeName(*iter) == kAtomicAddrCleanOpName) {
      (void)atomic_addr_clean.emplace_back(*iter);
      continue;
    }
    auto last_input_pos = cnodes->end();
    for (auto &input : GetInputKernels(*iter)) {
      auto pos = find(cnodes->begin(), cnodes->end(), input);
      if (pos != cnodes->end()) {
        last_input_pos = (last_input_pos == cnodes->end() || last_input_pos < pos) ? pos : last_input_pos;
      }
    }
    if (last_input_pos == cnodes->end()) {
      auto hcom_it = find(cnodes->begin(), cnodes->end(), hcom_ptr);
      if (hcom_it == cnodes->end() || common::AnfAlgo::GetCNodeName(*iter) == kLabelGotoOpName ||
          common::AnfAlgo::GetCNodeName(*iter) == kLabelSetOpName ||
          common::AnfAlgo::GetCNodeName(*iter) == kLabelSwitchOpName) {
        (void)cnodes->emplace_back(*iter);
      } else {
        (void)cnodes->insert(hcom_it, *iter);
      }
    } else {
      (void)cnodes->insert(last_input_pos + 1, *iter);
    }
  }

  for (auto &node : atomic_addr_clean) {
    auto first_input_pos = cnodes->end();
    for (auto &input : GetInputKernels(node)) {
      auto pos = find(cnodes->begin(), cnodes->end(), input);
      first_input_pos = (first_input_pos == cnodes->end() || first_input_pos > pos) ? pos : first_input_pos;
    }
    if (first_input_pos == cnodes->end()) {
      return false;
    } else {
      (void)cnodes->insert(first_input_pos, node);
    }
  }
  return cnodes->size() == ori_cnodes.size();
}

// performance optimization for trailing time in distribute mode
// allreduce of the last batch of gradients and the optimizer can be done parallel
void AscendStreamAssign::TrailingTimeOptimizationByReorder(const NotNull<KernelGraphPtr> &graph_ptr) const {
  vector<CNodePtr> last_grad_and_status;
  CheckScenario(graph_ptr, &last_grad_and_status);
  if (last_grad_and_status.empty()) {
    MS_LOG(INFO) << "Unsuitable scenario, no optimization needed";
    return;
  }

  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> moved_forward_cnodes;
  vector<CNodePtr> moved_backward_cnodes;
  CNodePtr first_output_ptr =
    GetCNodesNeededMoved(&moved_backward_cnodes, &moved_forward_cnodes, last_grad_and_status, graph_ptr);
  if (moved_backward_cnodes.empty() || first_output_ptr == nullptr) {
    MS_LOG(INFO) << "Unsuitable scenario, no optimization needed";
    return;
  }

  uint32_t subgraph_id = AnfAlgo::GetGraphId(first_output_ptr.get());
  auto last_grad_ptr = last_grad_and_status[0];
  auto last_grad_pos = find(cnode_ptr_list.begin(), cnode_ptr_list.end(), last_grad_ptr);
  vector<CNodePtr> cnodes(cnode_ptr_list.begin(), last_grad_pos + 1);
  (void)cnodes.insert(cnodes.cend(), moved_forward_cnodes.cbegin(), moved_forward_cnodes.cend());
  auto pos = last_grad_pos + static_cast<std::vector<CNodePtr>::difference_type>(moved_forward_cnodes.size() +
                                                                                 moved_backward_cnodes.size() + 1);
  while (pos < cnode_ptr_list.end() && AnfAlgo::GetGraphId((*pos).get()) != subgraph_id) {
    (void)cnodes.emplace_back(*pos);
    ++pos;
  }

  vector<CNodePtr> subgraph_cnodes;
  while (pos < cnode_ptr_list.end() && AnfAlgo::GetGraphId((*pos).get()) == subgraph_id) {
    if (common::AnfAlgo::GetCNodeName(*pos) == kLabelGotoOpName) {
      break;
    }
    if (*pos != first_output_ptr) {
      (void)subgraph_cnodes.emplace_back(*pos);
    } else {
      (void)subgraph_cnodes.insert(subgraph_cnodes.cend(), moved_backward_cnodes.cbegin(),
                                   moved_backward_cnodes.cend());
      (void)subgraph_cnodes.emplace_back(*pos);
    }
    ++pos;
  }

  if (!FinetuneSubgraphExecOrder(&subgraph_cnodes) || subgraph_cnodes.empty()) {
    MS_LOG(INFO) << "Finetune subgraph execute order failed, no optimization needed";
    return;
  }

  (void)cnodes.insert(cnodes.cend(), subgraph_cnodes.cbegin(), subgraph_cnodes.cend());
  (void)cnodes.insert(cnodes.cend(), static_cast<std::vector<CNodePtr>::const_iterator>(pos), cnode_ptr_list.cend());
  if (cnodes.size() != cnode_ptr_list.size()) {
    return;
  }
  for (auto &node : subgraph_cnodes) {
    AnfAlgo::SetGraphId(subgraph_id, node.get());
  }

  graph_ptr->set_execution_order(cnodes);
}

// section 2
void AscendStreamAssign::AssignAllNodesStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  std::vector<CNodePtr> common_node_list;
  std::vector<CNodePtr> hcom_node_list;
  std::vector<CNodePtr> independent_node_list;
  std::vector<CNodePtr> comm_sub_graph_node_list;
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  ClassifyNodeByKernel(graph_ptr, &common_node_list, &hcom_node_list, &independent_node_list,
                       &comm_sub_graph_node_list);

  // Assign Stream for common node
  common_stream_ = AssignNodeStreamInOrder(common_node_list);
  // Common stream assignment of GetNext-While and EOS is executed in kernel-adjust, so the common_stream_num is
  // acquired from resource manager rather than common_stream_.
  auto common_stream_num = resource_manager.cur_stream_num();

  // Assign Stream for comm subgraph hcom node
  // each comm subgraph hcom node gets its exclusive stream
  for (auto cur_cnode_ptr : comm_sub_graph_node_list) {
    if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
      MS_LOG(WARNING) << "Node: " << cur_cnode_ptr->DebugString() << " has been assigned stream id "
                      << AnfAlgo::GetStreamId(cur_cnode_ptr);
      continue;
    }
    uint32_t cur_stream_id = resource_manager.ApplyNewStream();
    AnfAlgo::SetStreamId(cur_stream_id, cur_cnode_ptr.get());
    (void)comm_sub_graph_stream_.insert(cur_stream_id);
  }

  // Assign Stream for hcom node
  std::map<std::string, std::map<uint32_t, std::vector<CNodePtr>>> group_graph_nodes_map;
  ClassifyNodeByGroupAndGraph(hcom_node_list, &group_graph_nodes_map);
  for (const auto &iter_group : std::as_const(group_graph_nodes_map)) {
    for (const auto &iter_graph : iter_group.second) {
      auto stream_set = AssignNodeStreamInOrder(iter_graph.second);
      hcom_stream_.insert(stream_set.cbegin(), stream_set.cend());
      group_hcom_graph_map_[iter_group.first][iter_graph.first] = stream_set;
    }
  }

  // Assign Stream for independent node
  std::map<uint32_t, std::vector<CNodePtr>> graph_nodes_map;
  ClassifyNodeByGraph(independent_node_list, &graph_nodes_map);
  for (auto iter_graph : std::as_const(graph_nodes_map)) {
    auto stream_set = AssignNodeStreamInOrder(iter_graph.second);
    independent_stream_.insert(stream_set.cbegin(), stream_set.cend());
    independent_graph_map_[iter_graph.first] = stream_set;
  }

  auto total_stream_num =
    resource_manager.cur_stream_num() +
    Uint32tMulWithOverflowCheck(static_cast<uint32_t>(hcom_stream_.size() + comm_sub_graph_stream_.size()),
                                kHcomSecondaryStreamNum);
  MS_LOG(INFO) << "Total stream number: " << total_stream_num << ", common stream number: " << common_stream_num
               << ", hcom stream number: " << hcom_stream_.size() << "*" << (kHcomSecondaryStreamNum + 1)
               << ", comm subgraph stream number: " << comm_sub_graph_stream_.size() << "*"
               << (kHcomSecondaryStreamNum + 1) << ", independent stream number: " << independent_stream_.size() << ".";

  if (total_stream_num > max_stream_count_) {
    MS_LOG(EXCEPTION) << "Total stream number " << total_stream_num << " exceeds the limit of " << max_stream_count_
                      << ", search details information in mindspore's FAQ.";
  }
}

void AscendStreamAssign::ClassifyNodeByKernel(const NotNull<KernelGraphPtr> &graph_ptr,
                                              std::vector<CNodePtr> *common_list, std::vector<CNodePtr> *hcom_list,
                                              std::vector<CNodePtr> *independent_list,
                                              std::vector<CNodePtr> *comm_sub_graph_list) const {
  MS_EXCEPTION_IF_NULL(common_list);
  MS_EXCEPTION_IF_NULL(hcom_list);
  MS_EXCEPTION_IF_NULL(independent_list);
  MS_EXCEPTION_IF_NULL(comm_sub_graph_list);
  for (auto cur_cnode : graph_ptr->execution_order()) {
    MS_EXCEPTION_IF_NULL(cur_cnode);
    if (AnfAlgo::IsKernelSelectBackoffOp(cur_cnode)) {
      MS_LOG(EXCEPTION) << "Not support kernel select backoff node in task sink mode, node:"
                        << cur_cnode->fullname_with_scope();
    }
    if (IsHcom(cur_cnode)) {
      if (IsCommSubGraph(graph_ptr, cur_cnode)) {
        comm_sub_graph_list->push_back(cur_cnode);
        continue;
      }
      hcom_list->push_back(cur_cnode);
      continue;
    }
    if (AnfAlgo::IsIndependentNode(cur_cnode)) {
      independent_list->push_back(cur_cnode);
      continue;
    }
    common_list->push_back(cur_cnode);
  }
}

void AscendStreamAssign::ClassifyNodeByGroupAndGraph(const std::vector<CNodePtr> hcom_list,
                                                     GroupGraphMap *group_graph_map) const {
  MS_EXCEPTION_IF_NULL(group_graph_map);
  for (auto cur_cnode_ptr : hcom_list) {
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (!IsHcom(cur_cnode_ptr)) {
      MS_LOG(EXCEPTION) << "Node is not hcom node, it's " << cur_cnode_ptr->fullname_with_scope();
    }
    auto group_name = GetHcomGroup(cur_cnode_ptr);
    auto hcom_graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
    auto iter = group_graph_map->find(group_name);
    if (iter == group_graph_map->end()) {
      std::map<uint32_t, std::vector<CNodePtr>> graph_nodes_map;
      graph_nodes_map[hcom_graph_id] = {cur_cnode_ptr};
      (*group_graph_map)[group_name] = graph_nodes_map;
    } else {
      auto &graph_nodes_map = iter->second;
      auto it = graph_nodes_map.find(hcom_graph_id);
      if (it == graph_nodes_map.cend()) {
        graph_nodes_map[hcom_graph_id] = {cur_cnode_ptr};
      } else {
        (void)it->second.emplace_back(cur_cnode_ptr);
      }
    }
  }
}

std::set<uint32_t> AscendStreamAssign::AssignNodeStreamInOrder(const std::vector<CNodePtr> node_list) const {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto cur_stream_id = resource_manager.ApplyNewStream();
  std::map<uint32_t, uint32_t> stream_task_map;
  std::set<uint32_t> stream_set;
  for (auto cur_cnode_ptr : node_list) {
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::GetStreamId(cur_cnode_ptr) != kInvalidStreamId) {
      continue;
    }
    auto task_num = GetNodeTaskNum(cur_cnode_ptr);
    auto it = stream_task_map.find(cur_stream_id);
    if (it == stream_task_map.end()) {
      AnfAlgo::SetStreamId(cur_stream_id, cur_cnode_ptr.get());
      (void)stream_task_map.emplace(cur_stream_id, task_num);
      (void)stream_set.emplace(cur_stream_id);
    } else {
      if (it->second <= max_task_count_ - task_num) {
        AnfAlgo::SetStreamId(it->first, cur_cnode_ptr.get());
        it->second = Uint32tAddWithOverflowCheck(it->second, task_num);
      } else {
        cur_stream_id = resource_manager.ApplyNewStream();
        AnfAlgo::SetStreamId(cur_stream_id, cur_cnode_ptr.get());
        (void)stream_task_map.emplace(cur_stream_id, task_num);
        (void)stream_set.emplace(cur_stream_id);
      }
    }
  }
  if (stream_set.empty()) {
    resource_manager.DeleteStream();
  }
  return stream_set;
}

void AscendStreamAssign::ClassifyNodeByGraph(const std::vector<CNodePtr> indepent_list,
                                             std::map<uint32_t, std::vector<CNodePtr>> *graph_nodes_map) const {
  MS_EXCEPTION_IF_NULL(graph_nodes_map);
  for (auto cur_cnode_ptr : indepent_list) {
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (!AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      MS_LOG(EXCEPTION) << "Node is not independent node, it's " << cur_cnode_ptr->fullname_with_scope();
    }
    auto independent_graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
    auto it = graph_nodes_map->find(independent_graph_id);
    if (it == graph_nodes_map->cend()) {
      (*graph_nodes_map)[independent_graph_id] = {cur_cnode_ptr};
    } else {
      (void)it->second.emplace_back(cur_cnode_ptr);
    }
  }
}

uint32_t AscendStreamAssign::GetNodeTaskNum(const CNodePtr &cnode) const {
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  if (node_name == kHcomSendOpName || node_name == kReceiveOpName) {
    return kTaskNumPerHcomSendRecvNode;
  }
  return IsHcom(cnode) ? GetHcomTaskNum(cnode) : kTaskNumPerCommonNode;
}

// section 3
void AscendStreamAssign::UpdateAtomicAddrCleanStreamId(const NotNull<KernelGraphPtr> &graph_ptr) const {
  MS_LOG(INFO) << "Start";
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    // update AtomicAddrClean stream same with the next node
    if (i > 0 && common::AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]) == kAtomicAddrCleanOpName) {
      AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(cur_cnode_ptr), cnode_ptr_list[i - 1].get());
    }
  }
  MS_LOG(INFO) << "End";
}

// section 4
void AscendStreamAssign::InsertStreamActive(const NotNull<KernelGraphPtr> &graph_ptr) {
  InsertStreamActiveForCommon(graph_ptr);
  InsertStreamActiveForIndependent(graph_ptr);
  InsertStreamActiveForParallel(graph_ptr);
  InsertStreamActiveForCommSubGraph(graph_ptr);
}

void AscendStreamAssign::InsertStreamActiveForCommSubGraph(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (comm_sub_graph_stream_.empty()) {
    MS_LOG(INFO) << "No comm sub graph, skip.";
    return;
  }

  std::vector<CNodePtr> update_cnode_list;
  const auto &exe_orders = graph_ptr->execution_order();
  if (!exe_orders.empty()) {
    (void)update_cnode_list.emplace_back(exe_orders[0]);
  }

  for (size_t i = 1; i < exe_orders.size(); ++i) {
    const CNodePtr &cur_cnode_ptr = exe_orders[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (comm_sub_graph_stream_.find(cur_stream_id) == comm_sub_graph_stream_.end()) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }
    const CNodePtr &pre_cnode_ptr = exe_orders[i - 1];
    MS_EXCEPTION_IF_NULL(pre_cnode_ptr);
    uint32_t pre_stream_id = AnfAlgo::GetStreamId(pre_cnode_ptr);
    MS_LOG(INFO) << "Insert StreamActive for node " << cur_cnode_ptr->DebugString()
                 << " from stream id:" << pre_stream_id << " to stream id:" << cur_stream_id;
    CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
    AnfAlgo::SetStreamId(pre_stream_id, active_ptr.get());
    // active hcom stream
    common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>({cur_stream_id}), active_ptr);
    (void)update_cnode_list.emplace_back(active_ptr);
    (void)update_cnode_list.emplace_back(cur_cnode_ptr);
  }

  graph_ptr->set_execution_order(update_cnode_list);
}

void AscendStreamAssign::InsertStreamActiveForParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (group_hcom_graph_map_.empty() && independent_graph_map_.empty()) {
    MS_LOG(INFO) << "Hcom and independent is empty";
    return;
  }
  auto root_graph_id = graph_ptr->graph_id();
  if (root_graph_id == kInvalidGraphId) {
    MS_LOG(INFO) << "Root graph id is invalid";
    return;
  }

  std::map<uint32_t, std::set<uint32_t>> other_graph;
  std::set<uint32_t> hcom_streams;
  for (const auto &graph_nodes : std::as_const(group_hcom_graph_map_)) {
    for (const auto &item : graph_nodes.second) {
      MS_LOG(INFO) << "Graph id:" << item.first;
      if (item.first == root_graph_id) {
        if (loop_sink_) {
          hcom_streams.insert(item.second.cbegin(), item.second.cend());
        }
      } else {
        auto it = other_graph.find(item.first);
        if (it == other_graph.cend()) {
          other_graph[item.first] = item.second;
        } else {
          for (const auto &stream : std::as_const(item.second)) {
            (void)it->second.emplace(stream);
          }
        }
      }
    }
  }

  if (!hcom_streams.empty()) {
    ActiveRootGraphHcom(graph_ptr, hcom_streams);
  }

  MS_LOG(INFO) << "Independent graph map size:" << independent_graph_map_.size();
  for (const auto &item : std::as_const(independent_graph_map_)) {
    MS_LOG(DEBUG) << "Graph id:" << item.first;
    if (item.first == root_graph_id) {
      if (loop_sink_) {
        ActiveRootGraphIndependent(graph_ptr, item.second);
      }
    } else {
      auto it = other_graph.find(item.first);
      if (it == other_graph.cend()) {
        other_graph[item.first] = item.second;
      } else {
        for (const auto &stream : std::as_const(item.second)) {
          (void)it->second.emplace(stream);
        }
      }
    }
  }

  ActiveOtherGraphParallel(graph_ptr, other_graph);
}

void AscendStreamAssign::ActiveOtherGraphParallel(const NotNull<KernelGraphPtr> &graph_ptr,
                                                  std::map<uint32_t, std::set<uint32_t>> other_graph) const {
  MS_LOG(INFO) << "Other graph size:" << other_graph.size();
  if (other_graph.empty()) {
    return;
  }

  auto root_graph_id = graph_ptr->graph_id();

  std::vector<CNodePtr> update_stream_list;
  auto exe_order = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_order.size(); i++) {
    auto cur_cnode_ptr = exe_order[i];
    auto cur_graph_id = AnfAlgo::GetGraphId(cur_cnode_ptr.get());
    if (cur_graph_id == root_graph_id) {
      (void)update_stream_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    std::map<uint32_t, std::set<uint32_t>>::const_iterator it = other_graph.find(cur_graph_id);
    if (it == other_graph.cend()) {
      (void)update_stream_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
    // 1.set stream id
    AnfAlgo::SetStreamId(cur_stream_id, active_ptr.get());
    // 2.set active stream ids
    std::vector<uint32_t> active_index_list;
    (void)std::copy(it->second.begin(), it->second.end(), std::back_inserter(active_index_list));
    common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list),
                                 active_ptr);

    // find position for insert streamactive
    if (common::AnfAlgo::GetCNodeName(cur_cnode_ptr) == kLabelSetOpName) {
      (void)update_stream_list.emplace_back(cur_cnode_ptr);
      (void)update_stream_list.emplace_back(active_ptr);
    } else {
      (void)update_stream_list.emplace_back(active_ptr);
      (void)update_stream_list.emplace_back(cur_cnode_ptr);
    }
    (void)other_graph.erase(it);
  }
  graph_ptr->set_execution_order(update_stream_list);
}

void AscendStreamAssign::ActiveRootGraphHcom(const NotNull<KernelGraphPtr> &graph_ptr,
                                             const std::set<uint32_t> &hcom_streams) {
  MS_LOG(INFO) << "Active root graph hcom start";
  std::vector<CNodePtr> update_cnode_list;
  auto exe_orders = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_orders.size(); i++) {
    CNodePtr cur_cnode_ptr = exe_orders[i];
    if (common::AnfAlgo::GetCNodeName(cur_cnode_ptr) != kStreamSwitchOpName) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (!common::AnfAlgo::HasNodeAttr(kAttrStreamSwitchKind, cur_cnode_ptr)) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto kind = common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrStreamSwitchKind);
    if (kind != static_cast<uint32_t>(kFpBpStreamSwitch)) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto true_stream_id = common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrTrueBranchStream);
    MS_LOG(INFO) << "FpBpStreamswtich stream id:" << AnfAlgo::GetStreamId(cur_cnode_ptr)
                 << "; true branch stream id:" << true_stream_id;
    CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
    AnfAlgo::SetStreamId(true_stream_id, active_ptr.get());
    vector<uint32_t> active_ids;
    // active hcom stream
    (void)std::copy(hcom_streams.begin(), hcom_streams.end(), std::back_inserter(active_ids));
    common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_ids), active_ptr);
    (void)update_cnode_list.emplace_back(cur_cnode_ptr);
    (void)update_cnode_list.emplace_back(active_ptr);
    (void)std::copy(exe_orders.begin() + static_cast<std::vector<CNodePtr>::difference_type>(i + 1), exe_orders.end(),
                    std::back_inserter(update_cnode_list));
    break;
  }

  hcom_stream_activated_ = true;
  graph_ptr->set_execution_order(update_cnode_list);
}

void AscendStreamAssign::ActiveRootGraphIndependent(const NotNull<KernelGraphPtr> &graph_ptr,
                                                    const std::set<uint32_t> &independent_streams) {
  MS_LOG(DEBUG) << "Start active root graph independent";
  std::vector<CNodePtr> update_cnode_list;
  auto exe_orders = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_orders.size(); i++) {
    CNodePtr cur_cnode_ptr = exe_orders[i];
    if (common::AnfAlgo::GetCNodeName(cur_cnode_ptr) != kStreamSwitchOpName) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (!common::AnfAlgo::HasNodeAttr(kAttrStreamSwitchKind, cur_cnode_ptr)) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    auto kind = common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrStreamSwitchKind);
    if (kind != static_cast<uint32_t>(kIndependentStreamSwitch)) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    // first independetn stream id is minimum and order by std map;
    auto first_independent_stream = *(independent_streams.cbegin());
    common::AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(first_independent_stream), cur_cnode_ptr);
    (void)update_cnode_list.emplace_back(cur_cnode_ptr);
    (void)std::copy(exe_orders.begin() + static_cast<std::vector<CNodePtr>::difference_type>(i + 1), exe_orders.end(),
                    std::back_inserter(update_cnode_list));
    break;
  }

  independent_stream_activated_ = true;
  graph_ptr->set_execution_order(update_cnode_list);
}

void AscendStreamAssign::InsertStreamActiveForCommon(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  GetProcessedStream(graph_ptr);
  std::vector<CNodePtr> update_cnode_list;
  CNodePtr cur_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;

  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (IsHcom(cur_cnode_ptr)) {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
      continue;
    }
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    bool processed = std::any_of(processed_streams_.cbegin(), processed_streams_.cend(),
                                 [cur_stream_id](uint32_t iter_stream) { return iter_stream == cur_stream_id; });
    // 1)inner stream assign, need insert active op
    if (!processed) {
      MS_LOG(INFO) << "Common stream active info:" << pre_stream_id << "->active" << cur_stream_id;
      CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
      // 1.set stream id
      AnfAlgo::SetStreamId(pre_stream_id, active_ptr.get());
      // 2.set active stream ids
      std::vector<uint32_t> active_index_list{cur_stream_id};
      common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list),
                                   active_ptr);
      if (i > 0) {
        auto pre_node = common::AnfAlgo::GetCNodeName(cnode_ptr_list[i - 1]);
        if (pre_node == kLabelSwitchOpName || pre_node == kLabelGotoOpName) {
          (void)update_cnode_list.insert(update_cnode_list.cend() - 1, active_ptr);
          AnfAlgo::SetStreamId(cur_stream_id, cnode_ptr_list[i - 1].get());
        } else {
          (void)update_cnode_list.emplace_back(active_ptr);
        }
      }
    }
    if (common::AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName) {
      MS_LOG(INFO) << "Insert StreamActive op after FP StreamSwitch for stream parallel";
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
    } else {
      (void)update_cnode_list.emplace_back(cur_cnode_ptr);
    }

    (void)processed_streams_.emplace(cur_stream_id);
    pre_stream_id = cur_stream_id;
  }
  graph_ptr->set_execution_order(update_cnode_list);
}

vector<CNodePtr> AscendStreamAssign::GetIndependentNodesNeedsInsertActive(const std::vector<CNodePtr> exe_orders,
                                                                          const uint32_t graph_id) const {
  uint32_t pre_stream_id = 0;
  std::vector<CNodePtr> independent_nodes_need_insert_active;
  CNodePtr pre_cnode;
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode_ptr = exe_orders[i];
    if (!AnfAlgo::IsIndependentNode(cur_cnode_ptr)) {
      continue;
    }
    if (AnfAlgo::GetGraphId(cur_cnode_ptr.get()) != graph_id) {
      continue;
    }
    auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (pre_stream_id == 0) {
      pre_stream_id = cur_stream_id;
      pre_cnode = cur_cnode_ptr;
    }
    if (cur_stream_id != pre_stream_id) {
      independent_nodes_need_insert_active.push_back(pre_cnode);
    }
    pre_stream_id = cur_stream_id;
    pre_cnode = cur_cnode_ptr;
  }
  return independent_nodes_need_insert_active;
}

void AscendStreamAssign::InsertStreamActiveForIndependent(const NotNull<KernelGraphPtr> &graph_ptr) {
  auto root_graph_id = graph_ptr->graph_id();
  if (root_graph_id == kInvalidGraphId) {
    return;
  }
  std::set<uint32_t> independent_streams;
  for (const auto &item : std::as_const(independent_graph_map_)) {
    if (item.first != root_graph_id) {
      continue;
    }
    independent_streams = item.second;
  }

  // Root graph independent stream size is not more than one, no need insert active
  if (independent_streams.size() <= 1) {
    return;
  }
  std::vector<CNodePtr> update_cnode_list;
  auto exe_orders = graph_ptr->execution_order();
  auto independent_nodes_need_insert_active = GetIndependentNodesNeedsInsertActive(exe_orders, root_graph_id);
  // first independent is been activated, active other independent stream
  std::vector<uint32_t> streams;
  (void)std::copy(independent_streams.begin(), independent_streams.end(), std::back_inserter(streams));
  std::sort(streams.begin(), streams.end());
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode_ptr = exe_orders[i];
    (void)update_cnode_list.emplace_back(cur_cnode_ptr);
    if (!AnfAlgo::IsIndependentNode(cur_cnode_ptr) || AnfAlgo::GetGraphId(cur_cnode_ptr.get()) != root_graph_id) {
      continue;
    }

    auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    auto it = std::find(streams.begin(), streams.end(), cur_stream_id);
    if (it == streams.end()) {
      MS_LOG(EXCEPTION) << "Can't find independent stream id:" << cur_stream_id;
    } else if (it == streams.end() - 1) {
      (void)std::copy(exe_orders.begin() + static_cast<std::vector<CNodePtr>::difference_type>(i + 1), exe_orders.end(),
                      std::back_inserter(update_cnode_list));
      break;
    } else {
      if (std::find(independent_nodes_need_insert_active.begin(), independent_nodes_need_insert_active.end(),
                    cur_cnode_ptr) != independent_nodes_need_insert_active.end()) {
        CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
        // 1.set stream id
        AnfAlgo::SetStreamId(cur_stream_id, active_ptr.get());
        // 2.set active stream ids
        std::vector<uint32_t> active_index_list{*(it + 1)};
        common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list),
                                     active_ptr);
        (void)update_cnode_list.emplace_back(active_ptr);
      }
    }
  }
  graph_ptr->set_execution_order(update_cnode_list);
}

void AscendStreamAssign::GetProcessedStream(const NotNull<KernelGraphPtr> &graph_ptr) {
  // 0 stream is activated at first
  (void)processed_streams_.emplace(0);
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    auto cur_cnode_ptr = cnode_ptr_list[i];
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);

    if (common::AnfAlgo::GetCNodeName(cur_cnode_ptr) == kStreamSwitchOpName) {
      if (common::AnfAlgo::HasNodeAttr(kAttrTrueBranchStream, cur_cnode_ptr)) {
        auto true_stream_id = common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrTrueBranchStream);
        (void)processed_streams_.emplace(true_stream_id);
      }

      if (!common::AnfAlgo::HasNodeAttr(kStreamNeedActivedFirst, cur_cnode_ptr)) {
        continue;
      }
      auto need_active = common::AnfAlgo::GetNodeAttr<bool>(cur_cnode_ptr, kStreamNeedActivedFirst);
      if (need_active) {
        (void)processed_streams_.emplace(cur_stream_id);
      }
    }
  }
  for (const auto &item : std::as_const(processed_streams_)) {
    MS_LOG(INFO) << "Before active:" << item << " is been processed";
  }
}

bool AscendStreamAssign::IsAllOutGraphOut(const KernelGraphPtr &graph, const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto cnode_out_num = AnfAlgo::GetOutputTensorNum(cnode);
  auto nodes = common::AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  std::set<int> output_index_set;
  // Assign Communicate Op Memory firstly.
  for (const auto &node : nodes) {
    auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(node, 0, true);
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (!item_with_index.first->isa<CNode>() || !AnfUtils::IsRealKernel(item_with_index.first)) {
      continue;
    }
    if (item_with_index.first == cnode) {
      (void)output_index_set.insert(item_with_index.second);
    }
  }

  MS_LOG(INFO) << "Node " << cnode->fullname_with_scope() << " has " << cnode_out_num
               << " outputs, in graph output num:" << output_index_set.size();
  return cnode_out_num == output_index_set.size();
}

// section5
void AscendStreamAssign::InsertEventForHcomParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  InsertEventCommonDependHcom(graph_ptr);
  InsertEventForIndependentHcom(graph_ptr);
  InsertEventHcomDependCommonBak(graph_ptr);
  InsertEventHcomDependHcom(graph_ptr);
  InsertEventForCallCommSubGraph(graph_ptr);
  MS_LOG(INFO) << "End";
}

bool AscendStreamAssign::ExistStreamSendAfterLastHcomNode(const NotNull<KernelGraphPtr> &graph_ptr,
                                                          uint32_t graph_id) const {
  auto cnodes = graph_ptr->execution_order();
  for (int64_t i = static_cast<int64_t>(cnodes.size() - 1); i >= 0; --i) {
    if (AnfAlgo::GetGraphId(cnodes[static_cast<size_t>(i)].get()) == graph_id &&
        IsHcom(cnodes[static_cast<size_t>(i)])) {
      return (common::AnfAlgo::GetCNodeName(cnodes[static_cast<size_t>(i)]) == kSendOpName) ||
             ((i < SizeToLong(cnodes.size() - 1)) &&
              common::AnfAlgo::GetCNodeName(cnodes[static_cast<size_t>(i + 1)]) == kSendOpName);
    }
  }
  MS_LOG(INFO) << "There is no hcom nodes in graph " << graph_id << ", root graph: " << graph_ptr->graph_id();
  return true;
}

void AscendStreamAssign::InsertRecvForLoopSink(const NotNull<KernelGraphPtr> &root_graph, std::vector<CNodePtr> *cnodes,
                                               uint32_t cur_event_id, uint32_t graph_id) const {
  MS_EXCEPTION_IF_NULL(cnodes);
  std::set<std::string> ending_nodes = {kStreamActiveOpName, kLabelGotoOpName};
  for (auto iter = cnodes->end() - 1; iter >= cnodes->begin(); --iter) {
    if (AnfAlgo::GetGraphId((*iter).get()) != graph_id) {
      continue;
    }
    auto node_name = common::AnfAlgo::GetCNodeName(*iter);
    auto cnode = (*iter)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    CNodePtr recv_cnode = CreateRecvApplyKernel(root_graph, cur_event_id, AnfAlgo::GetStreamId(cnode));
    // insert StreamRecv node before the last node in the graph if the node is <StreamActive, LabelGoto> or insert
    // StreamRecv node after the last node, at the same time, the next node of the last not in the graph is LabelSet.
    if (ending_nodes.find(node_name) != ending_nodes.end()) {
      MS_LOG(INFO) << "Insert StreamRecv " << cur_event_id << " before node: " << (*iter)->fullname_with_scope();
      (void)cnodes->insert(iter, recv_cnode);
      break;
    } else if ((iter < cnodes->end() - 1) && common::AnfAlgo::GetCNodeName(*(iter + 1)) == kLabelSetOpName) {
      MS_LOG(INFO) << "Insert StreamRecv " << cur_event_id << " after node: " << (*iter)->fullname_with_scope();
      (void)cnodes->insert(iter + 1, recv_cnode);
      break;
    } else {
      MS_LOG(EXCEPTION) << "The last node of graph " << graph_id
                        << " is not in the set <StreamActive, LabelGoto>, whereas is " << (*iter)->fullname_with_scope()
                        << ", and check whether the next node exists and is LabelSet.";
    }
  }
}

void AscendStreamAssign::InsertRecvForNotLoopSink(const NotNull<KernelGraphPtr> &root_graph,
                                                  std::vector<CNodePtr> *cnodes, uint32_t cur_event_id,
                                                  uint32_t graph_id) const {
  uint32_t stream_id = UINT32_MAX;
  for (auto iter = cnodes->end() - 1; iter >= cnodes->begin(); --iter) {
    if (AnfAlgo::GetGraphId((*iter).get()) != graph_id) {
      continue;
    }
    if (IsHcom(*iter) || IsPrimitiveCNode(*iter, prim::kPrimStreamSend) ||
        IsPrimitiveCNode(*iter, prim::kPrimStreamRecv)) {
      continue;
    }
    stream_id = AnfAlgo::GetStreamId(*iter);
  }
  if (stream_id == UINT32_MAX) {
    MS_LOG(EXCEPTION) << "Can not find compute node in graph " << graph_id;
  }

  for (auto iter = cnodes->end() - 1; iter >= cnodes->begin(); --iter) {
    if (AnfAlgo::GetGraphId((*iter).get()) != graph_id) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(*iter);
    CNodePtr recv_cnode = CreateRecvApplyKernel(root_graph, cur_event_id, stream_id);
    MS_LOG(INFO) << "Insert StreamRecv " << cur_event_id << " after node: " << (*iter)->fullname_with_scope();
    (void)cnodes->insert(iter + 1, recv_cnode);
    break;
  }
}

void AscendStreamAssign::GraphLoopSync(const NotNull<KernelGraphPtr> &root_graph, uint32_t graph_id) const {
  if (ExistStreamSendAfterLastHcomNode(root_graph, graph_id)) {
    return;
  }
  MS_LOG(WARNING) << "There is no event between computing stream and hcom stream in graph " << graph_id
                  << " need insert event.";

  auto cnodes = root_graph->execution_order();
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();

  // insert StreamSend node after the last hcom node
  for (auto iter = cnodes.end() - 1; iter >= cnodes.begin(); --iter) {
    if (IsHcom(*iter) && AnfAlgo::GetGraphId((*iter).get()) == graph_id) {
      CNodePtr send_cnode = CreateSendApplyKernel(root_graph, cur_event_id, AnfAlgo::GetStreamId((*iter)));
      MS_LOG(INFO) << "Insert StreamSend " << cur_event_id << " after node: " << (*iter)->fullname_with_scope();
      (void)cnodes.insert(iter + 1, send_cnode);
      break;
    }
  }
  if (KernelAdjust::NeedLoopSink()) {
    InsertRecvForLoopSink(root_graph, &cnodes, cur_event_id, graph_id);
  } else {
    InsertRecvForNotLoopSink(root_graph, &cnodes, cur_event_id, graph_id);
  }

  root_graph->set_execution_order(cnodes);
}

void AscendStreamAssign::GetAllGraphID(const NotNull<KernelGraphPtr> &graph_ptr, std::vector<uint32_t> *graphs_id) {
  MS_EXCEPTION_IF_NULL(graphs_id);
  if (std::find(graphs_id->begin(), graphs_id->end(), graph_ptr->graph_id()) != graphs_id->end()) {
    return;
  }
  graphs_id->push_back(graph_ptr->graph_id());
  for (auto child_graph : graph_ptr->child_graph_order()) {
    GetAllGraphID(NOT_NULL(child_graph.lock()), graphs_id);
  }
}
// Application scenario: In the loop sink mode, the last communication operator did not send 'send' to the calculation
// stream, causing the next loop to start without waiting for the end of the communication stream.
// Solution: In the above scenario, insert 'send' after the last communication operator, and insert 'recv' before the
//  `active` operator in the calculation stream to ensure loop synchronization.
void AscendStreamAssign::InsertEventForIndependentHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  std::vector<uint32_t> graphs_id;
  GetAllGraphID(graph_ptr, &graphs_id);
  for (auto graph_id : graphs_id) {
    GraphLoopSync(graph_ptr, graph_id);
  }
}

void AscendStreamAssign::InsertEventCommonDependHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  auto it = cnodes.begin();
  while (it != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    if (IsHcom(*it)) {
      auto cur_hcom_node = *it;
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));
      it = cnodes.insert(it + 1, send_cnode_ptr);

      auto target = FindFirstUserInExecutionOrder(it, cnodes.end(), cur_hcom_node, true);
      if (target == cnodes.end()) {
        if (IsAllOutGraphOut(graph_ptr, cur_hcom_node)) {
          // if hcom's all output is graph output, we need to insert send/recv to fpbp end in data sink mode
          target = std::find_if(it, cnodes.end(), [](const CNodePtr &temp_node) {
            return common::AnfAlgo::HasNodeAttr(kAttrFpBpEnd, temp_node);
          });
        }

        if (target == cnodes.end()) {
          MS_EXCEPTION_IF_NULL(*(it - 1));
          MS_LOG(WARNING) << "Hcom node:" << (*(it - 1))->fullname_with_scope()
                          << ", can't find target for insert recv op, no insert send/recv";
          it = cnodes.erase(it);
          continue;
        }
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
  MS_LOG(INFO) << "After common depend hcom, total event nums:" << resource_manager.cur_event_num();
}

// after memory reuse is correct, use this function
void AscendStreamAssign::InsertEventHcomDependCommonBak(const NotNull<KernelGraphPtr> &graph_ptr) const {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes;
  CNodePtr cur_cnode_ptr = nullptr;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (i == 0) {
      (void)cnodes.emplace_back(cur_cnode_ptr);
      continue;
    }

    if (!IsHcom(cur_cnode_ptr)) {
      (void)cnodes.emplace_back(cur_cnode_ptr);
      continue;
    }

    // get the input which located in the last exe orders
    vector<CNodePtr> inputs_cnode = GetLastInputCnode(graph_ptr, cur_cnode_ptr);
    if (inputs_cnode.empty()) {
      (void)cnodes.emplace_back(cur_cnode_ptr);
      MS_LOG(WARNING) << "Hcom op:" << common::AnfAlgo::GetCNodeName(cur_cnode_ptr) << " can't find inputs nodes";
      continue;
    }

    MS_LOG(INFO) << "Current hcom:" << common::AnfAlgo::GetCNodeName(cur_cnode_ptr)
                 << "; inputs cnode size:" << inputs_cnode.size();

    for (size_t j = 0; j < inputs_cnode.size(); j++) {
      auto &cur_input = inputs_cnode.at(j);
      MS_LOG(INFO) << "The index:" << j << " input, name:" << common::AnfAlgo::GetCNodeName(cur_input);
      uint32_t cur_event_id = resource_manager.ApplyNewEvent();
      auto pre_stream_id = AnfAlgo::GetStreamId(cur_input);
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, pre_stream_id);
      auto it = std::find(cnodes.begin(), cnodes.end(), cur_input);
      if (it == cnodes.end()) {
        MS_LOG(EXCEPTION) << "Hcom:" << common::AnfAlgo::GetCNodeName(cur_cnode_ptr)
                          << " can't find input node:" << common::AnfAlgo::GetCNodeName(cur_input);
      }
      (void)cnodes.insert(it + 1, send);
      uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_stream_id);
      (void)cnodes.emplace_back(recv);
      (void)cnodes.emplace_back(cur_cnode_ptr);
    }
  }

  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After hcom depend common, total event nums:" << resource_manager.cur_event_num();
}

vector<CNodePtr> AscendStreamAssign::GetLastInputCnode(const NotNull<KernelGraphPtr> &graph_ptr,
                                                       const CNodePtr &cur_cnode_ptr) const {
  auto group_name = GetHcomGroup(cur_cnode_ptr);
  auto input_cnodes = GetInputKernels(cur_cnode_ptr);
  if (input_cnodes.empty()) {
    return {};
  }
  // record max index node for each stream
  std::map<uint32_t, std::pair<CNodePtr, uint32_t>> result;
  for (size_t i = 0; i < input_cnodes.size(); i++) {
    auto &cur_input = input_cnodes.at(i);
    auto stream_id = AnfAlgo::GetStreamId(cur_input);
    auto cur_index = GetIndexByKey(graph_ptr, cur_input.get());
    if (cur_index == UINT32_MAX) {
      MS_LOG(EXCEPTION) << "The input node:" << common::AnfAlgo::GetCNodeName(cur_input) << " is not found in graph";
    }
    std::map<uint32_t, std::pair<CNodePtr, uint32_t>>::const_iterator it = result.find(stream_id);
    if (it == result.cend()) {
      result[stream_id] = std::make_pair(cur_input, cur_index);
    } else {
      auto max_index = it->second.second;
      if (cur_index > max_index) {
        result[stream_id] = std::make_pair(cur_input, cur_index);
      }
    }
  }

  vector<CNodePtr> final_inputs;
  CNodePtr max_common_cnode = nullptr;
  for (const auto &item : std::as_const(result)) {
    if (IsHcom(item.second.first)) {
      auto cur_group = GetHcomGroup(item.second.first);
      if (cur_group == group_name) {
        continue;
      } else {
        (void)final_inputs.emplace_back(item.second.first);
      }
    } else {
      max_common_cnode = item.second.first;
    }
  }

  if (max_common_cnode != nullptr) {
    (void)final_inputs.emplace_back(max_common_cnode);
  }
  return final_inputs;
}

vector<CNodePtr> AscendStreamAssign::GetInputKernels(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  vector<CNodePtr> input_cnodes;
  queue<CNodePtr> nop_nodes;
  auto inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    auto real_input = common::AnfAlgo::VisitKernel(inputs[i], 0);
    auto node = real_input.first;
    MS_EXCEPTION_IF_NULL(node);
    if (common::AnfAlgo::IsNopNode(node)) {
      nop_nodes.push(node->cast<CNodePtr>());
      while (!nop_nodes.empty()) {
        auto cur_node = nop_nodes.front();
        nop_nodes.pop();
        auto new_inputs = cur_node->inputs();
        for (size_t j = 1; j < new_inputs.size(); j++) {
          auto new_real_input = common::AnfAlgo::VisitKernel(new_inputs[j], 0);
          auto new_node = new_real_input.first;
          MS_EXCEPTION_IF_NULL(new_node);
          if (common::AnfAlgo::IsNopNode(new_node)) {
            nop_nodes.push(new_node->cast<CNodePtr>());
          } else if (new_node->isa<CNode>()) {
            (void)input_cnodes.emplace_back(new_node->cast<CNodePtr>());
          }
        }
      }
    } else if (node->isa<CNode>()) {
      (void)input_cnodes.emplace_back(node->cast<CNodePtr>());
    }
  }
  return input_cnodes;
}

void AscendStreamAssign::InsertEventHcomDependCommon(const NotNull<KernelGraphPtr> &graph_ptr) const {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes;
  CNodePtr cur_cnode_ptr = nullptr;
  uint32_t pre_stream_id = UINT32_MAX;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (i == 0) {
      (void)cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (!IsHcom(cur_cnode_ptr)) {
      (void)cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (cur_stream_id == pre_stream_id) {
      (void)cnodes.emplace_back(cur_cnode_ptr);
      pre_stream_id = cur_stream_id;
      continue;
    }

    if (!IsHcom(cnode_ptr_list[i - 1])) {
      uint32_t cur_event_id = resource_manager.ApplyNewEvent();
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, pre_stream_id);
      (void)cnodes.emplace_back(send);
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_stream_id);
      (void)cnodes.emplace_back(recv);
      (void)cnodes.emplace_back(cur_cnode_ptr);
    } else {
      (void)cnodes.emplace_back(cur_cnode_ptr);
    }
    pre_stream_id = cur_stream_id;
  }

  graph_ptr->set_execution_order(cnodes);
  MS_LOG(INFO) << "After hcom depend common, total event nums:" << resource_manager.cur_event_num();
}

std::vector<std::pair<uint32_t, vector<size_t>>> AscendStreamAssign::GetStreamIDHcomMap(
  std::vector<CNodePtr> cnode_ptr_list, const std::string group, size_t graph_id) const {
  std::vector<std::pair<uint32_t, vector<size_t>>> stream_indices;
  for (size_t i = 0; i < cnode_ptr_list.size(); i++) {
    auto cur_cnode = cnode_ptr_list[i];
    if (!IsHcom(cur_cnode)) {
      continue;
    }

    uint32_t cur_stream_id = AnfAlgo::GetStreamId(cur_cnode);
    auto group_name = GetHcomGroup(cur_cnode);
    auto cur_graph_id = AnfAlgo::GetGraphId(cur_cnode.get());
    MS_LOG(INFO) << "Hcom node name:" << common::AnfAlgo::GetCNodeName(cur_cnode) << "; group:" << group_name
                 << "; stream id:" << cur_stream_id;
    if (group_name != group || cur_graph_id != graph_id) {
      continue;
    }

    bool exit = false;
    for (auto &item : stream_indices) {
      if (item.first != cur_stream_id) {
        continue;
      }

      (void)item.second.emplace_back(i);
      exit = true;
      break;
    }
    if (!exit) {
      (void)stream_indices.emplace_back(std::make_pair(cur_stream_id, std::vector<size_t>{i}));
    }
  }
  return stream_indices;
}

void AscendStreamAssign::InsertEventHcomDependHcomAtSameGroup(
  const NotNull<KernelGraphPtr> &graph_ptr,
  std::pair<std::string, std::map<uint32_t, std::set<uint32_t>>> group_item) const {
  for (const auto &graph_item : std::as_const(group_item.second)) {
    auto stream_indices = GetStreamIDHcomMap(graph_ptr->execution_order(), group_item.first, graph_item.first);
    constexpr size_t kStreamMax = 2;
    if (stream_indices.size() < kStreamMax) {
      MS_LOG(INFO) << "Group:" << group_item.first << ", Graph: " << graph_item.first
                   << " different stream hcom size is less than 2, no need insert event between them";
      continue;
    }
    InsertEventBetweenHcom(graph_ptr, stream_indices);
  }
}

void AscendStreamAssign::InsertEventForCallCommSubGraph(const NotNull<KernelGraphPtr> &graph_ptr) const {
  if (comm_sub_graph_stream_.empty()) {
    MS_LOG(INFO) << "No comm sub graph, skip.";
    return;
  }

  std::map<uint32_t, std::string> comm_sub_graph_id_to_group = {};  // key: label id, value: hcom group
  const auto &cnode_list = graph_ptr->execution_order();
  for (const auto &n : cnode_list) {
    if (!IsHcom(n) || !IsCommSubGraph(graph_ptr, n)) {
      continue;
    }
    auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(n, kAttrGroup);
    FuncGraphPtr sub_graph = n->func_graph();
    MS_EXCEPTION_IF_NULL(sub_graph);
    KernelGraphPtr sub_kernel_graph = sub_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(sub_kernel_graph);
    MS_EXCEPTION_IF_NULL(sub_kernel_graph->get_start_label());
    uint32_t label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(sub_kernel_graph->get_start_label(), kAttrLabelIndex);
    (void)comm_sub_graph_id_to_group.emplace(label_id, group_name);
  }

  std::map<std::string, CNodePtr> nearest_comm_op_by_group;  // key: group, value: nearest comm op whose group is key
  std::vector<CNodePtr> new_order;
  for (const auto &n : std::as_const(cnode_list)) {
    MS_EXCEPTION_IF_NULL(n);
    if (IsHcom(n) && !IsCommSubGraph(graph_ptr, n)) {
      std::string group = common::AnfAlgo::GetNodeAttr<std::string>(n, kAttrGroup);
      auto iter = nearest_comm_op_by_group.find(group);
      if (iter == nearest_comm_op_by_group.end()) {
        (void)nearest_comm_op_by_group.emplace(group, n);
      } else {
        iter->second = n;
      }
    } else if (uint32_t label_id = GetGotoLabelId(n); label_id != kInvalidLabelId) {
      // insert event
      std::map<uint32_t, std::string>::const_iterator label_iter = comm_sub_graph_id_to_group.find(label_id);
      if (label_iter == comm_sub_graph_id_to_group.cend()) {
        MS_LOG(INFO) << "Cannot find comm group for sub comm graph label id " << label_id;
        new_order.push_back(n);
        continue;
      }
      std::string group = label_iter->second;
      std::map<std::string, CNodePtr>::const_iterator node_iter = nearest_comm_op_by_group.find(group);
      if (node_iter == nearest_comm_op_by_group.cend()) {
        new_order.push_back(n);
        continue;
      }
      CNodePtr nearest_comm_op = node_iter->second;
      MS_EXCEPTION_IF_NULL(nearest_comm_op);
      uint32_t cur_event_id = AscendStreamMng::GetInstance().ApplyNewEvent();
      uint32_t send_stream_id = AnfAlgo::GetStreamId(nearest_comm_op);
      uint32_t recv_Stream_id = AnfAlgo::GetStreamId(n);
      MS_LOG(INFO) << "Insert event " << cur_event_id << " from " << nearest_comm_op->fullname_with_scope() << " to "
                   << n->fullname_with_scope() << " label id " << label_id;
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, send_stream_id);
      (void)new_order.insert(++std::find(new_order.cbegin(), new_order.cend(), nearest_comm_op), send);
      (void)new_order.emplace_back(CreateRecvApplyKernel(graph_ptr, cur_event_id, recv_Stream_id));
    }
    new_order.push_back(n);
  }
  graph_ptr->set_execution_order(new_order);
}

void AscendStreamAssign::InsertEventHcomDependHcom(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (group_hcom_graph_map_.empty()) {
    return;
  }
  for (const auto &group_item : std::as_const(group_hcom_graph_map_)) {
    InsertEventHcomDependHcomAtSameGroup(graph_ptr, group_item);
  }
}

void AscendStreamAssign::InsertEventBetweenHcom(
  const NotNull<KernelGraphPtr> &graph_ptr, const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index) const {
  vector<CNodePtr> orders;
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  if (hcom_index.empty()) {
    MS_LOG(EXCEPTION) << "Hcom stream number is empty";
  }
  size_t first_stream_last_index = hcom_index[0].second.back();
  size_t last_stream_first_index = hcom_index.back().second.front();
  MS_LOG(INFO) << "First stream last index:" << first_stream_last_index
               << "; last stream first index:" << last_stream_first_index;
  (void)std::copy(cnode_ptr_list.begin(),
                  cnode_ptr_list.begin() + static_cast<std::vector<CNodePtr>::difference_type>(first_stream_last_index),
                  std::back_inserter(orders));
  for (size_t i = first_stream_last_index; i <= last_stream_first_index; i++) {
    auto cur_cnode = cnode_ptr_list[i];
    if (!IsSatisfiedHcom(hcom_index, cur_cnode, i)) {
      (void)orders.emplace_back(cur_cnode);
      continue;
    }
    auto cur_hcom_stream_id = AnfAlgo::GetStreamId(cur_cnode);
    if (i == first_stream_last_index) {
      // first fusion hcom
      (void)orders.emplace_back(cur_cnode);
      auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      (void)orders.emplace_back(send);
    } else if (i == last_stream_first_index) {
      // last fusion hcom
      auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
      (void)orders.emplace_back(recv);
      (void)orders.emplace_back(cur_cnode);
    } else {
      size_t cur_stream_hcom_size = UINT32_MAX;
      size_t first_index = UINT32_MAX;
      size_t last_index = UINT32_MAX;
      for (const auto &item : hcom_index) {
        if (item.first == cur_hcom_stream_id) {
          cur_stream_hcom_size = item.second.size();
          first_index = item.second.front();
          last_index = item.second.back();
        }
      }

      if (cur_stream_hcom_size == 1) {
        auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
        (void)orders.emplace_back(recv);
        cur_event_id = resource_manager.ApplyNewEvent();
        (void)orders.emplace_back(cur_cnode);
        auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
        (void)orders.emplace_back(send);
      } else {
        // current stream, first hcom:add recv op
        if (i == first_index) {
          auto recv = CreateRecvApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
          (void)orders.emplace_back(recv);
          cur_event_id = resource_manager.ApplyNewEvent();
          (void)orders.emplace_back(cur_cnode);
        } else if (i == last_index) {
          // current stream, last hcom:add send op
          (void)orders.emplace_back(cur_cnode);
          auto send = CreateSendApplyKernel(graph_ptr, cur_event_id, cur_hcom_stream_id);
          (void)orders.emplace_back(send);
        } else {
          // current stream, not first and last op
          (void)orders.emplace_back(cur_cnode);
        }
      }
    }
  }
  (void)std::copy(
    cnode_ptr_list.begin() + static_cast<std::vector<CNodePtr>::difference_type>(last_stream_first_index + 1),
    cnode_ptr_list.end(), std::back_inserter(orders));
  graph_ptr->set_execution_order(orders);
}

bool AscendStreamAssign::IsSatisfiedHcom(const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index,
                                         const CNodePtr &node_ptr, size_t index) const {
  MS_EXCEPTION_IF_NULL(node_ptr);
  auto cur_hcom_stream_id = AnfAlgo::GetStreamId(node_ptr);
  for (const auto &item : hcom_index) {
    if (item.first == cur_hcom_stream_id) {
      auto it = std::find(item.second.begin(), item.second.end(), index);
      if (it != item.second.end()) {
        return true;
      }
    }
  }
  return false;
}

// section6
void AscendStreamAssign::InsertEventForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto cnode_ptr_list = graph_ptr->execution_order();
  vector<CNodePtr> cnodes = cnode_ptr_list;
  uint32_t cur_event_id = resource_manager.ApplyNewEvent();
  std::map<CNodePtr, CNodePtr> cnode_send_map;
  std::map<CNodePtr, std::vector<CNodePtr>> cnode_recv_map;
  auto it = cnodes.begin();
  while (it != cnodes.end()) {
    MS_EXCEPTION_IF_NULL(*it);
    if (AnfAlgo::IsIndependentNode(*it)) {
      MS_LOG(DEBUG) << "Deal independent op[" << (*it)->DebugString() << "]";
      CNodePtr send_cnode_ptr = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(*it));

      auto target = FindFirstUserInExecutionOrder(it + 1, cnodes.end(), *it, false);
      if (target == cnodes.end()) {
        MS_LOG(DEBUG) << "Independent node[" << (*it)->fullname_with_scope()
                      << "] can't find target for insert recv op, no insert send/recv";
        ++it;
        continue;
      }

      // deal recv op
      uint32_t stream_id = AnfAlgo::GetStreamId(*target);
      CNodePtr recv_cnode_ptr = CreateRecvApplyKernel(graph_ptr, cur_event_id, stream_id);

      (void)cnode_send_map.insert(std::make_pair(*it, send_cnode_ptr));
      auto result = cnode_recv_map.find(*target);
      if (result == cnode_recv_map.cend()) {
        std::vector<CNodePtr> recv_cnodes = {recv_cnode_ptr};
        (void)cnode_recv_map.insert(std::make_pair(*target, recv_cnodes));
      } else {
        result->second.push_back(recv_cnode_ptr);
      }
      cur_event_id = resource_manager.ApplyNewEvent();
    }
    ++it;
  }
  // one event allocated additional, should delete
  resource_manager.DeleteEvent();

  std::vector<CNodePtr> new_cnodes;
  for (const auto &cnode : cnodes) {
    std::map<CNodePtr, std::vector<CNodePtr>>::const_iterator result_recv = cnode_recv_map.find(cnode);
    if (result_recv != cnode_recv_map.cend()) {
      const std::vector<CNodePtr> &result_recv_vec = result_recv->second;
      (void)new_cnodes.insert(new_cnodes.end(), result_recv_vec.cbegin(), result_recv_vec.cend());
    }
    new_cnodes.push_back(cnode);
    std::map<CNodePtr, CNodePtr>::const_iterator result_send = cnode_send_map.find(cnode);
    if (result_send != cnode_send_map.cend()) {
      new_cnodes.push_back(result_send->second);
    }
  }

  graph_ptr->set_execution_order(new_cnodes);
  MS_LOG(INFO) << "After independent parallel, total event nums:" << resource_manager.cur_event_num();
}

void AscendStreamAssign::GetIndependentMaxTarget(const NotNull<KernelGraphPtr> &graph_ptr) {
  MS_LOG(INFO) << "Start";
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); i++) {
    auto cur_node = cnode_ptr_list[i];
    auto key = cur_node.get();
    if (!AnfAlgo::IsIndependentNode(cur_node)) {
      continue;
    }

    bool flag = false;
    for (size_t j = cnode_ptr_list.size() - 1; j > i; j--) {
      auto target_node = cnode_ptr_list[j];
      auto inputs = target_node->inputs();
      for (size_t m = 1; m < inputs.size(); m++) {
        auto input = inputs[m];
        MS_EXCEPTION_IF_NULL(input);
        if (common::AnfAlgo::IsNopNode(input)) {
          auto cnode = input->cast<CNodePtr>();
          auto new_inputs = cnode->inputs();
          for (size_t k = 1; k < new_inputs.size(); k++) {
            auto new_real_input = common::AnfAlgo::VisitKernel(new_inputs[k], 0);
            if (key == new_real_input.first.get()) {
              MS_LOG(DEBUG) << "Nop node find max target op:" << common::AnfAlgo::GetCNodeName(cur_node);
              (void)independent_targets_.emplace(target_node.get());
              flag = true;
              break;
            }
          }
        } else {
          auto real_input = common::AnfAlgo::VisitKernel(input, 0);
          if (key == real_input.first.get()) {
            MS_LOG(DEBUG) << "Find max target op:" << common::AnfAlgo::GetCNodeName(cur_node);
            (void)independent_targets_.emplace(target_node.get());
            flag = true;
          }
        }
        if (flag) {
          break;
        }
      }
    }
  }

  MS_LOG(INFO) << "End";
}

uint32_t AscendStreamAssign::GetIndexByKey(const NotNull<KernelGraphPtr> &graph_ptr, const CNodeKey &key) const {
  auto &exe_orders = graph_ptr->execution_order();
  auto result =
    std::find_if(exe_orders.begin(), exe_orders.end(), [key](const CNodePtr &cnode) { return cnode.get() == key; });
  return result == exe_orders.end() ? UINT32_MAX : (result - exe_orders.begin());
}

uint32_t AscendStreamAssign::GetMaxIndexTarget(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (independent_targets_.empty()) {
    return UINT32_MAX;
  }

  std::set<uint32_t> indices;
  for (const auto &key : std::as_const(independent_targets_)) {
    auto index = GetIndexByKey(graph_ptr, key);
    if (index == UINT32_MAX) {
      MS_LOG(EXCEPTION) << "graph has no correspond key";
    }
    (void)indices.emplace(index);
  }

  return *(std::max_element(indices.begin(), indices.end()));
}

uint32_t AscendStreamAssign::GetIndependentStreamSwitchStreamId(const NotNull<KernelGraphPtr> &graph_ptr) const {
  auto &exe_orders = graph_ptr->execution_order();
  for (const auto &item : exe_orders) {
    if (common::AnfAlgo::GetCNodeName(item) == kStreamSwitchOpName) {
      if (!common::AnfAlgo::HasNodeAttr(kAttrStreamSwitchKind, item)) {
        continue;
      }
      auto kind = common::AnfAlgo::GetNodeAttr<uint32_t>(item, kAttrStreamSwitchKind);
      if (kind == static_cast<uint32_t>(kIndependentStreamSwitch)) {
        return AnfAlgo::GetStreamId(item);
      }
    }
  }
  return kInvalidStreamId;
}

void AscendStreamAssign::InsertCtrlForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr) {
  if (independent_targets_.empty()) {
    return;
  }

  uint32_t independent_switch_stream = GetIndependentStreamSwitchStreamId(graph_ptr);
  if (independent_switch_stream == kInvalidStreamId) {
    return;
  }

  auto max_index = GetMaxIndexTarget(graph_ptr);
  auto &exe_orders = graph_ptr->execution_order();
  if (max_index >= exe_orders.size()) {
    MS_LOG(EXCEPTION) << "Max target index:" << max_index << " is greater than graph orders size:" << exe_orders.size();
  }

  auto max_node_stream = AnfAlgo::GetStreamId(exe_orders[max_index]);

  CNodePtr active_ptr = KernelAdjust::GetInstance().CreateStreamActiveOp(graph_ptr);
  // 1.set stream id
  AnfAlgo::SetStreamId(max_node_stream, active_ptr.get());
  // 2.set active stream ids
  std::vector<uint32_t> active_index_list{independent_switch_stream};
  common::AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_list), active_ptr);

  std::vector<CNodePtr> update_cnode_list;
  (void)std::copy(exe_orders.begin(), exe_orders.begin() + max_index + 1, std::back_inserter(update_cnode_list));
  (void)update_cnode_list.emplace_back(active_ptr);
  (void)std::copy(exe_orders.begin() + max_index + 1, exe_orders.end(), std::back_inserter(update_cnode_list));
  graph_ptr->set_execution_order(update_cnode_list);
}

// section7
void AscendStreamAssign::GetNeedActiveStreams(const NotNull<KernelGraphPtr> &graph_ptr) {
  CNodePtr cur_cnode_ptr = nullptr;
  auto cnode_ptr_list = graph_ptr->execution_order();

  // 1)stream witch kStreamNeedActivedFirst attr should be activated;
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    if (!common::AnfAlgo::HasNodeAttr(kStreamNeedActivedFirst, cur_cnode_ptr)) {
      continue;
    }

    auto need_active = common::AnfAlgo::GetNodeAttr<bool>(cur_cnode_ptr, kStreamNeedActivedFirst);
    if (need_active) {
      auto stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
      MS_LOG(INFO) << "Stream id:" << stream_id << " is need activated at first";
      need_first_active_streams_.push_back(stream_id);
    }
  }

  // 2)independent stream:if has not been activate, push to need active vector
  auto root_graph_id = graph_ptr->graph_id();
  if (!independent_stream_activated_) {
    std::map<uint32_t, std::set<uint32_t>>::const_iterator it = independent_graph_map_.find(root_graph_id);
    if (it != independent_graph_map_.cend()) {
      need_first_active_streams_.push_back(*(it->second.begin()));
    }
  }

  // 3)hcom stream:if has not been activate, push to need active vector
  if (!hcom_stream_activated_) {
    for (const auto &item : std::as_const(group_hcom_graph_map_)) {
      auto &hcom_graph_map = item.second;
      auto it = hcom_graph_map.find(root_graph_id);
      if (it != hcom_graph_map.end()) {
        (void)std::copy(it->second.begin(), it->second.end(), std::back_inserter(need_first_active_streams_));
      }
    }
  }

  // 4)first stream 0 should be activated first;
  auto it = std::find(need_first_active_streams_.begin(), need_first_active_streams_.end(), 0);
  if (it == need_first_active_streams_.end()) {
    (void)need_first_active_streams_.emplace_back(0);
  }
  MS_LOG(INFO) << "Finally, need active first stream include:";
  for (const auto &item : need_first_active_streams_) {
    MS_LOG(INFO) << "stream id:" << item;
  }
}

// section8
void AscendStreamAssign::CheckResourceAssign(const NotNull<KernelGraphPtr> &graph_ptr) const {
  CheckStreamAssign(graph_ptr);
  CheckEventAssign(graph_ptr);
}

void AscendStreamAssign::CheckStreamAssign(const NotNull<KernelGraphPtr> &graph_ptr) const {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  std::set<uint32_t> streams;
  uint32_t max_stream = 0;
  uint32_t min_stream = kInvalidStreamId;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    uint32_t stream_id = AnfAlgo::GetStreamId(cur_cnode_ptr);
    if (stream_id == kInvalidStreamId) {
      MS_LOG(EXCEPTION) << "Node:" << common::AnfAlgo::GetCNodeName(cur_cnode_ptr) << "had not been assigned stream";
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
      MS_LOG(EXCEPTION) << "Stream should start from 0, now is from " << min_stream
                        << ", graph id: " << graph_ptr->graph_id();
    }
    uint32_t assigned_stream_num = resource_manager.cur_stream_num();
    if ((max_stream != assigned_stream_num - 1) || (streams.size() != assigned_stream_num)) {
      MS_LOG(EXCEPTION) << "Stream should be consecutive, max stream id:" << max_stream
                        << "; alloc stream nums:" << assigned_stream_num << "; streams size:" << streams.size();
    }
  }
}

void AscendStreamAssign::CheckEventAssign(const NotNull<KernelGraphPtr> &graph_ptr) const {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  std::map<uint32_t, std::vector<CNodePtr>> event_map;
  uint32_t max_event_id = 0;
  uint32_t min_event_id = kInvalidEventId;
  auto cnode_ptr_list = graph_ptr->execution_order();
  for (size_t i = 0; i < cnode_ptr_list.size(); ++i) {
    CNodePtr cur_cnode_ptr = cnode_ptr_list[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    auto name = common::AnfAlgo::GetCNodeName(cur_cnode_ptr);
    if (name == kSendOpName || name == kRecvOpName) {
      uint32_t event_id = common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrEventId);
      if (event_id > max_event_id) {
        max_event_id = event_id;
      }

      if (event_id < min_event_id) {
        min_event_id = event_id;
      }
      std::map<uint32_t, std::vector<CNodePtr>>::const_iterator it = event_map.find(event_id);
      if (it == event_map.cend()) {
        event_map[event_id] = {cur_cnode_ptr};
      } else {
        (void)event_map[event_id].emplace_back(cur_cnode_ptr);
      }
    }
  }
  // check event assign
  if (!event_map.empty()) {
    if (min_event_id != 0) {
      MS_LOG(EXCEPTION) << "Event should start from 0, now is from " << min_event_id;
    }
    uint32_t assigned_event_num = resource_manager.cur_event_num();
    if ((max_event_id != assigned_event_num - 1) || (event_map.size() != assigned_event_num)) {
      MS_LOG(EXCEPTION) << "Event should be consecutive, however, assigned event num is: " << assigned_event_num
                        << ", max event id:" << max_event_id << ", event map is:" << event_map;
    }
    for (const auto &item : std::as_const(event_map)) {
      constexpr auto pair_size = 2;
      if (item.second.size() != pair_size) {
        MS_LOG(EXCEPTION) << "Send/recv should be in pair and share one event id, invalid event id is:" << item.first
                          << ", event size is:" << item.second.size();
      }
      auto first_name = common::AnfAlgo::GetCNodeName(item.second[0]);
      auto second_name = common::AnfAlgo::GetCNodeName(item.second[1]);
      if (!(first_name == kSendOpName && second_name == kRecvOpName)) {
        MS_LOG(EXCEPTION) << "Send should be before recv, invalid event id is:" << item.first;
      }
    }
  }
}

// section9
CNodePtr AscendStreamAssign::CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                   uint32_t stream_id) const {
  auto send_node_ptr = KernelAdjust::GetInstance().CreateSendApplyKernel(graph_ptr, event_id);
  AnfAlgo::SetStreamId(stream_id, send_node_ptr.get());
  return send_node_ptr;
}

CNodePtr AscendStreamAssign::CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                   uint32_t stream_id) const {
  auto recv_node_ptr = KernelAdjust::GetInstance().CreateRecvApplyKernel(graph_ptr, event_id);
  AnfAlgo::SetStreamId(stream_id, recv_node_ptr.get());
  return recv_node_ptr;
}

bool AscendStreamAssign::IsNopNodeTarget(const AnfNodePtr &nop_node, const CNodePtr &target_node,
                                         const CNodePtr &cur_node, bool exclude_hcom) {
  MS_EXCEPTION_IF_NULL(nop_node);
  auto cnode = nop_node->cast<CNodePtr>();
  auto new_inputs = cnode->inputs();
  for (size_t i = 1; i < new_inputs.size(); i++) {
    if (common::AnfAlgo::IsNopNode(new_inputs[i])) {
      if (IsNopNodeTarget(new_inputs[i], target_node, cur_node, exclude_hcom)) {
        return true;
      }
    } else {
      auto new_real_input = common::AnfAlgo::VisitKernel(new_inputs[i], 0);
      if (target_node == new_real_input.first) {
        if (!(exclude_hcom && IsHcom(cur_node))) {
          return true;
        }
      } else if (common::AnfAlgo::IsNopNode(new_real_input.first)) {
        if (IsNopNodeTarget(new_real_input.first, target_node, cur_node, exclude_hcom)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Return the first user of node in graph execution order
// args node: the target node, find the node user
// args exclude_hcom: whether return hcom node, true: return node can be hcom node, false: return node can not be hcom.
// scenario-1: node -> target user
// scenario-2: node -> reshape -> target user
// scenario-3: node -> reshape ->reshape ->target user
// scenario-4: node -> depend -> target user
// scenario-5: node -> reshape -> depend ->target user
// scenario-6: execution order(node,hcom user,common user)
//             exclude_hcom(true): return hcom user
//             exclude_hcom(false): return common user
vector<CNodePtr>::iterator AscendStreamAssign::FindFirstUserInExecutionOrder(vector<CNodePtr>::iterator begin,
                                                                             vector<CNodePtr>::iterator end,
                                                                             const CNodePtr &node, bool exclude_hcom) {
  while (begin != end) {
    auto inputs = (*begin)->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      auto input = inputs[i];
      MS_EXCEPTION_IF_NULL(input);
      if (common::AnfAlgo::IsNopNode(input)) {
        if (IsNopNodeTarget(input, node, *begin, exclude_hcom)) {
          return begin;
        }
      } else {
        auto real_input = common::AnfAlgo::VisitKernel(input, 0);
        if (node == real_input.first) {
          if (!(exclude_hcom && IsHcom(*begin))) {
            MS_LOG(DEBUG) << "Nop node find target op[" << (*begin)->DebugString() << "]";
            return begin;
          }
        } else if (common::AnfAlgo::IsNopNode(real_input.first)) {
          if (IsNopNodeTarget(real_input.first, node, *begin, exclude_hcom)) {
            return begin;
          }
        }
      }
    }
    ++begin;
  }
  return end;
}

bool AscendStreamAssign::IsTaskSink() const { return KernelAdjust::GetInstance().IsTaskSink(); }

void AscendStreamAssign::GetWaitStreams(vector<uint32_t> *wait_active_stream_list) {
  MS_EXCEPTION_IF_NULL(wait_active_stream_list);
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  uint32_t total_stream_num = resource_manager.cur_stream_num();
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

void AscendStreamAssign::GetHcomStreams(std::vector<uint32_t> *streams) {
  MS_EXCEPTION_IF_NULL(streams);
  (void)std::transform(hcom_stream_.begin(), hcom_stream_.end(), std::back_inserter(*streams),
                       [](const uint32_t &item) { return item; });
  (void)std::transform(comm_sub_graph_stream_.begin(), comm_sub_graph_stream_.end(), std::back_inserter(*streams),
                       [](const uint32_t &item) { return item; });
}

bool AscendStreamAssign::IsHcom(const CNodePtr &cur_cnode_ptr) const {
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  auto node_name = common::AnfAlgo::GetCNodeName(cur_cnode_ptr);
  static const auto send_recv_parallel = (common::GetEnv("SEND_RECV_PARALLEL") == "1");
  if (cur_cnode_ptr->HasAttr(parallel::FIRST_RECEIVE)) {
    return true;
  }
  if ((node_name == kHcomSendOpName || node_name == kReceiveOpName) && !send_recv_parallel) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
  return AnfAlgo::GetKernelType(cur_cnode_ptr) == HCCL_KERNEL;
}

void AscendStreamAssign::Reset() {
  independent_stream_activated_ = false;
  hcom_stream_activated_ = false;
  loop_sink_ = false;
  independent_stream_.clear();
  hcom_stream_.clear();
  comm_sub_graph_stream_.clear();
  processed_streams_.clear();
  need_first_active_streams_.clear();
  stream_groups_.clear();
  stream_relations_.clear();
  event_map_.clear();
  independent_targets_.clear();
  independent_graph_map_.clear();
  group_hcom_graph_map_.clear();
  middle_active_streams_.clear();
}

// section 10
bool AscendStreamAssign::IsVecExist(const std::vector<uint32_t> &group) {
  auto group_size = group.size();
  if (group_size == 0) {
    return false;
  }
  for (const auto &item : stream_groups_) {
    if (item.size() < group.size()) {
      continue;
    }

    bool flag = true;
    for (size_t i = 0; i < group_size; i++) {
      if (item[i] != group.at(i)) {
        flag = false;
        break;
      }
    }

    if (flag) {
      return true;
    } else {
      continue;
    }
  }

  return false;
}

void AscendStreamAssign::DFS(uint32_t start, std::vector<uint32_t> *group) {
  MS_EXCEPTION_IF_NULL(group);
  std::map<uint32_t, std::vector<uint32_t>>::const_iterator it = stream_relations_.find(start);
  if (it == stream_relations_.cend()) {
    if (!IsVecExist(*group)) {
      (void)stream_groups_.emplace_back(*group);
    } else {
      MS_LOG(WARNING) << "DFS find same stream group, Not expected";
    }
    return;
  }

  vector<uint32_t> active_streams = stream_relations_[start];

  for (const auto &item : active_streams) {
    (void)group->emplace_back(item);
    DFS(item, group);
    group->pop_back();
  }
}

void AscendStreamAssign::GetStreamRelations() {
  auto starts = middle_active_streams_;
  for (const auto &stream : need_first_active_streams_) {
    (void)starts.emplace(stream);
  }

  for (const auto &start : std::as_const(starts)) {
    vector<uint32_t> group{start};
    DFS(start, &group);
  }
}

void AscendStreamAssign::FindStreamRelations(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto stream_num = resource_manager.cur_stream_num();
  if (stream_num <= 1) {
    return;
  }

  auto exe_orders = graph_ptr->execution_order();
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode = exe_orders[i];
    auto name = common::AnfAlgo::GetCNodeName(cur_cnode);
    if (name != kStreamSwitchOpName && name != kStreamActiveOpName) {
      continue;
    }

    // support:streamswitch is begin of the stream
    if (name == kStreamSwitchOpName) {
      GetStreamSwitchStreamRelation(cur_cnode);
    }

    if (name == kStreamActiveOpName) {
      GetStreamActiveStreamRelation(graph_ptr, i);
    }
  }
}

void AscendStreamAssign::GetStreamSwitchStreamRelation(const CNodePtr &node_ptr) {
  MS_EXCEPTION_IF_NULL(node_ptr);
  auto cur_stream_id = AnfAlgo::GetStreamId(node_ptr);
  auto true_stream_id = common::AnfAlgo::GetNodeAttr<uint32_t>(node_ptr, kAttrTrueBranchStream);
  if (true_stream_id <= cur_stream_id) {
    MS_LOG(ERROR) << "StreamSwitch self stream id " << cur_stream_id
                  << " is greater than true branch stream id:" << true_stream_id;
  }
  std::map<uint32_t, std::vector<uint32_t>>::const_iterator it = stream_relations_.find(cur_stream_id);
  if (it == stream_relations_.cend()) {
    stream_relations_[cur_stream_id] = {true_stream_id};
  } else {
    auto iter =
      std::find(stream_relations_[cur_stream_id].begin(), stream_relations_[cur_stream_id].end(), true_stream_id);
    if (iter == stream_relations_[cur_stream_id].end()) {
      (void)stream_relations_[cur_stream_id].emplace_back(true_stream_id);
    }
  }
}

void AscendStreamAssign::GetStreamActiveStreamRelation(const NotNull<KernelGraphPtr> &graph_ptr, size_t index) {
  StreamActiveKind kind = GetStreamActiveKind(graph_ptr, index);
  if (kind == kInvalid) {
    MS_LOG(INFO) << "Invalid streamActive kind";
    return;
  }

  auto orders = graph_ptr->execution_order();
  if (index >= orders.size()) {
    MS_LOG(EXCEPTION) << "Invalid index.";
  }
  auto cur_cnode = orders[index];
  auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode);

  auto active_list = common::AnfAlgo::GetNodeAttr<vector<uint32_t>>(cur_cnode, kAttrActiveStreamList);
  if (kind == kHead) {
    uint32_t active_current_stream_id = GetStreamByActivedStream(cur_stream_id);
    if (active_current_stream_id == kInvalidStreamId) {
      MS_LOG(EXCEPTION) << "No stream to active streamactive stream: " << cur_stream_id;
    }

    for (const auto &item : active_list) {
      if (item <= active_current_stream_id) {
        MS_LOG(WARNING) << "Activated stream is less than activing stream";
        continue;
      }
      auto it = std::find(stream_relations_[active_current_stream_id].begin(),
                          stream_relations_[active_current_stream_id].end(), item);
      if (it == stream_relations_[active_current_stream_id].end()) {
        (void)stream_relations_[active_current_stream_id].emplace_back(item);
      }
    }
  }

  if (kind == kMiddle) {
    for (const auto &stream : active_list) {
      if (stream <= cur_stream_id) {
        MS_LOG(INFO) << "MIDDLE StreamActive active stream is less than self stream, no need deal";
      } else {
        MS_LOG(INFO) << "MIDDLE StreamActive :" << cur_stream_id << ", active target stream:" << stream;
        (void)middle_active_streams_.emplace(stream);
      }
    }
  }

  if (kind == kTail) {
    std::map<uint32_t, std::vector<uint32_t>>::const_iterator it = stream_relations_.find(cur_stream_id);
    if (it == stream_relations_.cend()) {
      stream_relations_[cur_stream_id] = active_list;
    } else {
      for (const auto &stream : active_list) {
        if (stream <= cur_stream_id) {
          MS_LOG(WARNING) << "Activated stream is less than activing stream";
          continue;
        }
        auto iter = std::find(stream_relations_[cur_stream_id].begin(), stream_relations_[cur_stream_id].end(), stream);
        if (iter == stream_relations_[cur_stream_id].end()) {
          (void)stream_relations_[cur_stream_id].emplace_back(stream);
        }
      }
    }
  }
}

StreamActiveKind AscendStreamAssign::GetStreamActiveKind(const NotNull<KernelGraphPtr> &graph_ptr, size_t index) {
  auto exe_orders = graph_ptr->execution_order();
  if (index >= exe_orders.size()) {
    MS_LOG(EXCEPTION) << "Invalid op index:" << index;
  }

  auto cur_cnode = exe_orders[index];
  auto cur_stream_id = AnfAlgo::GetStreamId(cur_cnode);
  if (common::AnfAlgo::GetCNodeName(cur_cnode) != kStreamActiveOpName) {
    MS_LOG(EXCEPTION) << "Current node name [" << common::AnfAlgo::GetCNodeName(cur_cnode) << "] is not StreamActive.";
  }

  if (index == 0) {
    return kInvalid;
  }

  if (index == exe_orders.size() - 1) {
    return kInvalid;
  }

  uint32_t pre_stream_id = UINT32_MAX;
  uint32_t next_stream_id = UINT32_MAX;
  int32_t start = SizeToInt(index) - 1;
  for (int32_t i = start; i >= 0; i--) {
    auto cnode = exe_orders[IntToSize(i)];
    auto name = common::AnfAlgo::GetCNodeName(cnode);
    if (name == kSendOpName || name == kRecvOpName) {
      continue;
    }
    auto stream = AnfAlgo::GetStreamId(cnode);
    std::set<uint32_t>::const_iterator it = hcom_stream_.find(stream);
    if (it != hcom_stream_.cend()) {
      continue;
    }

    it = independent_stream_.find(stream);
    if (it != independent_stream_.cend()) {
      continue;
    }

    pre_stream_id = stream;
    break;
  }

  for (size_t i = index + 1; i < exe_orders.size(); i++) {
    auto cnode = exe_orders[i];
    if (common::AnfAlgo::GetCNodeName(cnode) == kSendOpName || common::AnfAlgo::GetCNodeName(cnode) == kRecvOpName ||
        common::AnfAlgo::GetCNodeName(cnode) == kEndGraph) {
      continue;
    }

    auto stream = AnfAlgo::GetStreamId(cnode);
    std::set<uint32_t>::const_iterator it = hcom_stream_.find(stream);
    if (it != hcom_stream_.cend()) {
      continue;
    }

    it = independent_stream_.find(stream);
    if (it != independent_stream_.cend()) {
      continue;
    }

    next_stream_id = stream;
    break;
  }

  return GetStreamKind(cur_stream_id, pre_stream_id, next_stream_id);
}

uint32_t AscendStreamAssign::GetStreamByActivedStream(uint32_t actived_stream_id) {
  if (stream_relations_.empty()) {
    return kInvalidStreamId;
  }

  for (const auto &item : std::as_const(stream_relations_)) {
    auto it = std::find(item.second.begin(), item.second.end(), actived_stream_id);
    if (it != item.second.end()) {
      return item.first;
    }
  }

  return kInvalidStreamId;
}

void AscendStreamAssign::PrintStreamRelations() {
  MS_LOG(INFO) << "Stream relations size:" << stream_relations_.size();
  for (const auto &item : std::as_const(stream_relations_)) {
    MS_LOG(INFO) << "Stream:" << item.first;
    for (const auto &stream : item.second) {
      MS_LOG(INFO) << "Activated stream id:" << stream;
    }
  }
}

void AscendStreamAssign::PrintStreamGroups() {
  MS_LOG(INFO) << "Stream group size:" << stream_groups_.size();
  for (const auto &item : stream_groups_) {
    MS_LOG(INFO) << "Group:";
    for (const auto &stream : item) {
      MS_LOG(INFO) << "Stream id:" << stream;
    }
  }
}

// section 11
bool AscendStreamAssign::IsSatisfiedEvent(uint32_t send_stream_id, uint32_t recv_stream_id) const {
  size_t send_group = 0;
  size_t recv_group = 0;
  bool send_flag = true;
  bool recv_flag = true;
  for (size_t i = 0; i < stream_groups_.size(); i++) {
    auto group = stream_groups_[i];
    if (send_flag) {
      auto it = std::find(group.begin(), group.end(), send_stream_id);
      if (it != group.end()) {
        send_group = i;
        send_flag = false;
      }
    }

    if (recv_flag) {
      auto it = std::find(group.begin(), group.end(), recv_stream_id);
      if (it != group.end()) {
        recv_group = i;
        recv_flag = false;
      }
    }
  }

  if (!(send_flag || recv_flag)) {
    return (send_group != recv_group);
  }

  return false;
}

void AscendStreamAssign::FindEventRelations(const NotNull<KernelGraphPtr> &graph_ptr) {
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  auto event_nums = resource_manager.cur_event_num();
  if (event_nums == 0) {
    return;
  }
  auto exe_orders = graph_ptr->execution_order();
  // find all event info
  for (size_t i = 0; i < exe_orders.size(); i++) {
    auto cur_cnode = exe_orders[i];
    auto name = common::AnfAlgo::GetCNodeName(cur_cnode);
    if (name == kSendOpName) {
      event_map_[cur_cnode] = {};
    }

    if (name == kRecvOpName) {
      auto recv_event_id = common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode, kAttrEventId);
      for (auto &item : event_map_) {
        auto send_event_id = common::AnfAlgo::GetNodeAttr<uint32_t>(item.first, kAttrEventId);
        if (recv_event_id == send_event_id) {
          item.second = cur_cnode;
          break;
        }
      }
    }
  }

  // delete useless event info
  auto begin = event_map_.begin();
  while (begin != event_map_.end()) {
    auto send_stream_id = AnfAlgo::GetStreamId(begin->first);
    auto recv_stream_id = AnfAlgo::GetStreamId(begin->second);
    bool flag = IsSatisfiedEvent(send_stream_id, recv_stream_id);
    if (!flag) {
      begin = event_map_.erase(begin);
    } else {
      ++begin;
    }
  }

  MS_LOG(INFO) << "Satisfied event info";
  for (const auto &item : event_map_) {
    MS_LOG(INFO) << "Event_id:" << common::AnfAlgo::GetNodeAttr<uint32_t>(item.first, kAttrEventId);
  }
}

// section12
void AscendStreamAssign::AdjustAtomicAddrCleanOrder(const NotNull<KernelGraphPtr> &graph_ptr) const {
  // Eg:[atomic, recv, memcpy] should be [recv, atomic, memcpy]
  std::vector<CNodePtr> update_orders;
  auto &exe_orders = graph_ptr->execution_order();
  size_t i = 0;
  while (i < exe_orders.size()) {
    auto cur_cnode = exe_orders.at(i);
    if (common::AnfAlgo::GetCNodeName(cur_cnode) != kAtomicAddrCleanOpName) {
      (void)update_orders.emplace_back(cur_cnode);
      i++;
      continue;
    }
    while (i < exe_orders.size() - 1) {
      i++;
      auto next_cnode = exe_orders.at(i);
      auto next_cnode_name = common::AnfAlgo::GetCNodeName(next_cnode);
      if (next_cnode_name == kSendOpName || next_cnode_name == kRecvOpName) {
        (void)update_orders.emplace_back(next_cnode);
      } else {
        (void)update_orders.emplace_back(cur_cnode);
        break;
      }
    }
  }
  graph_ptr->set_execution_order(update_orders);
}

CNodePtr FindNextGenMask(const NotNull<KernelGraphPtr> &graph_ptr, const CNodePtr do_mask_cnode) {
  auto &exec_order = graph_ptr->execution_order();
  auto iter = std::find(exec_order.begin(), exec_order.end(), do_mask_cnode);
  for (; iter != exec_order.end(); ++iter) {
    auto cnode = *iter;
    if (kDropoutGenMaskOps.find(common::AnfAlgo::GetCNodeName(cnode)) == kDropoutGenMaskOps.end() ||
        !cnode->HasPrimalAttr(kAttrMicro)) {
      continue;
    }
    return cnode;
  }
  return nullptr;
}

void AscendStreamAssign::InsertEventForMicroBatchIndependent(const NotNull<KernelGraphPtr> &graph_ptr) const {
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() <= 1) {
    return;
  }
  std::map<CNodePtr, CNodePtr> node_send_map;
  std::map<CNodePtr, CNodePtr> node_recv_map;
  std::map<size_t, CNodePtr> micro_last_cnode_map;
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();

  auto &exec_order = graph_ptr->execution_order();
  for (auto &cnode : exec_order) {
    if (common::AnfAlgo::GetCNodeName(cnode) != kDropOutDoMaskOpName &&
        common::AnfAlgo::GetCNodeName(cnode) != kDropOutDoMaskV3OpName &&
        common::AnfAlgo::GetCNodeName(cnode) != kDropOutDoMaskV3DOpName) {
      continue;
    }
    if (!cnode->HasPrimalAttr(kAttrMicro)) {
      MS_LOG(WARNING) << "Node doesn't have the attr [micro], node: " << cnode->fullname_with_scope();
      continue;
    }
    auto micro_ptr = cnode->GetPrimalAttr(kAttrMicro);
    auto micro_value = GetValue<int64_t>(micro_ptr);
    micro_last_cnode_map[micro_value] = cnode;
  }

  for (auto &micro_cnode_item : micro_last_cnode_map) {
    auto cnode = micro_cnode_item.second;
    MS_EXCEPTION_IF_NULL(cnode);
    auto micro_batch = micro_cnode_item.first;
    MS_LOG(INFO) << "Micro: " << micro_batch << ", last DropoutDoMask: " << cnode->fullname_with_scope();
    auto next_gen_mask = FindNextGenMask(graph_ptr, cnode);
    if (next_gen_mask == nullptr) {
      MS_LOG(INFO) << "Node doesn't have the next DropoutGenMask, node: " << cnode->fullname_with_scope()
                   << ", micro value: " << micro_batch;
      continue;
    }
    MS_LOG(INFO) << "Insert send after node: " << cnode->fullname_with_scope()
                 << ", insert recv before node: " << next_gen_mask->fullname_with_scope();
    uint32_t cur_event_id = resource_manager.ApplyNewEvent();
    CNodePtr send_cnode = CreateSendApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId((cnode)));
    CNodePtr recv_cnode = CreateRecvApplyKernel(graph_ptr, cur_event_id, AnfAlgo::GetStreamId(next_gen_mask));
    graph_ptr->InsertSendRecvPairForParallelOpInputs(next_gen_mask, std::make_pair(send_cnode, recv_cnode));
    node_send_map[cnode] = send_cnode;
    node_recv_map[next_gen_mask] = recv_cnode;
  }

  MS_LOG(INFO) << "Print execution order before inserting event between DropoutDoMask and DropoutGenMask.";
  graph_ptr->PrintGraphExecuteOrder();
  std::vector<CNodePtr> new_exec_order;
  for (auto &cnode : exec_order) {
    auto cnode_name = common::AnfAlgo::GetCNodeName(cnode);
    if (cnode_name == kDropOutDoMaskOpName || cnode_name == kDropOutDoMaskV3OpName ||
        cnode_name == kDropOutDoMaskV3DOpName) {
      auto send_iter = node_send_map.find(cnode);
      if (send_iter != node_send_map.end()) {
        new_exec_order.push_back(cnode);
        new_exec_order.push_back((*send_iter).second);
        continue;
      }
    }
    if (kDropoutGenMaskOps.find(cnode_name) != kDropoutGenMaskOps.end()) {
      auto recv_iter = node_recv_map.find(cnode);
      if (recv_iter != node_recv_map.end()) {
        new_exec_order.push_back((*recv_iter).second);
        new_exec_order.push_back(cnode);
        continue;
      }
    }
    new_exec_order.push_back(cnode);
  }
  graph_ptr->set_execution_order(new_exec_order);
  MS_LOG(INFO) << "Print execution order after inserting event between DropoutDoMask and DropoutGenMask.";
  graph_ptr->PrintGraphExecuteOrder();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
