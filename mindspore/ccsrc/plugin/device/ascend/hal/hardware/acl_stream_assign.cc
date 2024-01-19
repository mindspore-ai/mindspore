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
#include "plugin/device/ascend/hal/hardware/acl_stream_assign.h"
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/utils.h"
#include "ops/ascend_op_name.h"
#include "ops/framework_op_name.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
void AclStreamAssign::AssignStream(const NotNull<KernelGraphPtr> &kernel_graph) const {
  auto kernels = kernel_graph->execution_order();
  if (kernels.empty()) {
    return;
  }
  if (kernel_graph->is_from_single_op()) {
    MS_LOG(INFO) << "Not stream assign when pynative forward.";
    return;
  }
  for (const auto &node : kernels) {
    if (AnfAlgo::IsKernelSelectBackoffOp(node)) {
      continue;
    }
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      AnfAlgo::SetStreamId(kWorldGroupStreamIndex, node.get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kWorldGroupStreamIndex), node);
    } else {
      AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kDefaultStreamIndex), node);
    }
  }
  for (size_t i = 1; i < kernels.size(); ++i) {
    if (common::AnfAlgo::GetCNodeName(kernels[i - 1]) == kMemSetOpName) {
      auto stream_id = AnfAlgo::GetStreamId(kernels[i]);
      AnfAlgo::SetStreamId(stream_id, kernels[i - 1].get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(stream_id), kernels[i - 1]);
    }
  }
  InsertEventForNonTaskSink(kernel_graph);
}

void AclStreamAssign::GenKernelIoExecInfoMap(
  const NotNull<KernelGraphPtr> &kernel_graph,
  mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> *kernel_io_exec_info_map) const {
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

void AclStreamAssign::UpdateEventsToExecutionOrder(
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

void AclStreamAssign::InsertEventsForInputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                            const NodeIoExecInfoPtr &io_exec_info,
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

void AclStreamAssign::InsertEventsForOutputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                             const NodeIoExecInfoPtr &io_exec_info,
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

CNodePtr AclStreamAssign::CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                uint32_t stream_id) const {
  auto send_op = std::make_shared<Primitive>(kStreamSendOpName);
  MS_EXCEPTION_IF_NULL(send_op);
  auto send_apply = std::make_shared<ValueNode>(send_op);
  MS_EXCEPTION_IF_NULL(send_apply);
  auto send_node_ptr = graph_ptr->NewCNode({send_apply});
  MS_EXCEPTION_IF_NULL(send_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), send_node_ptr);
  AnfAlgo::SetStreamId(stream_id, send_node_ptr.get());
  return send_node_ptr;
}

CNodePtr AclStreamAssign::CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                uint32_t stream_id) const {
  auto recv_op = std::make_shared<Primitive>(kStreamRecvOpName);
  MS_EXCEPTION_IF_NULL(recv_op);
  auto recv_apply = std::make_shared<ValueNode>(recv_op);
  MS_EXCEPTION_IF_NULL(recv_apply);
  auto recv_node_ptr = graph_ptr->NewCNode({recv_apply});
  MS_EXCEPTION_IF_NULL(recv_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), recv_node_ptr);
  AnfAlgo::SetStreamId(stream_id, recv_node_ptr.get());
  return recv_node_ptr;
}

void AclStreamAssign::InsertEvents(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &parallel_cnode,
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

void AclStreamAssign::GenEventsForParallelOp(const NotNull<KernelGraphPtr> &kernel_graph,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
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

void AclStreamAssign::InsertEventForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph) const {
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_send;
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_recv;
  AnfAlgo::SetStreamId(kDefaultStreamIndex, kernel_graph->output().get());
  GenEventsForParallelOp(kernel_graph, &kernel_send, &kernel_recv);
  UpdateEventsToExecutionOrder(kernel_graph, kernel_send, kernel_recv);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
