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
#include <tuple>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/utils.h"
#include "ops/ascend_op_name.h"
#include "ops/framework_op_name.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
void SetForSwitchInline(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &send_cnode,
                        const CNodePtr &recv_cnode, const AnfNodePtr &pre_node, const AnfNodePtr &next_node) {
  if (pre_node == nullptr || next_node == nullptr) {
    return;
  }
  std::string branch_name = "";
  const auto &node_before_iter = kernel_graph->inline_sub_graph_kernels().find(pre_node);
  if (node_before_iter != kernel_graph->inline_sub_graph_kernels().end()) {
    branch_name = node_before_iter->second;
  } else {
    const auto &node_after_iter = kernel_graph->inline_sub_graph_kernels().find(next_node);
    if (node_after_iter != kernel_graph->inline_sub_graph_kernels().end()) {
      branch_name = node_after_iter->second;
    }
  }
  if (branch_name == "") {
    return;
  }
  kernel_graph->AddInlineSubgraphKernel(send_cnode, branch_name);
  MS_LOG(DEBUG) << "Add inline subgraph send kernel:" << send_cnode->fullname_with_scope()
                << " by before send node:" << pre_node->fullname_with_scope() << " branch name:" << branch_name
                << " for kernel graph:" << kernel_graph->ToString();

  kernel_graph->AddInlineSubgraphKernel(recv_cnode, branch_name);
  MS_LOG(DEBUG) << "Add inline subgraph recv kernel:" << recv_cnode->fullname_with_scope()
                << " by after receive node:" << next_node->fullname_with_scope() << " branch name:" << branch_name
                << " for kernel graph:" << kernel_graph->ToString();
}

void AddStreamIdForCommunicationOp(const AnfNodePtr &node, bool is_pp_interleave) {
  MS_EXCEPTION_IF_NULL(node);
  if (!is_pp_interleave) {
    AnfAlgo::SetStreamId(kWorldGroupStreamIndex, node.get());
    common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kWorldGroupStreamIndex), node);
  } else {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(node, std::make_shared<Primitive>(kSendOpName))) {
      if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
        AnfAlgo::SetStreamId(kBackwardSendStreamIndex, node.get());
        common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kBackwardSendStreamIndex), node);
      } else {
        AnfAlgo::SetStreamId(kForwardSendStreamIndex, node.get());
        common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kForwardSendStreamIndex), node);
      }
    } else if (IsPrimitiveCNode(node, std::make_shared<Primitive>(kReceiveOpName))) {
      if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
        AnfAlgo::SetStreamId(kBackwardReceiveStreamIndex, node.get());
        common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kBackwardReceiveStreamIndex), node);
      } else {
        AnfAlgo::SetStreamId(kForwardReceiveStreamIndex, node.get());
        common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kForwardReceiveStreamIndex), node);
      }
    } else {
      AnfAlgo::SetStreamId(kWorldGroupStreamIndex, node.get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kWorldGroupStreamIndex), node);
    }
  }
}
}  // namespace

void AclStreamAssign::AssignStream(const NotNull<KernelGraphPtr> &kernel_graph) {
  auto kernels = kernel_graph->execution_order();
  if (kernels.empty()) {
    return;
  }
  if (kernel_graph->is_from_single_op() || kernel_graph->has_flag(kFlagPyNativeRunInGraph)) {
    MS_LOG(INFO) << "Not stream assign when pynative forward.";
    return;
  }
  bool enable_multi_stream = false;
  for (const auto &node : kernels) {
    if (AnfAlgo::IsKernelSelectBackoffOp(node)) {
      continue;
    }
    auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(node);
    // for runtime speed up
    bool input_multi_graph_safe = true;
    for (size_t i = 0; i < input_tensor_num; i++) {
      auto input_node = node->input(i + 1);
      MS_EXCEPTION_IF_NULL(input_node);
      auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
      auto real_input = kernel_with_index.first;
      // input from other graph
      if (real_input->isa<Parameter>()) {
        input_multi_graph_safe = false;
      }
    }
    if (input_multi_graph_safe) {
      node->AddAttr(kAttrInputMultiStreamSafe, MakeValue(true));
    }
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    auto is_pp_interleave = parallel_context->pipeline_interleave();
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      AddStreamIdForCommunicationOp(node, is_pp_interleave);
      enable_multi_stream = true;
    } else {
      AnfAlgo::SetStreamId(kDefaultStreamIndex, node.get());
      common::AnfAlgo::SetNodeAttr(kAttrStreamId, MakeValue(kDefaultStreamIndex), node);
    }
  }
  kernel_graph->set_enable_multi_stream(enable_multi_stream);
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

void AclStreamAssign::AddBoundarySendRecvKernel(const NotNull<KernelGraphPtr> &kernel_graph, uint32_t record_stream_id,
                                                uint32_t wait_stream_id, std::vector<CNodePtr> *exec_order,
                                                std::map<size_t, std::set<size_t>> *no_event_streams,
                                                CNodePtr pre_cnode, CNodePtr next_cnode) {
  auto &resource_manager = AscendStreamMng::GetInstance();
  uint32_t event_id = resource_manager.ApplyNewEvent();
  auto event = resource_manager.ApplyRtEvent();
  auto event_generate_id = ++event_generate_id_;
  auto send_node = CreateSendApplyKernel(kernel_graph, event_id, record_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_node);
  auto recv_node = CreateRecvApplyKernel(kernel_graph, event_id, record_stream_id, wait_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_node);
  exec_order->push_back(send_node);
  exec_order->push_back(recv_node);
  (*no_event_streams)[wait_stream_id].erase(record_stream_id);
  SetForSwitchInline(kernel_graph, send_node, recv_node, pre_cnode, next_cnode);
}

void AclStreamAssign::ProcessSideEffect(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr kernel,
                                        size_t process_stream_id, const CNodePtr last_kernel,
                                        std::vector<AnfNodePtr> *real_inputs,
                                        std::map<AnfNodePtr, std::set<size_t>> *side_effect_map,
                                        std::map<size_t, std::set<size_t>> *no_event_streams,
                                        std::vector<CNodePtr> *new_exec_orders) {
  bool has_side_effect = false;
  auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto input_node = kernel->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    if (HasAbstractMonad(input_node)) {
      has_side_effect = true;
      continue;
    }
    auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
    auto real_input = kernel_with_index.first;
    real_inputs->push_back(real_input);
  }
  auto prim = GetValueNode<PrimitivePtr>(kernel->input(0));
  if (prim != nullptr &&
      (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM) || GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO))) {
    has_side_effect = true;
  }

  if (has_side_effect) {
    for (auto real_input : (*real_inputs)) {
      auto &stream_set = (*side_effect_map)[real_input];
      for (auto stream_id : stream_set) {
        if (stream_id == process_stream_id) {
          continue;
        }
        auto &no_event_stream_set = (*no_event_streams)[process_stream_id];
        auto no_event_iter = no_event_stream_set.find(stream_id);
        if (no_event_iter == no_event_stream_set.end()) {
          continue;
        }
        MS_LOG(INFO) << "Add side effect event " << stream_id << " to " << process_stream_id << " for kernel "
                     << kernel->fullname_with_scope();
        AddBoundarySendRecvKernel(kernel_graph, stream_id, process_stream_id, new_exec_orders, no_event_streams,
                                  last_kernel, kernel);
      }
    }
  }
}

void AclStreamAssign::UpdateEventsToExecutionOrder(
  const NotNull<KernelGraphPtr> &kernel_graph,
  const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &send_after_node,
  const mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> &recv_before_node,
  const mindspore::HashMap<AnfNodePtr, std::set<size_t>> &producer_streams) {
  MS_LOG(DEBUG) << "Start UpdateEventsToExecutionOrder...";
  std::map<AnfNodePtr, std::set<size_t>> side_effect_map;
  std::map<size_t, std::set<size_t>> no_event_streams;  // wait_stream -> record_stream
  auto exec_kernels = kernel_graph->execution_order();
  std::vector<CNodePtr> new_exec_orders;
  std::set<size_t> streams_set;
  for (auto &kernel : exec_kernels) {
    auto process_stream_id = AnfAlgo::GetStreamId(kernel);
    if (process_stream_id != kDefaultStreamIndex) {
      streams_set.insert(process_stream_id);
    }
    no_event_streams[process_stream_id] = {};
  }
  for (const auto &stream : streams_set) {
    AddBoundarySendRecvKernel(kernel_graph, kDefaultStreamIndex, stream, &new_exec_orders, &no_event_streams);
  }
  CNodePtr last_kernel = nullptr;
  for (auto &kernel : exec_kernels) {
    auto before_iter = recv_before_node.find(kernel);
    if (before_iter != recv_before_node.end()) {
      (void)std::copy(before_iter->second.begin(), before_iter->second.end(), std::back_inserter(new_exec_orders));
    }
    auto process_stream_id = AnfAlgo::GetStreamId(kernel);
    if (process_stream_id != kDefaultStreamIndex) {
      AddBoundarySendRecvKernel(kernel_graph, kDefaultStreamIndex, process_stream_id, &new_exec_orders,
                                &no_event_streams, last_kernel, kernel);
      auto it = producer_streams.find(kernel);
      if (it != producer_streams.end()) {
        for (auto record_stream_id : it->second) {
          if (record_stream_id == kDefaultStreamIndex) {
            continue;
          }
          AddBoundarySendRecvKernel(kernel_graph, record_stream_id, process_stream_id, &new_exec_orders,
                                    &no_event_streams, last_kernel, kernel);
        }
      }
    }
    std::vector<AnfNodePtr> real_inputs;
    ProcessSideEffect(kernel_graph, kernel, process_stream_id, last_kernel, &real_inputs, &side_effect_map,
                      &no_event_streams, &new_exec_orders);

    for (auto real_input : real_inputs) {
      side_effect_map[real_input].insert(process_stream_id);
    }
    for (auto &kv : no_event_streams) {
      if (kv.first != process_stream_id) {
        kv.second.insert(process_stream_id);
      }
    }
    new_exec_orders.push_back(kernel);
    last_kernel = kernel;
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
  for (const auto &stream : streams_set) {
    AddBoundarySendRecvKernel(kernel_graph, stream, kDefaultStreamIndex, &new_exec_orders, &no_event_streams);
  }

  kernel_graph->set_execution_order(new_exec_orders);
  MS_LOG(DEBUG) << "Finish UpdateEventsToExecutionOrder.";
}

void AclStreamAssign::ProcessStreamForInputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                             const NodeIoExecInfoPtr &io_exec_info,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                                             mindspore::HashMap<AnfNodePtr, std::set<size_t>> *producer_streams) {
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
    (*producer_streams)[kernel].insert(input_exec.second->stream_id);
  }
}

void AclStreamAssign::InsertEventsForOutputs(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                             const NodeIoExecInfoPtr &io_exec_info,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv) {
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
}

CNodePtr AclStreamAssign::CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                uint32_t stream_id, uint32_t event_generate_id) {
  auto send_op = std::make_shared<Primitive>(kStreamSendOpName);
  MS_EXCEPTION_IF_NULL(send_op);
  auto send_apply = std::make_shared<ValueNode>(send_op);
  MS_EXCEPTION_IF_NULL(send_apply);
  auto send_node_ptr = graph_ptr->NewCNode({send_apply});
  MS_EXCEPTION_IF_NULL(send_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), send_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrRecordWaitEventStreamPairId, MakeValue(event_generate_id), send_node_ptr);
  AnfAlgo::SetStreamId(stream_id, send_node_ptr.get());
  return send_node_ptr;
}

CNodePtr AclStreamAssign::CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id,
                                                uint32_t record_stream_id, uint32_t stream_id,
                                                uint32_t event_generate_id) {
  auto recv_op = std::make_shared<Primitive>(kStreamRecvOpName);
  MS_EXCEPTION_IF_NULL(recv_op);
  auto recv_apply = std::make_shared<ValueNode>(recv_op);
  MS_EXCEPTION_IF_NULL(recv_apply);
  auto recv_node_ptr = graph_ptr->NewCNode({recv_apply});
  MS_EXCEPTION_IF_NULL(recv_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrEventId, MakeValue(event_id), recv_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEventStream, MakeValue(record_stream_id), recv_node_ptr);
  common::AnfAlgo::SetNodeAttr(kAttrRecordWaitEventStreamPairId, MakeValue(event_generate_id), recv_node_ptr);
  AnfAlgo::SetStreamId(stream_id, recv_node_ptr.get());
  return recv_node_ptr;
}

void AclStreamAssign::InsertEvents(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &parallel_cnode,
                                   const AnfNodePtr &node_before_send,
                                   mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                   mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                                   const AnfNodePtr &node_after_recv) {
  MS_EXCEPTION_IF_NULL(kernel_send);
  MS_EXCEPTION_IF_NULL(kernel_recv);
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  uint32_t event_id = resource_manager.ApplyNewEvent();
  auto event = resource_manager.ApplyRtEvent();
  auto send_stream_id = AnfAlgo::GetStreamId(node_before_send);
  auto event_generate_id = ++event_generate_id_;
  auto send_cnode = CreateSendApplyKernel(kernel_graph, event_id, send_stream_id, event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), send_cnode);
  auto send_iter = kernel_send->find(node_before_send);
  if (send_iter == kernel_send->end()) {
    (*kernel_send)[node_before_send] = {send_cnode};
  } else {
    send_iter->second.push_back(send_cnode);
  }

  CNodePtr recv_cnode = CreateRecvApplyKernel(kernel_graph, event_id, send_stream_id,
                                              AnfAlgo::GetStreamId(node_after_recv), event_generate_id);
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), recv_cnode);
  auto process_iter = kernel_recv->find(node_after_recv);
  if (process_iter == kernel_recv->end()) {
    (*kernel_recv)[node_after_recv] = {recv_cnode};
  } else {
    process_iter->second.push_back(recv_cnode);
  }
  SetForSwitchInline(kernel_graph, send_cnode, recv_cnode, node_before_send, node_after_recv);
}

CNodePtr AclStreamAssign::GetTargetRecvKernel(const std::vector<CNodePtr> &recv_ops, int64_t target_micro,
                                              int64_t target_chunk, bool target_phase) {
  for (const auto &recv : recv_ops) {
    auto micro_attr = recv->GetPrimalAttr(kPrimalAttrMicro);
    MS_EXCEPTION_IF_NULL(micro_attr);
    auto micro = GetValue<int64_t>(micro_attr);
    if (micro != target_micro) {
      continue;
    }
    auto chunk_attr = recv->GetPrimalAttr(kPrimalAttrChunk);
    MS_EXCEPTION_IF_NULL(chunk_attr);
    auto chunk = GetValue<int64_t>(chunk_attr);
    if (chunk != target_chunk) {
      continue;
    }
    auto phase = recv->HasPrimalAttr(kPrimalAttrForwardNodeName);
    if (phase != target_phase) {
      continue;
    }
    return recv;
  }
  return nullptr;
}

void AclStreamAssign::InsertEventsForSendOp(const NotNull<KernelGraphPtr> &kernel_graph, const CNodePtr &kernel,
                                            const std::vector<CNodePtr> &recv_ops, int64_t chunk_num,
                                            int64_t micro_size,
                                            mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                            mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto is_pp_interleave = parallel_context->pipeline_interleave();
  auto pp_scheduler = parallel_context->pipeline_scheduler();
  if (!is_pp_interleave || pp_scheduler != parallel::kPipeline1F1B) {
    return;
  }
  if (!IsPrimitiveCNode(kernel, std::make_shared<Primitive>(kSendOpName))) {
    return;
  }
  auto micro_attr = kernel->GetPrimalAttr(kPrimalAttrMicro);
  MS_EXCEPTION_IF_NULL(micro_attr);
  auto micro = GetValue<int64_t>(micro_attr);
  auto chunk_attr = kernel->GetPrimalAttr(kPrimalAttrChunk);
  MS_EXCEPTION_IF_NULL(chunk_attr);
  auto chunk = GetValue<int64_t>(chunk_attr);
  auto is_backward = kernel->HasPrimalAttr(kPrimalAttrForwardNodeName);
  auto is_param = kernel->HasPrimalAttr(kPrimalAttrPipelineParam);
  auto stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  auto bias = micro_size % stage_num;
  auto loop_index = micro / (stage_num + bias);
  auto offset = (loop_index != 0) ? 0 : bias;
  CNodePtr recv = nullptr;
  if (!is_backward) {
    if (is_param) {
      recv = GetTargetRecvKernel(recv_ops, 0, 1, false);
    } else {
      recv = (chunk != chunk_num - 1) ? GetTargetRecvKernel(recv_ops, micro, chunk + 1, false)
                                      : GetTargetRecvKernel(recv_ops, micro, chunk, true);
    }
  } else {
    if (is_param) {
      return;
    } else {
      recv = (chunk != 0) ? GetTargetRecvKernel(recv_ops, micro, chunk - 1, true)
                          : GetTargetRecvKernel(recv_ops, micro + stage_num + offset, chunk_num - 1, true);
    }
  }
  if (recv == nullptr) {
    return;
  }
  InsertEvents(kernel_graph, kernel, kernel, kernel_send, kernel_recv, recv);
}

static std::tuple<std::vector<CNodePtr>, int64_t, int64_t> GetAllRecvs(const std::vector<CNodePtr> &exec_kernels) {
  int64_t chunk_num = 1;
  int64_t micro_size = 1;
  std::vector<CNodePtr> recv_ops;
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto is_pp_interleave = parallel_context->pipeline_interleave();
  auto pp_scheduler = parallel_context->pipeline_scheduler();
  if (!is_pp_interleave || pp_scheduler != parallel::kPipeline1F1B) {
    return std::make_tuple(recv_ops, chunk_num, micro_size);
  }
  for (const auto &process_kernel : exec_kernels) {
    if (!IsPrimitiveCNode(process_kernel, std::make_shared<Primitive>(kReceiveOpName))) {
      continue;
    }
    if (process_kernel->HasPrimalAttr(kPrimalAttrPipelineParam)) {
      continue;
    }
    recv_ops.emplace_back(process_kernel);
    auto chunk_attr = process_kernel->GetPrimalAttr(kPrimalAttrChunk);
    MS_EXCEPTION_IF_NULL(chunk_attr);
    auto chunk = GetValue<int64_t>(chunk_attr) + 1;
    chunk_num = chunk_num > chunk ? chunk_num : chunk;
    auto micro_attr = process_kernel->GetPrimalAttr(kPrimalAttrMicro);
    MS_EXCEPTION_IF_NULL(micro_attr);
    auto micro = GetValue<int64_t>(micro_attr) + 1;
    micro_size = micro_size > micro ? micro_size : micro;
  }
  return std::make_tuple(recv_ops, chunk_num, micro_size);
}

void AclStreamAssign::GenEventsForParallelOp(const NotNull<KernelGraphPtr> &kernel_graph,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_send,
                                             mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> *kernel_recv,
                                             mindspore::HashMap<AnfNodePtr, std::set<size_t>> *producer_streams) {
  MS_LOG(DEBUG) << "Start GenEventsForParallelOp...";
  constexpr size_t INDEX_TWO = 2;
  auto exec_kernels = kernel_graph->execution_order();
  mindspore::HashMap<CNodePtr, NodeIoExecInfoPtr> kernel_io_exec_info_map;
  GenKernelIoExecInfoMap(kernel_graph, &kernel_io_exec_info_map);
  auto recvs = GetAllRecvs(exec_kernels);
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
    ProcessStreamForInputs(kernel_graph, process_kernel, process_io_exec_info, kernel_send, kernel_recv,
                           producer_streams);
    InsertEventsForOutputs(kernel_graph, process_kernel, process_io_exec_info, kernel_send, kernel_recv);
    InsertEventsForSendOp(kernel_graph, process_kernel, std::get<0>(recvs), std::get<1>(recvs),
                          std::get<INDEX_TWO>(recvs), kernel_send, kernel_recv);
  }
  MS_LOG(DEBUG) << "Finish GenEventsForParallelOp.";
}

void AclStreamAssign::InsertEventForNonTaskSink(const NotNull<KernelGraphPtr> &kernel_graph) {
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_send;
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> kernel_recv;
  mindspore::HashMap<AnfNodePtr, std::set<size_t>> producer_streams;
  AnfAlgo::SetStreamId(kDefaultStreamIndex, kernel_graph->output().get());
  GenEventsForParallelOp(kernel_graph, &kernel_send, &kernel_recv, &producer_streams);
  UpdateEventsToExecutionOrder(kernel_graph, kernel_send, kernel_recv, producer_streams);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
