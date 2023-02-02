/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/device/gpu_stream_assign.h"
#include <set>
#include <string>
#include <memory>
#include <algorithm>
#include <utility>
#include "kernel/graph_kernel_info.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/hardware/gpu_device_context.h"

namespace mindspore {
namespace device {
namespace gpu {
void AssignGpuStream(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<CNodePtr> allreduce_kernels;
  auto execution_kernels = kernel_graph->execution_order();
  // All operators in pynative mode use default_stream.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool enable_mem_offload =
    ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD) && !kernel_graph->is_dynamic_shape();
  const bool use_default_stream = kernel_graph->has_flag(kFlagPyNativeRunInGraph) || enable_mem_offload;
  if (use_default_stream) {
    AssignDefaultGpuStream(kernel_graph);
    return;
  }
  const auto default_stream_id = GPUDeviceManager::GetInstance().default_stream_id();
  for (const auto &kernel_node : execution_kernels) {
    if (common::AnfAlgo::GetCNodeName(kernel_node) == kAllReduceOpName) {
      allreduce_kernels.emplace_back(kernel_node);
    } else {
      AnfAlgo::SetStreamId(default_stream_id, kernel_node.get());
    }
  }
  std::vector<SendRecvPair> send_recv_pairs;
  const bool multi_stream =
    allreduce_kernels.size() > 1 && FindAllReduceStreamSwitchPos(kernel_graph, &send_recv_pairs);
  // Assign multiple streams only when there're multiple AllReduce nodes.
  size_t allreduce_stream_id = default_stream_id;
  if (multi_stream) {
    if (!GPUDeviceManager::GetInstance().CreateStream(&allreduce_stream_id)) {
      MS_LOG(EXCEPTION) << "Fail to create new stream for communication operator";
    }
    std::transform(allreduce_kernels.begin(), allreduce_kernels.end(), allreduce_kernels.begin(),
                   [&](CNodePtr allreduce_kernel) {
                     AnfAlgo::SetStreamId(allreduce_stream_id, allreduce_kernel.get());
                     return allreduce_kernel;
                   });
    InsertStreamSwitchNode(kernel_graph, send_recv_pairs);
  } else {
    std::transform(allreduce_kernels.begin(), allreduce_kernels.end(), allreduce_kernels.begin(),
                   [&](CNodePtr allreduce_kernel) {
                     AnfAlgo::SetStreamId(default_stream_id, allreduce_kernel.get());
                     return allreduce_kernel;
                   });
  }
}

void AssignDefaultGpuStream(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto default_stream_id = GPUDeviceManager::GetInstance().default_stream_id();
  for (const auto &kernel_node : kernel_graph->execution_order()) {
    AnfAlgo::SetStreamId(default_stream_id, kernel_node.get());
  }
}

bool FindAllReduceStreamSwitchPos(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                  std::vector<SendRecvPair> *send_recv_pairs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(send_recv_pairs);
  auto execution_kernels = kernel_graph->execution_order();
  std::vector<CNodePtr>::iterator iter;
  std::vector<CNodePtr>::iterator iter_begin;
  iter = iter_begin = execution_kernels.begin();
  std::vector<CNodePtr>::iterator iter_end = execution_kernels.end();
  for (; iter != execution_kernels.end(); ++iter) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(*iter);
    if (kernel_name == kAllReduceOpName) {
      // Find AllReduce node's last input node.
      std::vector<CNodePtr>::iterator mock_send_node_iter =
        FindSendNodePos(iter_begin, iter + 1, *iter, kAllReduceStreamSwitch);
      if (mock_send_node_iter == iter + 1) {
        MS_LOG(INFO) << "Can't find send node place before AllReduce node.";
      } else if (common::AnfAlgo::GetCNodeName(*mock_send_node_iter) != kAllReduceOpName) {
        SendRecvPair pair1 = {kAllReduceStreamSwitch, *mock_send_node_iter, *iter,
                              IntToSize(mock_send_node_iter - iter_begin + 1), IntToSize(iter - iter_begin)};
        send_recv_pairs->push_back(pair1);
      } else {
        MS_LOG(INFO) << "mock_send_node is AllReduce, no need to add stream switch node.";
      }
      // Find node which uses AllReduce as input[0].
      std::vector<CNodePtr>::iterator mock_recv_node_iter =
        FindRecvNodePos(iter, iter_end, *iter, kAllReduceStreamSwitch);
      if (mock_recv_node_iter == iter_end) {
        // Each AllReduce must have its corresponding node which takes AllReduce as a input to synchronize stream,
        // otherwise consider FindAllReduceStreamSwitchPos as failed.
        MS_LOG(INFO) << "Can't find recv node place after AllReduce node.";
        return false;
      } else if (common::AnfAlgo::GetCNodeName(*mock_recv_node_iter) != kAllReduceOpName) {
        SendRecvPair pair2 = {kAllReduceStreamSwitch, *iter, *mock_recv_node_iter, IntToSize(iter - iter_begin + 1),
                              IntToSize(mock_recv_node_iter - iter_begin)};
        send_recv_pairs->push_back(pair2);
      } else {
        MS_LOG(INFO) << "mock_recv_node is AllReduce, no need to add stream switch node.";
      }
    }
  }
  if (send_recv_pairs->empty()) {
    MS_LOG(INFO) << "No stream switch node is found.";
    return false;
  }
  return true;
}

std::vector<CNodePtr>::iterator FindSendNodePos(std::vector<CNodePtr>::iterator begin,
                                                std::vector<CNodePtr>::iterator end, const CNodePtr mock_recv_node,
                                                StreamSwitchType stream_switch_type) {
  MS_EXCEPTION_IF_NULL(mock_recv_node);
  if (stream_switch_type == kAllReduceStreamSwitch) {
    for (auto iter = begin; iter != end; iter++) {
      if (*(iter + 1) == mock_recv_node) {
        return iter;
      }
    }
  }
  return end;
}

std::vector<CNodePtr>::iterator FindRecvNodePos(std::vector<CNodePtr>::iterator begin,
                                                std::vector<CNodePtr>::iterator end, const CNodePtr mock_send_node,
                                                StreamSwitchType stream_switch_type) {
  MS_EXCEPTION_IF_NULL(mock_send_node);
  auto ret = end;
  for (auto iter = begin; iter != end; iter++) {
    auto node = *iter;
    if (stream_switch_type == kAllReduceStreamSwitch) {
      MS_EXCEPTION_IF_NULL(node);
      for (auto input : node->inputs()) {
        if (mock_send_node == common::AnfAlgo::VisitKernel(input, 0).first) {
          if (common::AnfAlgo::GetCNodeName(node) != kAllReduceOpName) {
            return iter;
          } else if (ret == end) {
            ret = iter;
          }
        }
      }
    }
  }
  return ret;
}

void InsertStreamSwitchNode(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                            const std::vector<SendRecvPair> &send_recv_pairs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::set<StreamSwitchNode> ordered_stream_switch_nodes;
  for (SendRecvPair pair : send_recv_pairs) {
    StreamSwitchType stream_switch_type = pair.stream_switch_type;
    CNodePtr mock_send_node = pair.mock_send_node;
    CNodePtr mock_recv_node = pair.mock_recv_node;
    size_t send_node_offset = pair.send_node_offset;
    size_t recv_node_offset = pair.recv_node_offset;
    CNodePtr send_node = nullptr;
    CNodePtr recv_node = nullptr;
    // Step 1: Generate stream Send and Recv CNodes.
    if (stream_switch_type == kAllReduceStreamSwitch) {
      if (!GenSendRecvCNodesForAllReduce(kernel_graph, mock_send_node, mock_recv_node, &send_node, &recv_node)) {
        MS_LOG(EXCEPTION) << "Generating CNodes for send and recv failed. Stream switch type: kAllReduceStreamSwitch";
      }

      CacheSendRecvCNodesForAllReduce(kernel_graph, mock_send_node, mock_recv_node, send_node, recv_node);
    }
    // Step 2: Sort send and recv CNodes by offset.
    ordered_stream_switch_nodes.insert({send_node_offset, send_node});
    ordered_stream_switch_nodes.insert({recv_node_offset, recv_node});
  }
  // Step 3: Insert stream switch CNodes into execution kernel list.
  auto execution_kernels = kernel_graph->execution_order();
  for (auto node = ordered_stream_switch_nodes.rbegin(); node != ordered_stream_switch_nodes.rend(); node++) {
    execution_kernels.insert(execution_kernels.begin() + node->offset, node->cnode);
  }
  kernel_graph->set_execution_order(execution_kernels);
}

bool GenSendRecvCNodesForAllReduce(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                   const CNodePtr &mock_send_node, const CNodePtr &mock_recv_node, CNodePtr *send_node,
                                   CNodePtr *recv_node) {
  *send_node = CreateStreamSwitchNode(kernel_graph, kSendOpName);
  MS_EXCEPTION_IF_NULL(*send_node);
  *recv_node = CreateStreamSwitchNode(kernel_graph, kRecvOpName);
  MS_EXCEPTION_IF_NULL(*recv_node);

  cudaEvent_t event = nullptr;
  std::weak_ptr<CNode> send_node_ = *send_node;
  CHECK_CUDA_RET_WITH_EXCEPT(send_node_, cudaEventCreate(&event, cudaEventDisableTiming),
                             "Creating cuda event failed.");
  common::AnfAlgo::SetNodeAttr(kAttrRecordEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), *send_node);
  common::AnfAlgo::SetNodeAttr(kAttrWaitEvent, MakeValue(reinterpret_cast<uintptr_t>(event)), *recv_node);

  const size_t send_stream_id = AnfAlgo::GetStreamId(mock_send_node);
  AnfAlgo::SetStreamId(send_stream_id, send_node->get());
  const size_t recv_stream_id = AnfAlgo::GetStreamId(mock_recv_node);
  AnfAlgo::SetStreamId(recv_stream_id, recv_node->get());
  return true;
}

CNodePtr CreateStreamSwitchNode(const std::shared_ptr<session::KernelGraph> &kernel_graph, const std::string &name) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto op = std::make_shared<Primitive>(name);
  MS_EXCEPTION_IF_NULL(op);
  auto apply = std::make_shared<ValueNode>(op);
  MS_EXCEPTION_IF_NULL(apply);
  std::vector<AnfNodePtr> input_list = {apply};
  CNodePtr node = kernel_graph->NewCNode(input_list);
  MS_EXCEPTION_IF_NULL(node);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), node.get());
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract_none);
  node->set_abstract(abstract_none);
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
  kernel_info_setter->SetKernelInfo(node, KernelType::UNKNOWN_KERNEL_TYPE);
  return node;
}

void CacheSendRecvCNodesForAllReduce(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                     const CNodePtr &mock_send_node, const CNodePtr &mock_recv_node,
                                     const CNodePtr &send_node, const CNodePtr &recv_node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::pair<CNodePtr, CNodePtr> send_recv_nodes(send_node, recv_node);
  if (common::AnfAlgo::GetCNodeName(mock_send_node) == kAllReduceOpName) {
    kernel_graph->InsertSendRecvPairForParallelOpOutputs(mock_send_node, send_recv_nodes);
  }

  if (common::AnfAlgo::GetCNodeName(mock_recv_node) == kAllReduceOpName) {
    kernel_graph->InsertSendRecvPairForParallelOpInputs(mock_recv_node, send_recv_nodes);
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
