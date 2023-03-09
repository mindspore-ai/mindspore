/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/hardware/gpu_somas.h"
#include <string>
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace gpu {
constexpr size_t ALONE = 1;

bool GPUSomas::Initialize() { return true; }

std::string GPUSomas::GetDeviceName() const { return "GPU"; }

size_t GPUSomas::GetAlignSize(size_t original_size) const {
  constexpr size_t alignment = 512;
  size_t aligned_size = (original_size > 0) ? ((original_size + alignment - 1) / alignment) * alignment : 0;
  return aligned_size;
}

bool GPUSomas::GetDependExecOrderFlag(const session::KernelGraph &graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) != kOptimizeO0) {
    return true;
  } else {
    return false;
  }
}

bool GPUSomas::InitDevSpecControlTensors(const session::KernelGraph &graph) {
  InitEventInfo(graph);
  return true;
}

void GPUSomas::InitEventInfo(const session::KernelGraph &graph) {
  event_map_ = {};
  auto &kernels = graph.execution_order();
  for (const auto &kernel : kernels) {
    auto type = common::AnfAlgo::GetCNodeName(kernel);
    if (type == kSendOpName) {
      auto event = common::AnfAlgo::GetNodeAttr<uintptr_t>(kernel, kAttrRecordEvent);
      auto iter = event_map_.find(event);
      if (iter == event_map_.end()) {
        auto pair = somas::EventPair();
        pair.send_ = kernel;
        event_map_[event] = pair;
      } else {
        iter->second.send_ = kernel;
      }
    } else if (type == kRecvOpName) {
      auto event = common::AnfAlgo::GetNodeAttr<uintptr_t>(kernel, kAttrWaitEvent);
      auto iter = event_map_.find(event);
      if (iter == event_map_.end()) {
        auto pair = somas::EventPair();
        pair.recv_ = kernel;
        event_map_[event] = pair;
      } else {
        iter->second.recv_ = kernel;
      }
    }
  }

  for (auto &event : event_map_) {
    auto pair = event.second;
    auto send_iter = nodes_map_.find(pair.send_.get());
    if (send_iter == nodes_map_.end()) {
      MS_LOG(WARNING) << "Can't find GPU somas node for " << pair.send_->fullname_with_scope();
      continue;
    }

    auto recv_iter = nodes_map_.find(pair.recv_.get());
    if (recv_iter == nodes_map_.end()) {
      MS_LOG(WARNING) << "Can't find GPU somas node for " << pair.recv_->fullname_with_scope();
      continue;
    }

    auto &somas_send = send_iter->second.at(0);
    auto &somas_recv = recv_iter->second.at(0);
    AddControlTensor(somas_send, somas_recv);
  }
  MS_LOG(DEBUG) << "GPU Somas InitEventInfo end.";
}

bool GPUSomas::DevSpecNodeProcess(const session::KernelGraph &graph) { return InplaceNodeProcess(graph); }

bool GPUSomas::InplaceNodeProcess(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  for (auto &kernel : kernels) {
    if (!common::AnfAlgo::IsInplaceNode(kernel, "skip")) {
      continue;
    }
    auto iter = nodes_map_.find(kernel.get());
    if (iter != nodes_map_.end()) {
      auto &node = iter->second.at(0);
      MS_EXCEPTION_IF_NULL(node);
      auto input_tensors = node->input_tensors_;
      auto output_tensors = node->output_tensors_;
      std::vector<somas::SomasTensorPtr> union_tensors;
      union_tensors.insert(union_tensors.end(), input_tensors.begin(), input_tensors.end());
      union_tensors.insert(union_tensors.end(), output_tensors.begin(), output_tensors.end());
      // check whether the union tensor already in other union tensors
      for (auto &tensor : union_tensors) {
        auto tensor_id = tensor->GetId();
        for (auto &union_list : union_tensors_list_) {
          if (std::count(union_list.begin(), union_list.end(), tensor_id)) {
            MS_LOG(EXCEPTION) << "Inplace node union Tensor " << tensor_id << " already in other union tensor list.";
          }
        }
      }
      std::vector<size_t> inplace_union_tensor_list;
      for (auto &tensor : union_tensors) {
        tensor->type_ = somas::kUnion;
        inplace_union_tensor_list.push_back(tensor->GetId());
      }

      union_tensors_list_.push_back(inplace_union_tensor_list);
    } else {
      MS_LOG(EXCEPTION) << "Can't find somas node for inplace node " << kernel->fullname_with_scope();
    }
  }
  return true;
}

void GPUSomas::CommunicationTensorProcess(const std::vector<somas::SomasTensorPtr> &tensors) const {
  if (tensors.size() != ALONE) {
    for (auto &tensor : tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      MS_EXCEPTION_IF_CHECK_FAIL(tensor->aligned_size_ != 0, "The size of communication tensor is zero, tensor id: " +
                                                               std::to_string(tensor->GetId()));
    }
  }
}

bool GPUSomas::NeedContiguous(const std::vector<size_t> &inputs) const { return inputs.size() > ALONE; }
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
