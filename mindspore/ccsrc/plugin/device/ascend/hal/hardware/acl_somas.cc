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

#include "plugin/device/ascend/hal/hardware/acl_somas.h"
#include <string>
#include <map>
#include <utility>
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "utils/ms_context.h"
#include "ops/framework_op_name.h"
#include "mindspore/core/ops/framework_ops.h"

namespace mindspore {
namespace device {
namespace ascend {
using TensorType = somas::TensorType;
using LifeLongType = somas::LifeLongType;
constexpr size_t ALONE = 1;

bool AclSomas::Initialize() { return true; }

std::string AclSomas::GetDeviceName() const { return "Ascend"; }

size_t AclSomas::GetAlignSize(size_t original_size) const {
  constexpr size_t alignment = 512;
  constexpr size_t alignment_complement = 31;
  size_t aligned_size =
    (original_size > 0) ? ((original_size + alignment + alignment_complement) / alignment) * alignment : 0;
  return aligned_size;
}

bool AclSomas::GetDependExecOrderFlag(const session::KernelGraph &graph) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto opt_level = ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL);
  return opt_level != kOptimizeO0;
}

bool AclSomas::InitDevSpecControlTensors(const session::KernelGraph &graph) {
  InitEventInfo(graph);
  return true;
}

void AclSomas::InitEventInfo(const session::KernelGraph &graph) {
  MS_LOG(DEBUG) << "Acl Somas InitEventInfo start.";
  event_map_ = {};
  auto &kernels = graph.execution_order();
  for (const auto &kernel : kernels) {
    auto type = common::AnfAlgo::GetCNodeName(kernel);
    if (type == kStreamSendOpName) {
      auto event = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel, kAttrEventId);
      auto iter = event_map_.find(event);
      if (iter == event_map_.end()) {
        auto pair = somas::EventPair();
        pair.send_ = kernel;
        event_map_[event] = pair;
      } else {
        iter->second.send_ = kernel;
      }
    } else if (type == kStreamRecvOpName) {
      auto event = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel, kAttrEventId);
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
    auto send_iter = nodes_map_.find(event.second.send_.get());
    auto recv_iter = nodes_map_.find(event.second.recv_.get());
    if (send_iter == nodes_map_.end()) {
      MS_LOG(WARNING) << "Can't find Acl somas node for " << event.second.send_->fullname_with_scope();
      continue;
    }
    if (recv_iter == nodes_map_.end()) {
      MS_LOG(WARNING) << "Can't find Acl somas node for " << event.second.recv_->fullname_with_scope();
      continue;
    }
    AddControlTensor(send_iter->second.at(0), recv_iter->second.at(0));
  }
  MS_LOG(DEBUG) << "Acl Somas InitEventInfo end.";
}

bool AclSomas::RuntimeNodeProcess(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  for (auto &kernel : kernels) {
    if (!IsPrimitiveCNode(kernel, {prim::kPrimConditionGather})) {
      continue;
    }
    auto iter = nodes_map_.find(kernel.get());
    if (iter != nodes_map_.end()) {
      auto &node = iter->second.at(0);
      MS_EXCEPTION_IF_NULL(node);
      auto input_tensors = node->input_tensors_;
      auto output_tensors = node->output_tensors_;
      MS_EXCEPTION_IF_CHECK_FAIL(input_tensors.size() == output_tensors.size() * 2,
                                 "Invalid input and output tensors size" + std::to_string(input_tensors.size()) + ", " +
                                   std::to_string(output_tensors.size()));
      std::vector<std::vector<size_t>> union_tensors;
      for (auto &tensor : output_tensors) {
        tensor->type_ = somas::kUnion;
        union_tensors.push_back({tensor->GetId()});
      }
      for (size_t i = 0; i < output_tensors.size(); i++) {
        input_tensors[i]->type_ = somas::kUnion;
        input_tensors[i + output_tensors.size()]->type_ = somas::kUnion;
        union_tensors[i].push_back(input_tensors[i]->GetId());
        union_tensors[i].push_back(input_tensors[i + output_tensors.size()]->GetId());
      }
      union_tensors_list_.insert(union_tensors_list_.end(), union_tensors.begin(), union_tensors.end());
    } else {
      MS_LOG(EXCEPTION) << "Can't find somas node for inplace node " << kernel->fullname_with_scope();
    }
  }
  return true;
}

bool AclSomas::DevSpecNodeProcess(const session::KernelGraph &graph) { return RuntimeNodeProcess(graph); }

void AclSomas::CommunicationTensorProcess(const std::vector<somas::SomasTensorPtr> &tensors) const {
  if (tensors.size() != ALONE) {
    for (auto &tensor : tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      MS_EXCEPTION_IF_CHECK_FAIL(tensor->aligned_size_ != 0, "The size of communication tensor is zero, tensor id: " +
                                                               std::to_string(tensor->GetId()));
    }
  }
}

bool AclSomas::NeedContiguous(const std::vector<size_t> &inputs) const { return inputs.size() > ALONE; }

bool AclSomas::NeedReuseGraphOutput() const { return true; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
