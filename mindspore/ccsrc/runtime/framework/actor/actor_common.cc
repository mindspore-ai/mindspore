/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/actor/actor_common.h"
#include <unistd.h>
#ifdef __WIN32__
#include <windows.h>
#endif
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/framework/device_tensor_store.h"

namespace mindspore {
namespace runtime {
int64_t GetMaxThreadNum() {
#ifdef __WIN32__
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  auto max_thread_num = sys_info.dwNumberOfProcessors;
#else
  auto max_thread_num = sysconf(_SC_NPROCESSORS_ONLN);
#endif

  return max_thread_num;
}

bool IsDeviceQueueDSActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) == kGetNextOpName)) {
    return true;
  }
  return false;
}

bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph, const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
    // There is device address in tensor, indicating the input tensor is certain kernel's output,
    // so it's unnecessary to put the input node to host queue data source actor.
    if (tensor != nullptr && std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address()) != nullptr) {
      return false;
    }

    if (graph == nullptr) {
      return true;
    }

    //  Judge whether node is internal parameter.
    const auto &front_node = graph->GetFrontNodeByInternalParameter(node);
    if (front_node.first == nullptr) {
      return true;
    }
  }
  return false;
}

bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
    //  Judge whether node is internal parameter.
    const auto &front_node = graph->GetFrontNodeByInternalParameter(node);
    if (front_node.first != nullptr) {
      return true;
    }
  }
  return false;
}

bool IsKernelActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) != kGetNextOpName)) {
    return true;
  }
  return false;
}

bool IsSkippedKernelActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && AnfAlgo::IsInplaceNode(node, "skip")) {
    return true;
  }
  return false;
}

bool IsPersistentDeviceTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return true;
  }
  if (node->isa<Parameter>() && AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

bool IsGatherActor(const AnfNodePtr &front_node,
                   const std::unordered_map<std::string, OpActor<DeviceTensor> *> &actor_name_to_actor_) {
  if (front_node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(front_node->cast<ParameterPtr>())) &&
      front_node->func_graph() != nullptr) {
    const auto &func_graph = front_node->func_graph();
    if (func_graph != nullptr && actor_name_to_actor_.find(func_graph->ToString()) != actor_name_to_actor_.end()) {
      return true;
    }
  }
  return false;
}
}  // namespace runtime
}  // namespace mindspore
