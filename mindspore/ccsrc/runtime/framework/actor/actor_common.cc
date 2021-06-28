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
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/framework/device_tensor_store.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace runtime {
void ComputeThreadNums(size_t *actor_thread_num, size_t *OMP_thread_num) {
  MS_EXCEPTION_IF_NULL(actor_thread_num);
  MS_EXCEPTION_IF_NULL(OMP_thread_num);
  size_t cpu_core_num = std::thread::hardware_concurrency();

  const size_t kActorThreadMaxNum = 5;
  // The MemoryManagerActor binds single thread, and the other actors share one thread at least, so the min num is 2.
  const size_t kActorThreadMinNum = 2;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // The pyNative mode is the step execution strategy, so only need the kActorThreadMinNum.
  if (context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) == kPynativeMode) {
    *actor_thread_num = kActorThreadMinNum;
  } else {
    *actor_thread_num = cpu_core_num < kActorThreadMinNum ? kActorThreadMinNum : cpu_core_num;
    *actor_thread_num = *actor_thread_num > kActorThreadMaxNum ? kActorThreadMaxNum : *actor_thread_num;
  }

  const size_t kOMPThreadNumThreshold = 16;
  if (cpu_core_num <= kOMPThreadNumThreshold) {
    *OMP_thread_num = cpu_core_num;
  } else {
    *OMP_thread_num = cpu_core_num / (*actor_thread_num - 1);
  }
}

bool IsDeviceQueueDSActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) == kGetNextOpName)) {
    return true;
  }
  return false;
}

bool IsSwitchActor(const AnfNodePtr &node) { return AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch); }

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
                   const std::unordered_map<std::string, OpActor<DeviceTensor> *> &actor_name_to_actor) {
  if (front_node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(front_node->cast<ParameterPtr>())) &&
      front_node->func_graph() != nullptr) {
    const auto &func_graph = front_node->func_graph();
    if (func_graph != nullptr && actor_name_to_actor.find(func_graph->ToString()) != actor_name_to_actor.end()) {
      return true;
    }
  }
  return false;
}
}  // namespace runtime

// Judge whether to use mindRT. GPU and CPU use mindRT currently, and other hardwares will use it in the future.
// Return false in the transitional stage.
bool IsMindRTUsed() { return false; }
}  // namespace mindspore
