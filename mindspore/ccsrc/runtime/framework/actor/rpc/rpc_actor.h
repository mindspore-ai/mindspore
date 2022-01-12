/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTO_RPC_RPC_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTO_RPC_RPC_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "runtime/framework/actor/debug_aware_actor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::KernelInfo;

// RpcActor is used to do rpc with other processes in distributed execution.
// Besides data arrows and controlling arrows, RpcActor also has iter-process arrows which is in charge of remote
// communication with other processes. It supports both sync and async communication.
class RpcActor : public DebugAwareActor {
 public:
  RpcActor(const std::string &name, KernelTransformType type, const CNodePtr &kernel,
           const DeviceContext *device_context, const AID &memory_manager_aid, const AID *debug_aid,
           const AID *recorder_aid)
      : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid),
        rpc_kernel_(kernel),
        kernel_info_(nullptr) {
    (void)device_contexts_.emplace_back(device_context);
  }
  ~RpcActor() override = default;

  const CNodePtr &kernel() const { return rpc_kernel_; }

 protected:
  // The arrows represent iter-process communication.
  std::vector<AID> iter_process_input_arrows_;
  std::vector<AID> iter_process_output_arrows_;

 private:
  friend class GraphScheduler;

  CNodePtr rpc_kernel_;
  KernelInfo *kernel_info_;
};

using RpcActorPtr = std::shared_ptr<RpcActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTO_RPC_RPC_ACTOR_H_
