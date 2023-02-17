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

#include "src/control_flow/control_actor_creator.h"
#include "src/litert/kernel_exec_util.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/control_flow/actor/switch_actor.h"
#include "src/control_flow/actor/entrance_actor.h"
#include "src/control_flow/actor/exit_actor.h"
#include "src/litert/parallel_lite_actor.h"

namespace mindspore::lite {
std::shared_ptr<LiteOpActor> CreateActor(kernel::KernelExec *kernel, lite::InnerContext *ctx) {
  std::shared_ptr<LiteOpActor> actor = nullptr;
  if (kernel::KernelExecUtil::IsSwitchTypeCall(kernel)) {
    actor = std::make_shared<LiteSwitchOpActor>(kernel, ctx);
  } else if (kernel->subgraph_type() == kernel::kEntranceSubGraph) {
    actor = std::make_shared<LiteEntranceOpActor>(kernel, ctx);
  } else if (kernel->subgraph_type() == kernel::kExitSubGraph) {
    actor = std::make_shared<LiteExitOpActor>(kernel, ctx);
  } else if (kernel->subgraph_type() != kernel::kNotSubGraph) {
    auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
    if (subgraph_kernel->nodes().size() > 1 && ctx->inter_op_parallel_num_ > 1 &&
        (kernel->subgraph_type() == kernel::kCpuFP32SubGraph || kernel->subgraph_type() == kernel::kCpuFP16SubGraph)) {
      actor = std::make_shared<ParallelLiteActor>(kernel, ctx);
    } else {
      actor = std::make_shared<LiteOpActor>(kernel, ctx);
    }
  } else {
    actor = std::make_shared<LiteOpActor>(kernel, ctx);
  }
  return actor;
}
}  // namespace mindspore::lite

#else
namespace mindspore::lite {
std::shared_ptr<LiteOpActor> CreateActor(kernel::KernelExec *kernel, lite::InnerContext *ctx) {
  if (kernel::KernelExecUtil::IsSwitchTypeCall(kernel) || (kernel->subgraph_type() == kernel::kEntranceSubGraph) ||
      (kernel->subgraph_type() == kernel::kExitSubGraph)) {
    return nullptr;
  }
  return std::make_shared<LiteOpActor>(kernel, ctx);
}
}  // namespace mindspore::lite
#endif
