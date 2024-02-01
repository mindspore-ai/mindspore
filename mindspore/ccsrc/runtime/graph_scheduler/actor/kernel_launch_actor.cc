/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/kernel_launch_actor.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"

namespace mindspore {
namespace runtime {
void KernelLaunchActor::LaunchKernel(OpContext<DeviceTensor> *const context, KernelActor *kernel_actor) {
  try {
    kernel_actor->LaunchKernelWithMemManage(context);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
  }
}

void KernelLaunchActor::Wait() {
  if (enable_async_launch_) {
    Future<bool> f = Async(this->GetAID(), &KernelLaunchActor::OnTaskFinish);
    f.Wait();
  }
}

Future<bool> KernelLaunchActor::OnTaskFinish() { return Future<bool>(true); }
}  // namespace runtime
}  // namespace mindspore
