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

#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"

namespace mindspore {
namespace runtime {
void KernelAsyncInferActor::InferShape(OpContext<DeviceTensor> *const context, KernelActor *kernel_actor) {
  try {
    kernel_actor->ExecuteInferShapeTask(context);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    MS_LOG(ERROR) << "Failed to infer shape for kernel: " << kernel_actor->kernel()->fullname_with_scope()
                  << " and catch exception: " << e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
  }
}

void KernelAsyncInferActor::Wait() {
  MS_LOG(DEBUG) << "Begin wait kernel infer finish";
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kWaitKernelsInferFinish, GetAID().Name());
  Future<bool> f = Async(this->GetAID(), &KernelAsyncInferActor::OnTaskFinish);
  f.Wait();
  MS_LOG(DEBUG) << "End wait kernel infer finish";
}

Future<bool> KernelAsyncInferActor::OnTaskFinish() { return Future<bool>(true); }
}  // namespace runtime
}  // namespace mindspore
