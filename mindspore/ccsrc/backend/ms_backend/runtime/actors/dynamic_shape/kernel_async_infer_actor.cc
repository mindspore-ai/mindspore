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

#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_infer_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"
#include "utils/trace_base.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace runtime {
std::shared_ptr<KernelAsyncInferActor> &KernelAsyncInferActor::GetInstance() {
  static std::shared_ptr<KernelAsyncInferActor> instance =
    std::shared_ptr<KernelAsyncInferActor>(new KernelAsyncInferActor());
  return instance;
}

void KernelAsyncInferActor::Initialize() {
  Async(this->GetAID(), &KernelAsyncInferActor::GetThreadId);
  Wait();
}

void KernelAsyncInferActor::InferShape(OpContext<KernelTensor> *const context, KernelActor *kernel_actor) {
  try {
    kernel_actor->ExecuteInferShapeTask(context);
  } catch (const std::exception &e) {
    if (context->error_info_.empty()) {
      MsException::Instance().SetException();
      auto error_line = trace::DumpSourceLines(kernel_actor->kernel());
      MS_LOG(ERROR) << "Failed to infer shape for kernel: " << kernel_actor->kernel()->fullname_with_scope()
                    << " and catch exception: " << e.what() << error_line;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
    }
  }
}

void KernelAsyncInferActor::InferShapeV2(OpContext<KernelTensor> *const context, KernelRunner *kernel_runner,
                                         bool high_perf) {
  try {
    kernel_runner->ExecuteInferShapeTask(context, high_perf);
  } catch (const std::exception &e) {
    if (context->error_info_.empty()) {
      MsException::Instance().SetException();
      auto error_line = trace::DumpSourceLines(kernel_runner->kernel());
      MS_LOG(ERROR) << "Failed to infer shape for kernel: " << kernel_runner->kernel()->fullname_with_scope()
                    << " and catch exception: " << e.what() << error_line;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
    }
  }
}

void KernelAsyncInferActor::Wait() {
  // To prevent deadlocks, you cannot wait again inside the processing of all messages received by this actor.
  if (thread_id_ == std::this_thread::get_id()) {
    return;
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin wait kernel infer finish";
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kWaitKernelsInferFinish, GetAID().Name());
  Future<bool> f = Async(this->GetAID(), &KernelAsyncInferActor::OnTaskFinish);
  f.Wait();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End wait kernel infer finish";
}

Future<bool> KernelAsyncInferActor::OnTaskFinish() { return Future<bool>(true); }
}  // namespace runtime
}  // namespace mindspore
