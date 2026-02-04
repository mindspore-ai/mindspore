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

#include "backend/ms_backend/runtime/actors/base/kernel_async_launch_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"
#include "utils/trace_base.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace runtime {
std::shared_ptr<KernelAsyncLaunchActor> &KernelAsyncLaunchActor::GetInstance() {
  static std::shared_ptr<KernelAsyncLaunchActor> instance =
    std::shared_ptr<KernelAsyncLaunchActor>(new KernelAsyncLaunchActor());
  return instance;
}

void KernelAsyncLaunchActor::Initialize() {
  Async(this->GetAID(), &KernelAsyncLaunchActor::GetThreadId);
  // Bind LaunchKernel thread to current device.
  Async(this->GetAID(), &KernelAsyncLaunchActor::BindDevice);
  Wait();
}

void KernelAsyncLaunchActor::LaunchKernel(OpContext<KernelTensor> *const context, KernelActor *kernel_actor) {
  try {
    kernel_actor->ExecuteLaunchKernelTask(context);
  } catch (const std::exception &e) {
    if (context->error_info_.empty()) {
      MsException::Instance().SetException();
      auto error_line = trace::DumpSourceLines(kernel_actor->kernel());
      MS_LOG(ERROR) << "Failed to launch kernel: " << kernel_actor->kernel()->fullname_with_scope()
                    << " and catch exception: " << e.what() << error_line;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
    }
  }
}

void KernelAsyncLaunchActor::LaunchKernelV2(OpContext<KernelTensor> *const context, KernelRunner *kernel_runner) {
  try {
    kernel_runner->ExecuteLaunchKernelTask(context);
  } catch (const std::exception &e) {
    if (context->error_info_.empty()) {
      MsException::Instance().SetException();
      auto error_line = trace::DumpSourceLines(kernel_runner->kernel());
      MS_LOG(ERROR) << "Failed to launch kernel: " << kernel_runner->kernel()->fullname_with_scope()
                    << " and catch exception: " << e.what() << error_line;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
    }
  }
}

void KernelAsyncLaunchActor::LaunchKernelV2HP(OpContext<KernelTensor> *const context, KernelRunner *kernel_runner) {
  try {
    kernel_runner->ExecuteLaunchKernelTaskHP(context);
  } catch (const std::exception &e) {
    if (context->error_info_.empty()) {
      MsException::Instance().SetException();
      auto error_line = trace::DumpSourceLines(kernel_runner->kernel());
      MS_LOG(ERROR) << "Failed to launch kernel: " << kernel_runner->kernel()->fullname_with_scope()
                    << " and catch exception: " << e.what() << error_line;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
    }
  }
}

void KernelAsyncLaunchActor::Wait() {
  // To prevent deadlocks, you cannot wait again inside the processing of all messages received by this actor.
  if (thread_id_ == std::this_thread::get_id()) {
    return;
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin wait kernel launch finish";
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kWaitKernelsLaunchFinish, GetAID().Name());
  Future<bool> f = Async(this->GetAID(), &KernelAsyncLaunchActor::OnTaskFinish);
  f.Wait();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End wait kernel launch finish";
}

Future<bool> KernelAsyncLaunchActor::OnTaskFinish() { return Future<bool>(true); }

void KernelAsyncLaunchActor::AddDeviceContext(DeviceContext *device_context) {
  (void)device_contexts_.insert(device_context);
}

void KernelAsyncLaunchActor::BindDevice() {
  std::for_each(device_contexts_.begin(), device_contexts_.end(),
                [](const DeviceContext *item) { item->device_res_manager_->BindDeviceToCurrentThread(false); });
}
}  // namespace runtime
}  // namespace mindspore
