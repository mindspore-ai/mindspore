/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/utils/callback.h"
#include "include/cluster/init.h"
#include "include/backend/common/device_address_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_infer_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_resize_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_async_launch_actor.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "tools/profiler/profiler.h"

namespace mindspore {
namespace backend {
namespace ms_backend {
namespace {
const auto kSyncDataFromDeviceToHost = "SyncDataFromDeviceToHost";
// The runtime pipeline: InferShape->ResizeKernelMod->LaunchKernel, the latter cannot wait for the former, otherwise
// deadlock may occur.
// 1. infer shape task needs to wait for resize and kernel launch.
// 2. Internally, the resize task only needs to wait for the kernel launch
void WaitAsyncResizeAndLaunchFinish() {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  if (runtime::ActorDispatcher::enable_runtime_multi_pipeline()) {
    const auto &cur_thread_id = std::this_thread::get_id();
    if (runtime::EnableRuntimeNewPipeline()) {
      if (cur_thread_id != runtime::RuntimePipeline::GetInstance().resize_queue()->thread_id() &&
          cur_thread_id != runtime::RuntimePipeline::GetInstance().launch_queue()->thread_id()) {
        runtime::RuntimePipeline::GetInstance().infer_queue()->Wait();
      }

      if (cur_thread_id != runtime::RuntimePipeline::GetInstance().launch_queue()->thread_id()) {
        runtime::RuntimePipeline::GetInstance().resize_queue()->Wait();
      }
    } else {
      if (cur_thread_id != runtime::KernelAsyncResizeActor::GetInstance()->actor_thread_id() &&
          cur_thread_id != runtime::KernelAsyncLaunchActor::GetInstance()->actor_thread_id()) {
        runtime::KernelAsyncInferActor::GetInstance()->Wait();
      }

      if (cur_thread_id != runtime::KernelAsyncLaunchActor::GetInstance()->actor_thread_id()) {
        runtime::KernelAsyncResizeActor::GetInstance()->Wait();
      }
    }
  }

  if (runtime::ActorDispatcher::enable_async_launch_kernel()) {
    if (runtime::EnableRuntimeNewPipeline()) {
      runtime::RuntimePipeline::GetInstance().launch_queue()->Wait();
    } else {
      runtime::KernelAsyncLaunchActor::GetInstance()->Wait();
    }
  }

  PROFILER_END(start_time, runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kWaitTaskFinish,
               kSyncDataFromDeviceToHost, false);
}
}  // namespace

// Register a wait callback to kernel::KernelTensor, used to wait runtime async kernel launch task finish when get value
// from device side.
REGISTER_COMMON_CALLBACK(WaitAsyncResizeAndLaunchFinish);
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
