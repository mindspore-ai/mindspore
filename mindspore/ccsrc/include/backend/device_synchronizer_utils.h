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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_DEVICE_SYNCHRONIZER_UTILS_H
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_DEVICE_SYNCHRONIZER_UTILS_H

#ifndef BUILD_LITE
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#endif

namespace mindspore {
static inline void WaitAsyncResizeAndLaunchFinish() {
#ifndef BUILD_LITE
  if (runtime::ActorDispatcher::enable_runtime_multi_pipeline()) {
    runtime::KernelAsyncResizeActor::GetInstance()->Wait();
  }

  if (runtime::ActorDispatcher::enable_async_launch_kernel()) {
    runtime::KernelAsyncLaunchActor::GetInstance()->Wait();
  }
#endif
}
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_DEVICE_SYNCHRONIZER_UTILS_H
