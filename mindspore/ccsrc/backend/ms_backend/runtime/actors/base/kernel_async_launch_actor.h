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

#ifndef MINDSPORE_CCSRC_RUNTIME_CORE_ACTORS_BASE_KERNEL_ASYNC_LAUNCH_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_CORE_ACTORS_BASE_KERNEL_ASYNC_LAUNCH_ACTOR_H_

#include <set>
#include <vector>
#include <memory>

#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
class KernelActor;
class KernelRunner;

class BACKEND_EXPORT KernelAsyncLaunchActor : public ActorBase {
 public:
  static std::shared_ptr<KernelAsyncLaunchActor> &GetInstance();
  ~KernelAsyncLaunchActor() override = default;

  void Initialize();

  void LaunchKernel(OpContext<KernelTensor> *const context, KernelActor *kernel_actor);
  void LaunchKernelV2(OpContext<KernelTensor> *const context, KernelRunner *kernel_runner);
  void LaunchKernelV2HP(OpContext<KernelTensor> *const context, KernelRunner *kernel_runner);

  void Wait();

  Future<bool> OnTaskFinish();

  const std::thread::id &actor_thread_id() const { return thread_id_; }

  void AddDeviceContext(DeviceContext *device_context);

  void BindDevice();

 private:
  KernelAsyncLaunchActor() : ActorBase("KernelAsyncLaunchActor") {}
  DISABLE_COPY_AND_ASSIGN(KernelAsyncLaunchActor);

  void GetThreadId() { thread_id_ = std::this_thread::get_id(); }

  // The thread id of exclusive thread used by this actor.
  std::thread::id thread_id_;

  std::set<DeviceContext *> device_contexts_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_CORE_ACTORS_BASE_KERNEL_ASYNC_LAUNCH_ACTOR_H_
