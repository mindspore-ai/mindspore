/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_EXECUTOR_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_EXECUTOR_H_

#include <vector>
#include <memory>
#include <queue>
#include <map>
#include <string>
#include <set>
#include <utility>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/utils/visible.h"
#include "include/pynative/utils/runtime/task/device_task.h"
#include "include/runtime/pipeline/async_rqueue.h"

namespace mindspore {
using Tensor = tensor::Tensor;
using TensorPtr = tensor::TensorPtr;
namespace runtime {
class PYNATIVE_UTILS_EXPORT OpExecutor {
 public:
  static OpExecutor &GetInstance();

  void PushOpRunTask(const std::shared_ptr<DeviceOpRunTask> &op_run_task);

  void PushOpRunTask(const std::shared_ptr<PyBoostDeviceTask> &op_run_task);

  void PushSimpleOpRunTask(const std::shared_ptr<AsyncTask> &op_run_task);

  bool RunQueueEmpty();

  // When an exception occurs, the state needs to be reset.
  // Because we may sometimes have to ignore the exception and continue to run other tasks
  void Reset();

  // Thread join before the process exit.
  void WorkerJoin();

  // Child process reinitialize resource after fork.
  void ChildAfterFork();

  void set_async_for_graph(bool flag) { async_for_graph_ = flag; }

  bool async_for_graph() const { return async_for_graph_; }

  static bool NeedSync();

  void RegisterCallbackForMemoryPool();

  static void DispatchLaunchTask(std::function<void()> &&func);

 private:
  OpExecutor();
  ~OpExecutor();
  DISABLE_COPY_AND_ASSIGN(OpExecutor);

  std::function<void()> forward_callback_{nullptr};
  inline static bool async_for_graph_{false};
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_EXECUTOR_H_
