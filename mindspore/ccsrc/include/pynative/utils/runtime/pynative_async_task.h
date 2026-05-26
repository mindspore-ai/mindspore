/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_PYNATIVE_ASYNC_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_PYNATIVE_ASYNC_TASK_H_

#include <cstdint>
#include <utility>

#include "include/pynative/utils/runtime/auto_grad_guard.h"
#include "include/utils/pynative/variable.h"
#include "include/utils/visible.h"
#include "runtime/pipeline/task/task.h"

namespace mindspore {
namespace pynative {

struct PYNATIVE_UTILS_EXPORT PyNativeThreadContext {
  bool grad_flag{false};
  bool enable_grad{true};
  uint32_t stream_id{0};
  kernel::pyboost::OpStatus op_status{};
  bool op_require_grad{false};
  bool is_safe_view{true};
  autograd::AutoDiffInterfacePtr auto_diff_engine{nullptr};
};

PYNATIVE_UTILS_EXPORT PyNativeThreadContext CapturePyNativeThreadContext();
PYNATIVE_UTILS_EXPORT void RestorePyNativeThreadContext(const PyNativeThreadContext &ctx);

class PYNATIVE_UTILS_EXPORT PyNativeThreadContextGuard {
 public:
  explicit PyNativeThreadContextGuard(const PyNativeThreadContext &ctx);
  ~PyNativeThreadContextGuard();

 private:
  PyNativeThreadContext old_ctx_;
};

class PYNATIVE_UTILS_EXPORT PyNativeAsyncTask : public runtime::AsyncTask {
 public:
  explicit PyNativeAsyncTask(runtime::TaskType task_type)
      : runtime::AsyncTask(task_type), ctx_(CapturePyNativeThreadContext()) {}
  ~PyNativeAsyncTask() override = default;

 protected:
  template <typename Fn>
  void RunWithPyNativeContext(Fn &&fn) {
    PyNativeThreadContextGuard guard(ctx_);
    std::forward<Fn>(fn)();
  }

 private:
  PyNativeThreadContext ctx_;
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_PYNATIVE_ASYNC_TASK_H_
