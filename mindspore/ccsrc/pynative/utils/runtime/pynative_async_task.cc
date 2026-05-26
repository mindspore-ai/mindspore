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

#include "include/pynative/utils/runtime/pynative_async_task.h"

#include <utility>

#include "include/utils/pynative/grad_state.h"
#include "utils/current_state.h"

namespace mindspore {
namespace pynative {
PyNativeThreadContext CapturePyNativeThreadContext() {
  PyNativeThreadContext ctx;
  auto &grad_state = GradState::Get();
  ctx.grad_flag = grad_state.grad_flag();
  ctx.enable_grad = grad_state.enable_grad();
  ctx.stream_id = CurrentStream::id();

  auto &op_run_status = kernel::pyboost::OpRunStatus::Get();
  ctx.op_status = op_run_status.op_status();
  ctx.op_require_grad = op_run_status.RequireGrad();
  ctx.is_safe_view = op_run_status.IsSafeView();

  ctx.auto_diff_engine = autograd::impl::CurrentAutoDiffEngine();
  return ctx;
}

void RestorePyNativeThreadContext(const PyNativeThreadContext &ctx) {
  auto &grad_state = GradState::Get();
  grad_state.set_grad_flag(ctx.grad_flag);
  grad_state.set_enable_grad(ctx.enable_grad);
  CurrentStream::set_id(ctx.stream_id);

  auto &op_run_status = kernel::pyboost::OpRunStatus::Get();
  auto op_status = ctx.op_status;
  op_run_status.set_run_info(std::move(op_status));
  op_run_status.SetRequireGrad(ctx.op_require_grad);
  op_run_status.SetIsSafeView(ctx.is_safe_view);

  autograd::impl::SetCurrentAutoDiffEngine(ctx.auto_diff_engine);
}

PyNativeThreadContextGuard::PyNativeThreadContextGuard(const PyNativeThreadContext &ctx)
    : old_ctx_(CapturePyNativeThreadContext()) {
  RestorePyNativeThreadContext(ctx);
}

PyNativeThreadContextGuard::~PyNativeThreadContextGuard() { RestorePyNativeThreadContext(old_ctx_); }
}  // namespace pynative
}  // namespace mindspore
