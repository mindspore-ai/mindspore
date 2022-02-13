/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "kernel/ascend_kernel_mod.h"
#include "runtime/rt.h"
namespace mindspore {
namespace kernel {
void AscendKernelMod::UpdateOp() {
  MS_EXCEPTION_IF_NULL(stream_);
  // cppcheck-suppress unreadVariable
  auto lock = LockRuntime();
  if (RT_ERROR_NONE != rtStreamSynchronize(stream_)) {
    MS_LOG(EXCEPTION) << "Call runtime rtStreamSynchronize failed.";
  }
}

std::lock_guard<std::mutex> AscendKernelMod::LockRuntime() {
  static std::mutex mutex;
  return std::lock_guard<std::mutex>(mutex);
}
}  // namespace kernel
}  // namespace mindspore
