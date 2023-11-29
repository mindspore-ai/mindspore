/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/host/reshape_kernel.h"

#include <algorithm>
#include <functional>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "abstract/utils.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/check_convert_utils.h"
#include "utils/trace_base.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace kernel {
bool ReshapeKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  auto status = aclrtMemcpyAsync(outputs[0]->device_ptr(), outputs[0]->size(), inputs[0]->device_ptr(),
                                 inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "ReshapeKernelMod Launch failed. kernel: " << kernel_name_
                  << ", call rtMemcpyAsync failed, ret = 0x" << status;
    return false;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
