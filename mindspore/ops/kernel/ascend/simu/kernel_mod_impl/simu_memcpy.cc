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

#include "kernel/ascend/simu/kernel_mod_impl/simu_memcpy.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
bool SimuMemcpyKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Invalid simu " << op_name_ << " input, output size (" << inputs.size() << ", " << outputs.size()
                  << ").";
    return false;
  }

  auto input_size = inputs[0]->size();
  auto output_size = outputs[0]->size();
  auto size = input_size > output_size ? output_size : input_size;

  size_t offset = 0;
  auto device_ptr = static_cast<int8_t *>(outputs[0]->device_ptr());
  while (offset + size <= output_size) {
    auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, device_ptr + offset, size, inputs[0]->device_ptr(), size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
    if (cp_ret != EOK) {
      MS_LOG(ERROR) << "Simu " << op_name_ << "  aclrtMemcpy failed.";
      return false;
    }
    offset += size;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
