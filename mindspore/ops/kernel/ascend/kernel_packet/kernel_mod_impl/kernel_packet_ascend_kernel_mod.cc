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

#include "kernel/ascend/kernel_packet/kernel_mod_impl/kernel_packet_ascend_kernel_mod.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
bool KernelPacketAscendKernelMod::CopyHostToDevice(void *dst, const void *src, size_t size, void *stream) {
  aclError status = CALL_ASCEND_API(aclrtMemcpyAsync, dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
