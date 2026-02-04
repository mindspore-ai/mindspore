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

#include "kernel/ascend/atb/kernel_mod_impl/add_atb_kernel.h"
#include <vector>

namespace mindspore::kernel {
void AddATBKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
  uint64_t hash_id = device::ascend::AtbHash(param, op_name_);
  if (hash_id != hash_id_) {
    atb::CreateOperation(param, &op_);
    hash_id_ = hash_id;
  }
  param_setter_.SetIndex({0, 1}, {0}).Input(inputs[0]).Input(inputs[1]).Output(outputs[0]);
  UpdateWorkspace(device::ascend::GetWorkSpaceSize(op_, param_setter_.variant_pack, param_setter_.stream));
}

bool AddATBKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  param_setter_.Update(inputs, outputs);
  device::ascend::Launch(op_, param_setter_.variant_pack, workspace[0]->device_ptr(), workspace_size_list_, stream_ptr);
  return true;
}
}  // namespace mindspore::kernel
