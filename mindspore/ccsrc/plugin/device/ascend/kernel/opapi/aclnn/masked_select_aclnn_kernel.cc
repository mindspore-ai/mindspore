/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/masked_select_aclnn_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {
bool MaskedSelectAclnnKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                        const std::vector<AddressPtr> &workspaces,
                                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto input_device = std::make_shared<device::ascend::AscendDeviceAddress>(
    inputs[kIndex0]->addr, inputs[kIndex0]->size, kOpFormat_DEFAULT, input_params_[kIndex0].data_type);
  input_device->set_host_shape(input_params_[kIndex0].ori_shape);
  auto mask_device = std::make_shared<device::ascend::AscendDeviceAddress>(
    inputs[kIndex1]->addr, inputs[kIndex1]->size, kOpFormat_DEFAULT, input_params_[kIndex1].data_type);
  mask_device->set_host_shape(input_params_[kIndex1].ori_shape);
  auto output_device = std::make_shared<device::ascend::AscendDeviceAddress>(
    outputs[kIndex0]->addr, outputs[kIndex0]->size, kOpFormat_DEFAULT, output_params_[kIndex0].data_type);
  output_device->set_host_shape(output_params_[kIndex0].ori_shape);

  // TODO(ruige): Move to build and resize.
  auto [workspace_size, executor_, tensor_param] =
    GEN_EXECUTOR_CUSTOM(aclnnMaskedSelect, input_device, mask_device, output_device);
  if (workspace_size != 0) {
    std::vector<size_t> workspace_size_list = {workspace_size};
    SetWorkspaceSizeList(workspace_size_list);
  }

  if (workspace_size_list_.empty()) {
    RUN_OP_API_SYNC(aclnnMaskedSelect, stream_ptr, nullptr, 0, executor_);
  } else {
    RUN_OP_API_SYNC(aclnnMaskedSelect, stream_ptr, workspaces[0]->addr, workspace_size_list_[0], executor_);
  }

  // Update output shape.
  outputs_[0]->SetShapeVector(transform::UpdateOutputShape(tensor_param.get<2>()));
  return true;
}
}  // namespace kernel
}  // namespace mindspore
