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
#include "plugin/device/ascend/kernel/opapi/aclnn/abs_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;

bool AbsAclnnKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto input_device = std::make_shared<device::ascend::AscendDeviceAddress>(
    inputs[kInputIndex]->addr, inputs[kInputIndex]->size, kOpFormat_DEFAULT, input_params_[kInputIndex].data_type);
  input_device->set_host_shape(input_params_[kInputIndex].ori_shape);
  auto output_device =
    std::make_shared<device::ascend::AscendDeviceAddress>(outputs[kOutputIndex]->addr, outputs[kOutputIndex]->size,
                                                          kOpFormat_DEFAULT, output_params_[kOutputIndex].data_type);
  output_device->set_host_shape(output_params_[kOutputIndex].ori_shape);

  // TODO(ruige): Move to build and resize.
  ParseGenExecutor(GEN_EXECUTOR(aclnnAbs, input_device, output_device));

  if (workspace_size_list_.empty()) {
    RUN_OP_API(aclnnAbs, stream_ptr, nullptr, 0, executor_, after_launch_func_);
    return true;
  }
  RUN_OP_API(aclnnAbs, stream_ptr, workspaces[0]->addr, workspace_size_list_[0], executor_, after_launch_func_);
  return true;
}
MS_ACLLNN_KERNEL_FACTORY_REG(Abs, AbsAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
