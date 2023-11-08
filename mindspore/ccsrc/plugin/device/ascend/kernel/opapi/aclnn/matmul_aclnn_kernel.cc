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
#include "plugin/device/ascend/kernel/opapi/aclnn/matmul_aclnn_kernel.h"
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
#include "plugin/device/ascend/hal/hardware/ge_device_context.h"

namespace mindspore {
namespace kernel {

bool MMAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

#if 0
  auto input_device = std::make_shared<device::ascend::AscendDeviceAddress>(
    inputs[0]->addr, inputs[0]->size, kOpFormat_DEFAULT, input_params_[0].data_type);
  auto input_device2 = std::make_shared<device::ascend::AscendDeviceAddress>(
    inputs[1]->addr, inputs[1]->size, kOpFormat_DEFAULT, input_params_[1].data_type);
  input_device->set_host_shape(input_params_[0].ori_shape);
  input_device2->set_host_shape(input_params_[1].ori_shape);
  auto output_device = std::make_shared<device::ascend::AscendDeviceAddress>(
    outputs[0]->addr, outputs[0]->size, kOpFormat_DEFAULT, output_params_[0].data_type);
  output_device->set_host_shape(output_params_[0].ori_shape);

  ParseGenExecutor(GEN_EXECUTOR(aclnnMatmul, input_device, input_device2, output_device, OpApiUtil::GetCubeMathType()));

  if (workspace_size_list_.empty()) {
    RUN_OP_API(aclnnMatmul, stream_ptr, nullptr, 0, executor_, after_launch_func_);
    return true;
  }

  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto res_manager = dynamic_cast<device::ascend::GeDeviceResManager *>(device_context->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager);
  auto workspaces_addr = res_manager->AllocateMemory(workspace_size_list_[0]);

  RUN_OP_API(aclnnMatmul, stream_ptr, workspaces_addr, workspace_size_list_[0], executor_, after_launch_func_);
#endif
  return true;
}
}  // namespace kernel
}  // namespace mindspore
