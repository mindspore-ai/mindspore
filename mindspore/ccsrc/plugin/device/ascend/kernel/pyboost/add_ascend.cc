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

#include "plugin/device/ascend/kernel/pyboost/add_ascend.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "transform/acl_ir/op_api_exec.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
bool AddAscend::Launch(const tensor::TensorPtr &x, const tensor::TensorPtr &y, const tensor::TensorPtr &output) {
  auto device_context = PyBoostUtils::GetDeviceContext(kAscendDevice);

  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, x, "x");
  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, y, "y");
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, output, "output");

  // 910A not support 0
  int8_t cube_math_type = 0;
  auto [workspace_size, executor, after_launch_func] = GEN_EXECUTOR(aclnnAdd, x, y, output, cube_math_type);

  auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
  if (workspace_size == 0) {
    RUN_OP_API(aclnnAdd, stream_ptr, nullptr, 0, executor, after_launch_func);
  } else {
    auto workspace_device_address = runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context, workspace_size);
    RUN_OP_API(aclnnAdd, stream_ptr, workspace_device_address->GetMutablePtr(), workspace_size, executor,
               after_launch_func);
  }

  return true;
}

tensor::TensorPtr AddAscend::Call(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  InferOutput(x, y);
  Launch(x, y, output(0));
  return output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
