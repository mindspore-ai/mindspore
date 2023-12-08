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

#include "plugin/device/gpu/kernel/pyboost/customize/copy.h"
#include "kernel/pyboost/customize/op_common.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr CopyGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Call start";
  // No need to get default_stream here, after the multi-stream feature is complete.
  auto stream = device::gpu::GPUDeviceManager::GetInstance().default_stream();
  return CopyCustomizeCall(op, input_tensor, stream);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
