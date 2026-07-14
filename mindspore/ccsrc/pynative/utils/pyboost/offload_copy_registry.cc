/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/offload_copy_registry.h"

#include <utility>

#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

OffloadCopyRegistry &OffloadCopyRegistry::GetInstance() {
  static OffloadCopyRegistry instance;
  return instance;
}

void OffloadCopyRegistry::RegisterCopyToHost(CopyToHostFunc func) { copy_to_host_func_ = std::move(func); }

void OffloadCopyRegistry::RegisterCopyToDevice(CopyToDeviceFunc func) { copy_to_device_func_ = std::move(func); }

bool OffloadCopyRegistry::HasCopyToHost() const { return copy_to_host_func_ != nullptr; }

bool OffloadCopyRegistry::HasCopyToDevice() const { return copy_to_device_func_ != nullptr; }

tensor::TensorPtr OffloadCopyRegistry::CopyToHost(size_t stream_id, const tensor::TensorPtr &device_tensor) {
  if (!copy_to_host_func_) {
    MS_LOG(EXCEPTION) << "CopyToHost function not registered. Is mindspore_ops_ascend loaded?";
  }
  return copy_to_host_func_(stream_id, device_tensor);
}

tensor::TensorPtr OffloadCopyRegistry::CopyToDevice(size_t stream_id, const tensor::TensorPtr &host_tensor) {
  if (!copy_to_device_func_) {
    MS_LOG(EXCEPTION) << "CopyToDevice function not registered. Is mindspore_ops_ascend loaded?";
  }
  return copy_to_device_func_(stream_id, host_tensor);
}

OffloadCopyRegistry::OffloadCopyRegistry() = default;

OffloadCopyRegistry::~OffloadCopyRegistry() = default;

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
