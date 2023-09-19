/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/environ/environ_cpu_get.h"
#include "kernel/environ_manager.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
int EnvironGetCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  if (!EnvironMgr::GetInstance().CheckEnvInput(primitive_, inputs, outputs)) {
    MS_LOG(EXCEPTION) << "The input checks invalid, kernel: " << kernel_name_;
  }

  value_type_attr_ = TypeId(GetValue<int>(primitive_->GetAttr(kEnvValueTypeAttr)));
  MS_LOG(INFO) << "The EnvironGet kernel " << kernel_name_ << " value type: " << value_type_attr_;
  handle_size_ = sizeof(int64_t);
  key_size_ = sizeof(int64_t);

  auto value_type = outputs[kIndex0]->dtype_id();
  const auto &value_shapes = outputs[kIndex0]->GetShapeVector();
  auto default_value_type = inputs[kIndex2]->dtype_id();
  const auto &default_value_shapes = inputs[kIndex2]->GetShapeVector();
  if (IsDynamic(default_value_shapes)) {
    return KRET_UNKNOWN_SHAPE;
  }
  if (IsDynamic(value_shapes)) {
    return KRET_UNKNOWN_OUT_SHAPE;
  }
  if ((value_type != default_value_type) || (value_shapes != default_value_shapes)) {
    MS_LOG(EXCEPTION) << "The env value checks invalid, kernel: " << kernel_name_;
  }
  value_size_ = GetTypeByte(TypeIdToType(value_type));
  for (auto &i : value_shapes) {
    value_size_ *= static_cast<size_t>(i);
  }
  input_size_list_.clear();
  input_size_list_.push_back(handle_size_);
  input_size_list_.push_back(key_size_);
  input_size_list_.push_back(value_size_);
  output_size_list_.clear();
  output_size_list_.push_back(value_size_);
  return KRET_OK;
}

bool EnvironGetCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                    const std::vector<KernelTensor *> &outputs) {
  auto input_handle = GetDeviceAddress<int64_t>(inputs, kIndex0);
  auto input_key = GetDeviceAddress<int64_t>(inputs, kIndex1);
  auto input_default_value = GetDeviceAddress<void>(inputs, kIndex2);
  auto output_value = GetDeviceAddress<int64_t>(outputs, kIndex0);

  // Get host handle and host key.
  int64_t host_handle = input_handle[kIndex0];
  int64_t host_key = input_key[kIndex0];

  // Get env and value by handle and key.
  const auto &env = EnvironMgr::GetInstance().Get(host_handle);
  if (env == nullptr) {
    MS_LOG(EXCEPTION) << "Get the env failed, handle: " << host_handle << ", key: " << host_key;
  }
  MS_EXCEPTION_IF_NULL(env);
  const auto &env_value = env->Get(host_key);
  // Default value.
  auto value = input_default_value;
  auto value_size = inputs[kIndex2]->size();
  auto value_type = value_type_attr_;
  if (env_value != nullptr) {
    value = env_value->addr_;
    value_size = env_value->size_;
    value_type = env_value->value_type_;
  } else {
    MS_LOG(INFO) << "Use the default input value for kernel: " << kernel_name_ << ", env handle: " << host_handle
                 << ", env key: " << host_key;
  }

  // Check the env value size and type. The value size may be aligned, so must be greater then value_size_.
  if ((value_size < value_size_) || (value_type != value_type_attr_)) {
    MS_LOG(ERROR) << "The env value checks invalid, value_size: " << value_size << ", value_size_: " << value_size_
                  << ", value_type: " << value_type << ", value_type_attr_: " << value_type_attr_;
    return false;
  }

  auto ret = memcpy_s(output_value, value_size_, value, value_size_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Output memcpy error: " << ret;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EnvironGet, EnvironGetCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
