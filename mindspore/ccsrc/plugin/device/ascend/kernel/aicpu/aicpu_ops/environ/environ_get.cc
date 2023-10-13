/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "environ/environ_get.h"
#include <random>
#include <climits>
#include <vector>
#include <algorithm>
#include <string>
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/aicpu_sharder/aicpu_sharder.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/tensor.h"
#include "environ/aicpu_environ_manager.h"

namespace aicpu {
constexpr auto kEnvValueTypeAttr = "value_type";

uint32_t EnvironGetKernel::DoCompute() {
  AICPU_LOGD("Enter DoCompute.");
  auto &env_mgr = EnvironMgr::GetInstance();

  auto *input_handle_ptr = reinterpret_cast<int64_t *>((io_addrs_[aicpu::kIndex0]));
  auto *input_key_ptr = reinterpret_cast<int64_t *>((io_addrs_[aicpu::kIndex1]));
  auto *default_value_ptr = reinterpret_cast<void *>((io_addrs_[aicpu::kIndex2]));
  auto *output_ptr = reinterpret_cast<void *>((io_addrs_[aicpu::kIndex3]));

  // Get handle and key
  int64_t handle = input_handle_ptr[0];
  int64_t key = input_key_ptr[0];

  // Get env and value by handle and key
  const auto &env = env_mgr.Get(handle);
  AICPU_CHECK_NULLPTR(env, kAicpuKernelStateInvalid, "Get env [%d] failed", handle)
  const auto &env_value = env->Get(key);

  AICPU_LOGD("EnvironGetKernel: hindle[%d], key[%d], value[%d]", handle, key, (void *)&env_value);
  // Default value
  auto *output_value_ptr = default_value_ptr;
  auto output_value_size = default_value_size_;
  auto output_value_type = attr_value_type_;
  if (env_value != nullptr) {
    output_value_ptr = env_value->addr_;
    output_value_size = env_value->size_;
    output_value_type = env_value->value_type_;
  } else {
    AICPU_LOGE("Get key[%d] value checks failed.", key);
  }

  if ((output_value_size_ < output_value_size) || (output_value_type != attr_value_type_)) {
    AICPU_LOGE("The env value checks invalid, value_size: %d vs %d, value_type:%d vs %d", output_value_size_,
               output_value_size, output_value_type, attr_value_type_);
    return kAicpuKernelStateInvalid;
  }

  auto ret = memcpy_s(output_ptr, output_value_size_, output_value_ptr, output_value_size_);
  AICPU_CHECK_FALSE((ret == EOK), kAicpuKernelStateInvalid, "Memcpy size[%zu] from env map to output[0] failed.",
                    output_value_size_);

  return kAicpuKernelStateSucess;
}

uint32_t EnvironGetKernel::ParseKernelParam() {
  AICPU_LOGD("Enter ParseKernelParam.");
  auto &env_mgr = EnvironMgr::GetInstance();
  if (!env_mgr.CheckEnvInput(node_def_)) {
    AICPU_LOGE("The input checks invalid. ");
    return kAicpuKernelStateInvalid;
  }

  // Get value type attr
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> nodedef_map = node_def_.attrs();
  attr_value_type_ = nodedef_map[kEnvValueTypeAttr].i();

  // check output value
  auto default_value_tensor = node_def_.inputs(aicpu::kIndex2);
  auto output_value_ptr_tensor = node_def_.outputs(aicpu::kIndex0);
  if ((output_value_ptr_tensor.tensor_shape().dim_size() != default_value_tensor.tensor_shape().dim_size()) ||
      (output_value_ptr_tensor.tensor_type() != default_value_tensor.tensor_type())) {
    AICPU_LOGE("The env value checks invalid.");
    return kAicpuKernelStateInvalid;
  }

  // Get value size.
  default_value_size_ = GetTensorMemSizeByShape(default_value_tensor);
  output_value_size_ = GetTensorMemSizeByShape(output_value_ptr_tensor);
  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t EnvironGet(void *param) {
  aicpu::EnvironGetKernel environGetKernel;
  return environGetKernel.Compute(param);
}
}
