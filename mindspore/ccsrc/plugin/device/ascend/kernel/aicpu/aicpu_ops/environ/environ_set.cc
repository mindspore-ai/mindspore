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

#include "environ/environ_set.h"
#include <string>
#include <memory>
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/aicpu_sharder/aicpu_sharder.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/tensor.h"
#include "environ/aicpu_environ_manager.h"

namespace aicpu {
constexpr auto kEnvValueTypeAttr = "value_type";

uint32_t EnvironSetKernel::DoCompute() {
  AICPU_LOGD("Enter DoCompute.");
  auto &env_mgr = EnvironMgr::GetInstance();

  auto *input_handle_ptr = reinterpret_cast<int64_t *>(io_addrs_[aicpu::kIndex0]);
  auto *input_key_ptr = reinterpret_cast<int64_t *>(io_addrs_[aicpu::kIndex1]);
  auto *input_value_ptr = reinterpret_cast<void *>(io_addrs_[aicpu::kIndex2]);
  auto *output_handle_ptr = reinterpret_cast<int64_t *>(io_addrs_[aicpu::kIndex3]);

  auto *value_ptr = malloc(value_size_);
  AICPU_CHECK_NULLPTR(value_ptr, kAicpuKernelStateInvalid, "Malloc failed.")
  auto ret = memcpy_s(value_ptr, value_size_, input_value_ptr, value_size_);
  AICPU_CHECK_FALSE((ret == EOK), kAicpuKernelStateInvalid, "Memcpy size from input[2] to environ failed.",
                    value_size_);

  // Set env member.
  const auto &env = env_mgr.Get(input_handle_ptr[0]);
  AICPU_CHECK_NULLPTR(env, kAicpuKernelStateInvalid, "Get handle[%d] failed.", input_handle_ptr[0]);

  auto env_value = std::make_shared<EnvironValue>(value_ptr, value_size_, attr_value_type_);
  env->Set(input_key_ptr[0], env_value);
  AICPU_LOGD("EnvironSetKernel: handle[%d], key[%d], value[%d]", input_handle_ptr[0], input_key_ptr[0],
             (void *)&env_value);

  // Set output handle
  output_handle_ptr[0] = input_handle_ptr[0];
  return kAicpuKernelStateSucess;
}

uint32_t EnvironSetKernel::ParseKernelParam() {
  AICPU_LOGD("Enter ParseKernelParam.");
  auto &env_mgr = EnvironMgr::GetInstance();
  if (!env_mgr.CheckEnvInput(node_def_)) {
    AICPU_LOGE("The input checks invalid. ");
    return kAicpuKernelStateInvalid;
  }

  if (!env_mgr.IsScalarTensor(node_def_.outputs(aicpu::kIndex0))) {
    AICPU_LOGE("The output handle is not equal of input handle.");
    return kAicpuKernelStateInvalid;
  }

  // Get value type.
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> nodedef_map = node_def_.attrs();
  attr_value_type_ = nodedef_map[kEnvValueTypeAttr].i();

  // Get value size.
  aicpuops::Tensor value_tensor = node_def_.inputs(aicpu::kIndex2);
  value_size_ = value_tensor.data_size();
  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t EnvironSet(void *param) {
  aicpu::EnvironSetKernel environSetKernel;
  return environSetKernel.Compute(param);
}
}
