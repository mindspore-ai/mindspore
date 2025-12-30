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

#include "kernel/cpu/op_plugin/custom_op_plugin_kernel.h"
#include "ops_utils/op_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/utils/callback.h"

namespace mindspore::kernel {
namespace op_plugin {
void RegisterOpPluginKernels() {
  const auto &op_names = ops::GetAllOpPluginKernelNames();
  for (const auto &op_name : op_names) {
    Factory<CustomOpPluginCpuKernelMod>::Instance().Register(
      op_name, []() { return std::make_shared<CustomOpPluginCpuKernelMod>(); });
    MS_LOG(INFO) << "Register op plugin kernel: " << op_name;
  }
}
REGISTER_COMMON_CALLBACK(RegisterOpPluginKernels);
}  // namespace op_plugin
}  // namespace mindspore::kernel
