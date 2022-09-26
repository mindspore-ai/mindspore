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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_JULIA_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_JULIA_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
class CustomJULIACpuKernelMod : public NativeCpuKernelMod {
 public:
  CustomJULIACpuKernelMod() = default;
  ~CustomJULIACpuKernelMod() = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  ShapeArray shape_list_;
  std::vector<size_t> ndims_;
  std::vector<std::string> type_list_;

  std::vector<ShapeValueDType *> shapes_;
  std::vector<const char *> type_pointer_list_;

  std::string file_path_;
  std::string module_name_;
  std::string func_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_JULIA_KERNEL_H_
