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

#ifndef MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_OP_PLUGIN_CPU_KERNEL_H_
#define MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_OP_PLUGIN_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include "kernel/cpu/op_plugin/custom_kernel_input_info_impl.h"
#include "kernel/cpu/cpu_kernel.h"
#include "kernel/cpu/utils/visible.h"

namespace mindspore {
namespace kernel {
namespace op_plugin {
class OPS_HOST_API CustomOpPluginCpuKernelMod : public NativeCpuKernelMod {
 public:
  CustomOpPluginCpuKernelMod() = default;
  ~CustomOpPluginCpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool IsNeedUpdateOutputShapeAndSize() override { return is_compute_depend_op_; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<int> ndims_;
  std::vector<std::string> type_list_;

  std::vector<int64_t *> shapes_;
  std::vector<const char *> type_pointer_list_;

  std::string file_path_;
  std::string func_name_;

  KernelInputInfoImpl kernel_info_;

  bool is_compute_depend_op_{false};

 private:
  void SetKernelPath();
};
}  // namespace op_plugin
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_OP_PLUGIN_CPU_KERNEL_H_
