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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_AOT_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_AOT_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "utils/custom_aot_extra.h"

namespace mindspore {
namespace kernel {

class CustomAOTCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  CustomAOTCpuKernelMod() : num_input_(0), num_output_(0), handle_(nullptr), init_func_(nullptr), aot_func_(nullptr) {}
  ~CustomAOTCpuKernelMod();

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  void InitSizeLists();
  void SetKernelAttr(const CNodePtr &kernel_node);

  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<int> ndims_;
  std::vector<std::string> type_list_;

  std::vector<int64_t *> shapes_;
  std::vector<const char *> type_pointer_list_;

  size_t num_input_;
  size_t num_output_;
  std::string file_path_;
  std::string func_name_;
  void *handle_;
  int (*init_func_)(int *, int64_t **, const char **, AotExtra *);
  int (*aot_func_)(int, void **, int *, int64_t **, const char **, void *, void *);

  AotExtraImpl attrs_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_AOT_KERNEL_H_
