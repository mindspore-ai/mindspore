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

#ifndef MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_OP_PLUGIN_UTILS_H_
#define MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_OP_PLUGIN_UTILS_H_

#include <string>
#include <vector>
#include <unordered_set>
#include "kernel/cpu/op_plugin/custom_kernel_input_info_impl.h"
#include "kernel/cpu/utils/visible.h"

namespace mindspore::kernel {
namespace op_plugin {
struct OpPluginKernelParam {
  std::vector<void *> params;
  std::vector<int> ndims;
  std::vector<int64_t *> shapes;
  std::vector<std::string> dtype_strings;
  std::vector<const char *> dtypes;
  KernelInputInfoImpl kernel_info;
  void *stream{nullptr};
};

int LaunchOpPluginKernel(const std::string &op_name, size_t nparam, void **params, int *ndims, int64_t **shapes,
                         const char **type_pointer_list, void *kernel_info, void *stream = nullptr);
int LaunchOpPluginKernel(const std::string &op_name, OpPluginKernelParam *param);
OpPluginKernelParam CreateOpPluginParam(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs,
                                        const std::vector<KernelTensor *> &workspace);
}  // namespace op_plugin
}  // namespace mindspore::kernel

#endif  // MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_OP_PLUGIN_UTILS_H_
