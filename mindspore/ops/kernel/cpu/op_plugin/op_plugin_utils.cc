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

#include "kernel/cpu/op_plugin/op_plugin_utils.h"
#include <unordered_set>
#include <string>
#include <algorithm>
#include "ops_utils/dl_utils.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"

namespace mindspore::kernel {
namespace op_plugin {
OpPluginKernelParam CreateOpPluginParam(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs,
                                        const std::vector<KernelTensor *> &workspace) {
  OpPluginKernelParam param;

  // Process inputs
  for (const auto &input : inputs) {
    param.params.push_back(input->device_ptr());
    const auto &in_shape = input->GetShapeVector();
    param.shapes.push_back(const_cast<int64_t *>(in_shape.data()));
    param.ndims.push_back(SizeToInt(in_shape.size()));
    param.dtype_strings.push_back(TypeIdToString(input->dtype_id(), true));
  }

  // Process outputs
  for (const auto &output : outputs) {
    param.params.push_back(output->device_ptr());
    const auto &out_shape = output->GetShapeVector();
    param.shapes.push_back(const_cast<int64_t *>(out_shape.data()));
    param.ndims.push_back(SizeToInt(out_shape.size()));
    param.dtype_strings.push_back(TypeIdToString(output->dtype_id(), true));
  }

  // Process workspace
  std::transform(workspace.begin(), workspace.end(), std::back_inserter(param.params),
                 [](const KernelTensor *ws) { return ws->device_ptr(); });

  param.dtypes.reserve(param.dtype_strings.size());
  std::transform(param.dtype_strings.begin(), param.dtype_strings.end(), std::back_inserter(param.dtypes),
                 [](const std::string &dtype_str) { return dtype_str.c_str(); });

  param.kernel_info.SetKernelInput(inputs);
  return param;
}

int LaunchOpPluginKernel(const std::string &op_name, size_t nparam, void **params, int *ndims, int64_t **shapes,
                         const char **dtypes, void *kernel_info, void *stream) {
  int (*op_plugin_func)(int, void **, int *, int64_t **, const char **, void *, void *) = nullptr;
  void *handle = ops::GetOpPluginHandle();
  if (handle == nullptr) {
    MS_LOG(ERROR) << "Op plugin handle is not initialized. Please ensure MS_OP_PLUGIN_PATH is set correctly.";
    return -1;
  }

  // Clear previous errors before dlsym
  (void)DL_ERROR();
#ifdef _WIN32
  SetLastError(0);
#endif
  op_plugin_func =
    reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
      DL_SYM(handle, op_name.c_str()));
  if (auto error_info = DL_ERROR(); error_info != nullptr) {
    MS_LOG(ERROR) << "Failed to load op plugin kernel function for '" << op_name << "'. Error info: " << error_info;
    return -1;
  }

  return op_plugin_func(nparam, params, ndims, shapes, dtypes, stream, kernel_info);
}

int LaunchOpPluginKernel(const std::string &op_name, OpPluginKernelParam *param) {
  return LaunchOpPluginKernel(op_name, param->params.size(), param->params.data(), param->ndims.data(),
                              param->shapes.data(), param->dtypes.data(), reinterpret_cast<void *>(&param->kernel_info),
                              param->stream);
}
}  // namespace op_plugin
}  // namespace mindspore::kernel
