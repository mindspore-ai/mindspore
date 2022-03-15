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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_HELPER_BASE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_HELPER_BASE_H_

#include <string>
#include <vector>
#include "mindspore/core/utils/log_adapter.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/cuda_class_common.h"
namespace mindspore {
namespace cukernel {
struct GpuKernelAttrBase {
  virtual ~GpuKernelAttrBase() = default;
};

class GpuKernelHelperBase {
 public:
  explicit GpuKernelHelperBase(std::string &kernel_name) : kernel_name_(kernel_name) {}
  virtual ~GpuKernelHelperBase() {
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

  virtual int CalMemSize(const std::vector<std::vector<size_t>> &input_shapes,
                         const std::vector<std::vector<size_t>> &output_shapes) = 0;

  virtual int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
                      const std::vector<void *> &work_ptrs, void *cuda_stream) = 0;

  virtual void ResetResource() {
    MS_LOG(ERROR) << "kernel must override the `ResetResource()` method when dynamic shape";
  }

  std::vector<size_t> GetInputSizeList() { return input_size_list_; }
  std::vector<size_t> GetOutputSizeList() { return output_size_list_; }
  std::vector<size_t> GetWorkSizeList() { return work_size_list_; }

  virtual int CheckKernelParam(GpuKernelAttrBase *kernel_attr) { return 0; }

 protected:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> work_size_list_;
  std::string kernel_name_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_HELPER_BASE_H_
