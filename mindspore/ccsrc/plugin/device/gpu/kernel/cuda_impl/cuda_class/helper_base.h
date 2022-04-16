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
#include <memory>
#include "mindspore/core/utils/log_adapter.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/cuda_class_common.h"
#include "ir/dtype/type_id.h"
#include "include/api/format.h"
namespace mindspore {
namespace cukernel {
class GpuKernelAttrBase {
 public:
  GpuKernelAttrBase() = default;
  virtual ~GpuKernelAttrBase() = default;
};

using GpuKernelAttrBasePtr = std::shared_ptr<GpuKernelAttrBase>;

struct TensorInfo {
  std::vector<std::vector<int>> shapes;
  std::vector<std::vector<TypeId>> types;
  std::vector<std::vector<Format>> formats;
};

class GpuKernelHelperBase {
 public:
  explicit GpuKernelHelperBase(const std::string &kernel_name, const uint32_t &device_id)
      : kernel_name_(kernel_name), device_id_(device_id) {}
  virtual ~GpuKernelHelperBase() {
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

  virtual int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                         const std::vector<std::vector<int64_t>> &output_shapes) = 0;

  virtual int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
                      const std::vector<void *> &work_ptrs, void *cuda_stream) = 0;

  std::vector<size_t> GetInputSizeList() { return input_size_list_; }
  std::vector<size_t> GetOutputSizeList() { return output_size_list_; }
  std::vector<size_t> GetWorkSizeList() { return work_size_list_; }

  // Dynamic kernel can pass output information by this interface.
  virtual TensorInfo GetOutputTensorInfo() { return TensorInfo(); }
  virtual void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) {}
  virtual void ResetResource() {
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }

 protected:
  virtual int CheckKernelParam() { return 0; }

 protected:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> work_size_list_;
  std::string kernel_name_;
  uint32_t device_id_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_HELPER_BASE_H_
