/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COALESCE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COALESCE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <map>
#include <functional>
#include "mindspore/ops/infer/coalesce.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_class/coalesce_helper.h"
namespace mindspore {
namespace kernel {
class CoalesceGpuKernelMod : public NativeGpuKernelMod {
 public:
  CoalesceGpuKernelMod() = default;
  ~CoalesceGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  void ResetResource() noexcept {
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void InitSizeLists() {
    output_size_list_ = helper_ptr_->GetOutputSizeList();
    workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  cudaStream_t cuda_stream_;
  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_ = nullptr;
  std::optional<bool> is_input_dynamic_shape_ = {};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COALESCE_GPU_KERNEL_H_
