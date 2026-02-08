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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_CONSECUTIVE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_CONSECUTIVE_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_class/unique_consecutive_helper.h"
namespace mindspore {
namespace kernel {
class UniqueConsecutiveGpuKernelMod : public NativeGpuKernelMod {
 public:
  UniqueConsecutiveGpuKernelMod() {
    KernelMod::kernel_name_ = "UniqueConsecutive";
    ResetResource();
  }
  ~UniqueConsecutiveGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  void ResetResource() noexcept {
    is_null_input_ = false;
    stream_ptr_ = nullptr;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    output_size_list_ = helper_ptr_->GetOutputSizeList();
    workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  }

  std::vector<KernelAttr> GetOpSupport() override;
  void InitUniqueConsecutiveAttrs(const std::vector<KernelTensor *> &inputs);

 private:
  void *stream_ptr_ = nullptr;
  bool is_null_input_{true};
  bool return_idx_{false};
  bool return_counts_{false};
  bool is_flattend_{false};
  int64_t axis_;
  std::unique_ptr<cukernel::UniqueConsecutiveHelperBase> helper_ptr_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_CONSECUTIVE_GPU_KERNEL_H_
