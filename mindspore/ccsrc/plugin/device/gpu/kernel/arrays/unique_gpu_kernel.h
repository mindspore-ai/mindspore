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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/unique_helper.h"
namespace mindspore {
namespace kernel {
class UniqueGpuKernelMod : public NativeGpuKernelMod {
 public:
  UniqueGpuKernelMod() {
    KernelMod::kernel_name_ = "Unique";
    ResetResource();
  }
  ~UniqueGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = stream_ptr;
    std::vector<void *> input_ptrs = ConvertPtrs(inputs);
    std::vector<void *> work_ptrs = ConvertPtrs(workspace);
    std::vector<void *> output_ptrs = ConvertPtrs(outputs);
    if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
      return false;
    }
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 protected:
  void SyncData() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

  void ResetResource() noexcept {
    is_null_input_ = false;
    stream_ptr_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    input_size_list_ = helper_ptr_->GetInputSizeList();
    output_size_list_ = helper_ptr_->GetOutputSizeList();
    workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_ = nullptr;
  std::optional<bool> is_input_dynamic_shape_ = {};
  bool is_null_input_;
  void *stream_ptr_;
  BaseOperatorPtr base_operator_ = nullptr;
  std::vector<KernelTensorPtr> inputs_ = {};
  std::vector<KernelTensorPtr> outputs_ = {};
  size_t batch_rank_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_
