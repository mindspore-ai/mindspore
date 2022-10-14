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
#include "mindspore/core/ops/coalesce.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/coalesce_helper.h"
namespace mindspore {
namespace kernel {
class CoalesceGpuKernelMod : public NativeGpuKernelMod {
 public:
  CoalesceGpuKernelMod() = default;
  ~CoalesceGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

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

  void InitSizeLists() {
    input_size_list_ = helper_ptr_->GetInputSizeList();
    output_size_list_ = helper_ptr_->GetOutputSizeList();
    workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void *stream_ptr_;
  bool is_null_input_;
  bool First_Resize = true;
  std::unique_ptr<cukernel::GpuKernelHelperBase> helper_ptr_ = nullptr;
  std::optional<bool> is_input_dynamic_shape_ = {};
  BaseOperatorPtr base_operator_ = nullptr;
  std::vector<KernelTensorPtr> inputs_ = {};
  std::vector<KernelTensorPtr> outputs_ = {};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COALESCE_GPU_KERNEL_H_
