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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_ADAGRAD_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_ADAGRAD_V2_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <utility>
#include <memory>
#include <functional>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_apply_adagrad_v2_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseApplyAdagradV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseApplyAdagradV2GpuKernelMod() = default;
  ~SparseApplyAdagradV2GpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    MS_EXCEPTION_IF_NULL(cuda_stream);
    if (is_null_input_) {
      return true;
    }

    cuda_stream_ = cuda_stream;
    kernel_func_(this, inputs, workspace, outputs);
    return true;
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs);
  using SparseApplyAdagradV2Func =
    std::function<bool(SparseApplyAdagradV2GpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SparseApplyAdagradV2Func>> func_list_;
  SparseApplyAdagradV2Func kernel_func_;

  void *cuda_stream_{nullptr};
  bool is_null_input_{false};
  float lr_;
  float epsilon_;
  bool update_slots_;
  int unit_size_;
  size_t input_elements_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_ADAGRAD_V2_GPU_KERNEL_H_
