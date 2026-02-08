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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_CENTERED_RMS_PROP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_CENTERED_RMS_PROP_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include <iostream>
#include <utility>
#include <memory>
#include <functional>
#include <map>
#include <string>
#include <cstdio>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_apply_centered_rms_prop_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseApplyCenteredRMSPropGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseApplyCenteredRMSPropGpuKernelMod() = default;
  ~SparseApplyCenteredRMSPropGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    MS_EXCEPTION_IF_NULL(cuda_stream);
    if (is_null_input_) {
      return true;
    }

    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<KernelTensor *> &outputs);
  using SparseApplyCenteredRMSPropFunc =
    std::function<bool(SparseApplyCenteredRMSPropGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, SparseApplyCenteredRMSPropFunc>> func_list_;
  SparseApplyCenteredRMSPropFunc kernel_func_;

  void *cuda_stream_{nullptr};
  bool is_null_input_{false};
  bool use_locking_;
  int unit_size_;
  size_t input_elements_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_CENTERED_RMS_PROP_GPU_KERNEL_H_
