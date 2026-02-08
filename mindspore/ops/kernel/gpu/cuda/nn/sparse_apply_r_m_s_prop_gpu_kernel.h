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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_R_M_S_PROP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_R_M_S_PROP_GPU_KERNEL_H_

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_apply_r_m_s_prop_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseApplyRMSPropGpuKernelMod : public NativeGpuKernelMod,
                                       public MatchKernelHelper<SparseApplyRMSPropGpuKernelMod> {
 public:
  SparseApplyRMSPropGpuKernelMod() = default;
  ~SparseApplyRMSPropGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool ResizedInputSize(const std::vector<KernelTensor *> &inputs);
  bool ResizedOutputSize(const std::vector<KernelTensor *> &outputs);

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    MS_EXCEPTION_IF_NULL(cuda_stream);
    MS_EXCEPTION_IF_NULL(kernel_func_);
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    kernel_func_(this, inputs, workspace, outputs);
    return true;
  }
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  void *cuda_stream_{nullptr};
  float rho_;
  float momentum_;
  float epsilon_;
  ShapeVector var_shape_;
  bool is_null_input_{false};
  size_t var_first_dim_size_;
  size_t var_outer_dim_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SPARSE_APPLY_R_M_S_PROP_GPU_KERNEL_H_
