/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GPU_KERNEL_H_

#include <map>
#include <utility>
#include <string>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "include/utils/utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/instance_norm_impl.cuh"

namespace mindspore {
namespace kernel {
class InstanceNormGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<InstanceNormGpuKernelMod> {
 public:
  InstanceNormGpuKernelMod() = default;
  ~InstanceNormGpuKernelMod() override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_),
                                       "For 'InstanceNormGpuKernelMod', it destroy x desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_),
                                       "For 'InstanceNormGpuKernelMod', it destroy y desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_),
                                       "For 'InstanceNormGpuKernelMod', it destroy para desc failed");
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<KernelTensor *> &outputs);

  static constexpr cudnnBatchNormOps_t bn_ops_{CUDNN_BATCHNORM_OPS_BN};
  static constexpr cudnnBatchNormMode_t mode_{CUDNN_BATCHNORM_SPATIAL_PERSISTENT};

  size_t batch_{0};
  size_t channel_{0};
  size_t batch_rank_{0};
  size_t workspace_size_{0};
  size_t batch_rank_cum_{0};

  size_t input_offset_{0};
  size_t para_offset_{0};
  size_t updated_para_offset_{0};

  double epsilon_{10e-5};
  double exp_avg_factor_{0.1};
  bool is_null_input_{false};
  cudnnTensorDescriptor_t x_desc_{nullptr};
  cudnnTensorDescriptor_t y_desc_{nullptr};
  cudnnTensorDescriptor_t z_desc_{nullptr};
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_{nullptr};

  cudnnHandle_t handle_{nullptr};
  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};

  cudaStream_t stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GPU_KERNEL_H_
