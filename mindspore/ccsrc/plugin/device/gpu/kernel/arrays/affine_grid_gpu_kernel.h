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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_AFFINE_GRID_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_AFFINE_GRID_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include "mindspore/core/ops/affine_grid.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/affine_grid_impl.cuh"

namespace mindspore {
namespace kernel {
enum class AffineGridDim {
  unknown = 0,
  spatial = 2,     // 4-D, (N, C, H, W)
  volumetric = 3,  // 5-D, (N, C, D, H, W)
};

class AffineGridGpuKernelMod : public NativeGpuKernelMod {
 public:
  AffineGridGpuKernelMod() { ResetResource(); }
  ~AffineGridGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  bool CheckShapeOfInputs(const std::vector<KernelTensorPtr> &inputs);
  bool CheckShapeOfOutputs(const std::vector<KernelTensorPtr> &outputs);
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using AffineGridFunc = std::function<bool(AffineGridGpuKernelMod *, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  AffineGridFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, AffineGridFunc>> func_list_;

 private:
  bool align_corners_{false};
  AffineGridDim grid_dim_{AffineGridDim::unknown};
  size_t data_type_bytes_{1};
  std::vector<int64_t> theta_shape_;
  std::vector<int64_t> grid_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_AFFINE_GRID_GPU_KERNEL_H_
