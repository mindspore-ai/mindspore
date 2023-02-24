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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UPSAMPLE_TRILINEAR_3D_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UPSAMPLE_TRILINEAR_3D_GRAD_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include <functional>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleTrilinear3DGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  UpsampleTrilinear3DGradGpuKernelMod() = default;
  ~UpsampleTrilinear3DGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;

  float ScalingD(const size_t in_size, const size_t out_size, bool align_corners);
  float ScalingS(float scale_value, int idx, const size_t out_size);

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  using UpsampleTrilinear3DGradFunc =
    std::function<bool(UpsampleTrilinear3DGradGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;

  bool GetUpsampleTrilinear3DGradAttr(const BaseOperatorPtr &base_operator);

  void *cuda_stream_{nullptr};
  bool is_null_input_{false};
  bool align_corners_{};
  size_t t_size_{0};

  // array dims -> reset these
  size_t n_{};
  size_t c_{};
  size_t grad_d_{};
  size_t grad_h_{};
  size_t grad_w_{};
  size_t dinput_d_{};
  size_t dinput_h_{};
  size_t dinput_w_{};

  // only need these
  std::vector<int64_t> out_spatial_size_me_;
  std::vector<float> scale_factors_;
  UpsampleTrilinear3DGradFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UpsampleTrilinear3DGradFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_UPSAMPLE_TRILINEAR_3D_GRAD_GPU_KERNEL_H_
