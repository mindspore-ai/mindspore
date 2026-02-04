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

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleTrilinear3DGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  UpsampleTrilinear3DGradGpuKernelMod() = default;
  ~UpsampleTrilinear3DGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex2}; }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  using UpsampleTrilinear3DGradFunc =
    std::function<bool(UpsampleTrilinear3DGradGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  UpsampleTrilinear3DGradFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UpsampleTrilinear3DGradFunc>> func_list_;

  void *cuda_stream_{nullptr};
  bool align_corners_{};
  // array dims -> reset these
  int64_t n_{};
  int64_t c_{};
  int64_t grad_d_{};
  int64_t grad_h_{};
  int64_t grad_w_{};
  int64_t dinput_d_{};
  int64_t dinput_h_{};
  int64_t dinput_w_{};

  // only need these
  std::vector<float> scales_{0., 0., 0.};
  std::vector<int64_t> none_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_UPSAMPLE_TRILINEAR_3D_GRAD_GPU_KERNEL_H_
