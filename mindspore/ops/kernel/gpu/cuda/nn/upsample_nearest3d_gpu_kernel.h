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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UPSAMPLE_NEAREST3D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UPSAMPLE_NEAREST3D_GPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleNearest3dGpuKernelMod : public NativeGpuKernelMod {
 public:
  UpsampleNearest3dGpuKernelMod() = default;
  ~UpsampleNearest3dGpuKernelMod() override = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex1}; }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using UpsampleNearest3dFunc =
    std::function<bool(UpsampleNearest3dGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  UpsampleNearest3dFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UpsampleNearest3dFunc>> func_list_;

  void *cuda_stream_{nullptr};
  std::vector<int64_t> input_shape_{};
  std::vector<int64_t> output_shape_{};
  std::vector<float> scales_{0., 0., 0.};
  std::vector<int64_t> none_list_{};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UPSAMPLE_NEAREST3D_GPU_KERNEL_H_
