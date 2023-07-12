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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_UPSAMPLE_TRILINEAR_3D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_UPSAMPLE_TRILINEAR_3D_GPU_KERNEL_H_

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleTrilinear3DGpuKernelMod : public NativeGpuKernelMod {
 public:
  UpsampleTrilinear3DGpuKernelMod() = default;
  ~UpsampleTrilinear3DGpuKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex1}; }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using UpsampleTrilinear3DFunc = std::function<bool(UpsampleTrilinear3DGpuKernelMod *, const std::vector<AddressPtr> &,
                                                     const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  UpsampleTrilinear3DFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UpsampleTrilinear3DFunc>> func_list_;
  void *cuda_stream_{nullptr};

  bool align_corners_{};
  int64_t n_{};
  int64_t c_{};
  int64_t input_d_{};
  int64_t input_h_{};
  int64_t input_w_{};
  int64_t output_d_{};
  int64_t output_h_{};
  int64_t output_w_{};
  std::vector<double> scales_{0., 0., 0.};
  std::vector<int64_t> none_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_UPSAMPLE_TRILINEAR_3D_GPU_KERNEL_H_
