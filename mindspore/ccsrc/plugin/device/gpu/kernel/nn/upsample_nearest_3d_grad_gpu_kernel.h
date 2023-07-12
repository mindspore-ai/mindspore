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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GRAD_GPU_KERNEL_NN_UPSAMPLE_NEAREST_3D_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GRAD_GPU_KERNEL_NN_UPSAMPLE_NEAREST_3D_GRAD_GPU_KERNEL_H_
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleNearest3DGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  UpsampleNearest3DGradGpuKernelMod() = default;
  ~UpsampleNearest3DGradGpuKernelMod() override = default;

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

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex2}; }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using UpsampleNearest3DGradFunc =
    std::function<bool(UpsampleNearest3DGradGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  void *cuda_stream_{nullptr};
  int64_t n_{};
  int64_t c_{};
  int64_t dy_d_{};
  int64_t dy_h_{};
  int64_t dy_w_{};
  int64_t dx_d_{};
  int64_t dx_h_{};
  int64_t dx_w_{};
  std::vector<int64_t> none_list_;
  std::vector<double> scales_{0., 0., 0.};
  UpsampleNearest3DGradFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UpsampleNearest3DGradFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GRAD_GPU_KERNEL_NN_UPSAMPLE_NEAREST_3D_GRAD_GPU_KERNEL_H_
