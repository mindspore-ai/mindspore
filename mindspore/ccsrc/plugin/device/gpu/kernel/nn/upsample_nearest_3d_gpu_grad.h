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
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>
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
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool GetUpsampleNearest3DAttr(const BaseOperatorPtr &base_operator);

  template <typename F>
  void CheckDims(string check_dim_name, int expected_size, std::vector<F> check_vector);

  void ResetResource() noexcept;
  float ScalingSizes(const size_t in_size, const size_t out_size);
  float ScalingScales(float scale_value, size_t idx);
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using UpsampleNearest3DGradFunc =
    std::function<bool(UpsampleNearest3DGradGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  void *cuda_stream_{nullptr};
  bool is_null_input_{false};
  size_t t_size_{0};
  size_t n_{};
  size_t c_{};
  size_t dy_d_{};
  size_t dy_h_{};
  size_t dy_w_{};
  size_t dx_d_{};
  size_t dx_h_{};
  size_t dx_w_{};
  std::vector<int64_t> out_spatial_size_me_;
  std::vector<float> scale_factors_;
  UpsampleNearest3DGradFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UpsampleNearest3DGradFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GRAD_GPU_KERNEL_NN_UPSAMPLE_NEAREST_3D_GRAD_GPU_KERNEL_H_
