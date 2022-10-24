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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_TO_DENSE_V2_GPU_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_TO_DENSE_V2_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "mindspore/core/ops/sparse_to_dense_v2.h"
#include "abstract/utils.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_to_dense_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseToDenseV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseToDenseV2GpuKernelMod() { ResetResource(); }
  ~SparseToDenseV2GpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  template <typename I, typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  template <typename I, typename T>
  void CheckValidateOneDim(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &workspace,
                           const std::vector<kernel::AddressPtr> &outputs);
  template <typename I, typename T>
  void CheckValidateTwoDim(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &workspace,
                           const std::vector<kernel::AddressPtr> &outputs);

  using SparseToDenseV2LaunchFunc =
    std::function<bool(SparseToDenseV2GpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseToDenseV2LaunchFunc>> func_list_;
  SparseToDenseV2LaunchFunc kernel_func_{};
  size_t indice_size_{1};
  size_t value_size_{1};
  size_t input_elements_indices;
  size_t input_elements_values;
  size_t input_elements_output_shape;
  size_t output_elements;
  int ndims;
  int num_elems;
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};
  bool validate_indices_{true};
  std::vector<size_t> indices_shape_;
  std::vector<size_t> output_shape_;
  size_t indices_dims_{0};
  size_t values_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_TO_DENSE_V2_GPU_KERNEL_H_
