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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COALESCE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COALESCE_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unique_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/coalesce_impl.cuh"
namespace mindspore {
namespace kernel {
class CoalesceGpuKernelMod : public NativeGpuKernelMod {
 public:
  CoalesceGpuKernelMod() = default;
  ~CoalesceGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  void ResetResource() noexcept;

 protected:
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &others);
  void InitSizeLists();
  template <typename T, typename S, typename V>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  std::vector<KernelAttr> GetOpSupport() override;
  void SyncData() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  using CoalesceFunc = std::function<bool(CoalesceGpuKernelMod *, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, CoalesceFunc>> func_list_;
  CoalesceFunc kernel_func_;
  void *stream_ptr_;
  bool is_null_input_{false};
  size_t input_indices_size_{0};
  size_t input_values_size_{0};
  size_t input_shape_size_{0};
  size_t output_indices_size_{0};
  size_t output_values_size_{0};
  size_t output_shape_size_{0};
  int post_output_size_{0};
  int dims_{0};
  int num_indices_{0};
  int num_values_{0};
  std::vector<int64_t> input_indices_shape_;
  std::vector<int64_t> input_values_shape_;
  std::vector<int64_t> input_shape_shape_;
  std::vector<int64_t> output_indices_shape_;
  std::vector<int64_t> output_values_shape_;
  std::vector<int64_t> output_shape_shape_;
  std::vector<KernelTensorPtr> outputs_ = {};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_COALESCE_GPU_KERNEL_H_
