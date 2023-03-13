/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SEQUENCE_CONCAT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SEQUENCE_CONCAT_GPU_KERNEL_H_
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/concatv2_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
class SequenceConcatGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<SequenceConcatGpuKernelMod> {
 public:
  SequenceConcatGpuKernelMod() = default;
  ~SequenceConcatGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  using FuncList = std::vector<std::pair<KernelAttr, KernelRunFunc>>;
  const FuncList &GetFuncList() const override;

 private:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  int all_size_before_axis_{1};
  int all_size_axis_{1};
  int ori_axis_{0};
  int axis_{0};
  size_t output_dim_{1};
  int input_num_{1};
  size_t output_size_{0};
  std::vector<int64_t> tuple_shape_;
  std::vector<ShapeVector> inputs_shape_;
  std::vector<int> len_axis_;
  std::vector<void *> inputs_host_;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SEQUENCE_CONCAT_GPU_KERNEL_H_
