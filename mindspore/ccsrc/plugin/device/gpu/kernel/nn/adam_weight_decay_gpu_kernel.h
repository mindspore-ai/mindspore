/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_WEIGHT_DECAY_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_WEIGHT_DECAY_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adam_impl.cuh"
namespace mindspore {
namespace kernel {
class AdamWeightDecayGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<AdamWeightDecayGpuKernelMod> {
 public:
  AdamWeightDecayGpuKernelMod() = default;

  ~AdamWeightDecayGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    stream_ptr_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  void InitSizeLists();

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 private:
  size_t variable_size_{0};
  size_t m_size_{0};
  size_t v_size_{0};
  size_t learning_rate_size_{0};
  size_t beta1_size_{0};
  size_t beta2_size_{0};
  size_t epsilon_size_{0};
  size_t decay_size_{0};
  size_t gradient_size_{0};
  cudaStream_t stream_ptr_{nullptr};
  size_t t_type_id_size_{0};
  size_t s_type_id_size_{0};
  bool is_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_WEIGHT_DECAY_GPU_KERNEL_H_
