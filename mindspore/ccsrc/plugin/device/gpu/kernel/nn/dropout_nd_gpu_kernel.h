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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_ND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_ND_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DropoutNDGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<DropoutNDGpuKernelMod> {
 public:
  DropoutNDGpuKernelMod() = default;
  ~DropoutNDGpuKernelMod() override = default;

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

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  bool CheckDropOutNdShape();

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  bool is_null_input_{false};
  bool states_init_{false};
  std::vector<size_t> input_shape_;
  size_t input_elements_{};
  size_t channels_{1};
  size_t num_per_channel_{1};
  float keep_prob_{0.5};
  void *cuda_stream_{nullptr};
  curandGenerator_t cu_rand_generator_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_ND_GPU_KERNEL_H_
