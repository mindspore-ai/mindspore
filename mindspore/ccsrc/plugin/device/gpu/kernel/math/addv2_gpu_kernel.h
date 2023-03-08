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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDV2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDV2_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <functional>
#include <string>
#include "ops/complex.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/addv2_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
constexpr int INPUTS_SIZE = 2;
constexpr int MIN_DIMS = 0;
constexpr auto kUnknown = "Unknown";
class AddV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  AddV2GpuKernelMod() = default;
  ~AddV2GpuKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using AddV2Func = std::function<bool(AddV2GpuKernelMod *, const std::vector<AddressPtr> &,
                                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;

 private:
  size_t unit_size_{1};
  bool need_broadcast_;
  std::string kernel_name_{kUnknown};
  size_t input_elements_{0};
  size_t output_num_{1};
  void *stream_ptr_{nullptr};
  std::vector<size_t> input1_shape_;
  std::vector<size_t> input2_shape_;
  std::vector<size_t> output_shape_;
  AddV2Func kernel_func_;
  static std::vector<std::pair<KernelAttr, AddV2Func>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif
