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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EUCLIDEANNORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EUCLIDEANNORM_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class EuclideanNormGpuKernelMod : public NativeGpuKernelMod {
 public:
  EuclideanNormGpuKernelMod() = default;
  ~EuclideanNormGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (inputs[kIndex0]->size() == 0) {
      // input is null, axis is not null, infer output is not null. memset outputs 0.
      cudaMemset(outputs[kIndex0]->device_ptr(), 0, outputs[kIndex0]->size());
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void InitWorkSpaceSizeList();
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using EuclideanNormFunc =
    std::function<bool(EuclideanNormGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

  bool GetEuclideanNormAttr();

  TypeId data_type_{kNumberTypeFloat32};
  std::vector<int64_t> axes_;
  bool keep_dims_;
  void *cuda_stream_{nullptr};
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_axes_;
  std::vector<size_t> output_stride_;
  size_t input_elements_{};
  size_t output_elements_{};
  EuclideanNormFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, EuclideanNormFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_EUCLIDEANNORM_GPU_KERNEL_H_
