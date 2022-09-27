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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAX_POOL_GRAD_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAX_POOL_GRAD_GRAD_CPU_KERNEL_H_

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include <unordered_map>
#include "mindspore/core/ops/grad/max_pool_grad_grad.h"
#include "mindspore/core/ops/grad/max_pool_3d_grad_grad.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/pooling_parameter.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMaxPool2DGradGradDim = 2;
constexpr size_t kMaxPool3DGradGradDim = 3;

class MaxPoolGradGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  explicit MaxPoolGradGradCpuKernelMod(const int &dim) : dim_(dim) {}
  ~MaxPoolGradGradCpuKernelMod() override {
    if (param_ != nullptr) {
      free(param_);
      param_ = nullptr;
    }
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void CheckInputVaild() const;
  void CalPad();

  std::vector<int64_t> kernels_;
  std::vector<int64_t> strides_;
  PadMode pad_mode_ = PadMode::PAD;
  std::vector<int64_t> in_shapes_;
  std::vector<int64_t> out_shapes_;
  PoolingParameter *param_ = nullptr;

  size_t dim_ = 0;
  size_t depth_index_ = 0;
  size_t height_index_ = 0;
  size_t width_index_ = 0;
  size_t input_batch_stride_ = 0;
  size_t output_batch_stride_ = 0;
  size_t output_elements_ = 0;
};

class MaxPool2DGradGradCpuKernelMod : public MaxPoolGradGradCpuKernelMod {
 public:
  MaxPool2DGradGradCpuKernelMod() : MaxPoolGradGradCpuKernelMod(kMaxPool2DGradGradDim) {}
  ~MaxPool2DGradGradCpuKernelMod() = default;
};

class MaxPool3DGradGradCpuKernelMod : public MaxPoolGradGradCpuKernelMod {
 public:
  MaxPool3DGradGradCpuKernelMod() : MaxPoolGradGradCpuKernelMod(kMaxPool3DGradGradDim) {}
  ~MaxPool3DGradGradCpuKernelMod() = default;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAX_POOL_GRAD_GRAD_CPU_KERNEL_H_
