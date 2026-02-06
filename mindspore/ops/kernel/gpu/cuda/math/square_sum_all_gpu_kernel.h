/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARE_SUM_ALL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARE_SUM_ALL_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "mindspore/ops/infer/square_sum_all.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/square_sum_all_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SquareSumAllFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  SquareSumAllFwdGpuKernelMod() = default;
  ~SquareSumAllFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    constexpr size_t kSquareSumAllInputsNum = 2;
    constexpr size_t kSquareSumAllOutputsNum = 2;
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSquareSumAllInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSquareSumAllOutputsNum, kernel_name_);
    T *input_addr_0 = GetDeviceAddress<T>(inputs, 0);
    T *input_addr_1 = GetDeviceAddress<T>(inputs, 1);
    T *output_addr_0 = GetDeviceAddress<T>(outputs, 0);
    T *output_addr_1 = GetDeviceAddress<T>(outputs, 1);
    float *ws_addr_0 = GetDeviceAddress<float>(workspace, 0);
    float *ws_addr_1 = GetDeviceAddress<float>(workspace, 1);
    auto status = SquareSumAll(input_size_, input_addr_0, input_addr_1, output_addr_0, output_addr_1, ws_addr_0,
                               ws_addr_1, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    dtype_ = inputs.at(kIndex0)->dtype_id();
    dtype_size_ = abstract::TypeIdSize(dtype_);
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto input_shape = inputs[0]->GetShapeVector();
    auto output_shape = outputs[0]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    is_null_output_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    input_size_ = 1;
    output_size_ = 1;
    if (is_null_input_ || is_null_output_) {
      InitSizeLists();
      return KRET_OK;
    }
    for (size_t i = 0; i < input_shape.size(); ++i) {
      input_size_ *= static_cast<size_t>(input_shape[i]);
    }
    for (size_t i = 0; i < output_shape.size(); ++i) {
      output_size_ *= static_cast<size_t>(output_shape[i]);
    }
    InitSizeLists();
    return KRET_OK;
  }

 protected:
  void InitSizeLists() {
    workspace_size_list_.clear();
    output_size_list_.clear();
    output_size_list_.push_back(dtype_size_);
    output_size_list_.push_back(dtype_size_);
    workspace_size_list_.push_back(output_size_ * dtype_size_);
    workspace_size_list_.push_back(output_size_ * dtype_size_);
  }

 private:
  TypeId dtype_;
  size_t dtype_size_;
  size_t input_size_{1};
  size_t output_size_{1};
  bool is_null_input_{false};
  bool is_null_output_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SQUARE_SUM_ALL_GPU_KERNEL_H_
