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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RELU_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RELU_GRAD_KERNEL_H_

#include <vector>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/relu_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ReluGradFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  ReluGradFwdGpuKernelMod() { ResetResource(); }
  ~ReluGradFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *y = GetDeviceAddress<T>(inputs, 1);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    const int size = input_size_ / sizeof(T);
    CalReLUGrad(size, dy, y, dx, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    size_t size = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      size *= input_shape[i];
    }
    input_size_ = size * sizeof(T);

    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    input_size_ = 0;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
  }

 private:
  bool is_null_input_;

  size_t input_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RELU_GRAD_KERNEL_H_
