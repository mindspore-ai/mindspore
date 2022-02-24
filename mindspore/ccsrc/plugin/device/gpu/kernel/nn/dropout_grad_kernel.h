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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GRAD_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dropout_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class DropoutGradBwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  DropoutGradBwdGpuKernelMod()
      : cudnn_handle_(nullptr), is_null_input_(false), kernel_name_("DropoutGrad"), num_count_(0), keep_prob_(0.0) {}
  ~DropoutGradBwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *mask = GetDeviceAddress<T>(inputs, 1);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    DropoutBackward(dy, mask, dx, num_count_, keep_prob_, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    InitResource();

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }

    auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    num_count_ = 1;
    for (size_t x : input_shape) {
      num_count_ *= x;
    }
    keep_prob_ = GetAttr<float>(kernel_node, "keep_prob");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override { cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle(); }
  void InitSizeLists() override {
    size_t dy_size = num_count_ * sizeof(T);
    size_t mask_size = dy_size;
    size_t dx_size = dy_size;

    input_size_list_.push_back(dy_size);
    input_size_list_.push_back(mask_size);
    output_size_list_.push_back(dx_size);
  }

 private:
  cudnnHandle_t cudnn_handle_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t num_count_;
  float keep_prob_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GRAD_KERNEL_H_
