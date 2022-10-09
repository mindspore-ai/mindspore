/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDN_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/math/broadcast_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
class AddNFwdGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  AddNFwdGpuKernelMod() : input_size_(0), output_size_(0), workspace_size_(0), is_null_input_(false), num_input_(0) {}
  ~AddNFwdGpuKernelMod() override {}

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    auto work_addr = output_addr;
    for (size_t i = 0; i < num_input_; i++) {
      if (output_addr == GetDeviceAddress<T>(inputs, i)) {
        work_addr = GetDeviceAddress<T>(workspace, 0);
        break;
      }
    }
    FillDeviceArray(outputs[0]->size / sizeof(T), output_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr));
    FillDeviceArray(outputs[0]->size / sizeof(T), work_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr));
    for (size_t i = 0; i < num_input_; i++) {
      T *input_addr = GetDeviceAddress<T>(inputs, i);
      if constexpr (std::is_same<T, Complex<float>>::value || std::is_same<T, Complex<double>>::value) {
        ElewiseComplexArith(outputs[0]->size / sizeof(T), BROADCAST_TYPE_ADD, input_addr, work_addr, work_addr,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        ElewiseArith(outputs[0]->size / sizeof(T), BROADCAST_TYPE_ADD, input_addr, work_addr, work_addr,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
      }
    }
    if (work_addr != output_addr) {
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(output_addr, work_addr, outputs[0]->size, cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Addn cudaMemcpyAsync outputs failed");
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    num_input_ = GetAttr<int64_t>(kernel_node, "n");
    if (num_input_ != input_num) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be  " << num_input_ << ", but got "
                        << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_size_ = sizeof(T);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= static_cast<size_t>(input_shape[i]);
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    ResetSizeLists();
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    is_null_input_ = false;
    num_input_ = 0;
  }

 protected:
  void InitSizeLists() override {
    for (size_t i = 0; i < num_input_; i++) {
      input_size_list_.push_back(input_size_);
    }
    output_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
  size_t num_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDN_GPU_KERNEL_H_
