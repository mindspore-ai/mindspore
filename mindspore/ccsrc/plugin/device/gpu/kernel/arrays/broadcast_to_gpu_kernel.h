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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_TO_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_TO_GPU_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/broadcast_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 4;
template <typename T>
class BroadcastToGpuKernelMod : public NativeGpuKernelMod {
 public:
  BroadcastToGpuKernelMod() : kernel_name_("BroadcastTo") {}
  ~BroadcastToGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    BroadcastTo(input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3], output_shape_[0], output_shape_[1],
                output_shape_[2], output_shape_[3], input_addr, output_addr,
                reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    auto input_shapes = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shapes = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shapes, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shapes, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (input_shapes.size() > SHAPE_SIZE || output_shapes.size() > SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output cannot be greater than "
                        << SHAPE_SIZE << ", but got the dimension of input: " << input_shapes.size()
                        << ", the dimension of output: " << output_shapes.size();
    }

    if (output_shapes.size() < input_shapes.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of output cannot be less than the dimension of input "
                        << ", but got the dimension of input: " << input_shapes.size()
                        << ", the dimension of output: " << output_shapes.size();
    }

    size_t offset = output_shapes.size() - input_shapes.size();
    for (size_t i = 0; i < input_shapes.size(); i++) {
      input_shape_[i + offset] = input_shapes[i];
    }

    for (size_t j = 0; j < output_shapes.size(); j++) {
      output_shape_[j] = output_shapes[j];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3] * sizeof(T));
    output_size_list_.push_back(output_shape_[0] * output_shape_[1] * output_shape_[2] * output_shape_[3] * sizeof(T));
  }

 private:
  size_t input_shape_[4] = {1, 1, 1, 1};
  size_t output_shape_[4] = {1, 1, 1, 1};
  bool is_null_input_ = false;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_TO_GPU_KERNEL_H_
