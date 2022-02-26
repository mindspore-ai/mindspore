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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEAVGPOOL2D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEAVGPOOL2D_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool2d_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class AdaptiveAvgPool2DKernelMod : public NativeGpuKernelMod {
 public:
  AdaptiveAvgPool2DKernelMod()
      : input_size_(0),
        output_size_(0),
        len(0),
        input_height(0),
        input_width(0),
        output_height(0),
        output_width(0),
        size(0),
        is_null_input_(false),
        kernel_name_("AdaptiveAvgPool2D") {}
  ~AdaptiveAvgPool2DKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    ApplyAdaptiveAvgPool2D(size, input_height, input_width, output_height, output_width, input_addr, output_addr,
                           reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    auto shape_addr = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "output_size");
    if (shape_addr.size() == 1) {
      output_height = shape_addr[0];
      output_width = shape_addr[0];
    } else if (shape_addr.size() == 2) {
      output_height = static_cast<uint>(shape_addr[0]);
      output_width = static_cast<uint>(shape_addr[1]);
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'output_size' should be 1 or 2, but got "
                        << shape_addr.size();
    }

    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }

    input_size_ = sizeof(T);
    output_size_ = sizeof(T);

    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    len = static_cast<uint>(input_shape.size());

    if (len < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 2, but got "
                        << len;
    }

    input_height = static_cast<uint>(input_shape[len - 2]);
    input_width = static_cast<uint>(input_shape[len - 1]);
    size = static_cast<uint>(len == 3 ? input_shape[0] : input_shape[0] * input_shape[1]);
    for (uint i = 0; i < len; i++) {
      input_size_ *= input_shape[i];
    }

    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t input_size_;
  size_t output_size_;
  uint len;
  uint input_height;
  uint input_width;
  uint output_height;
  uint output_width;
  uint size;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEAVGPOOL2D_GPU_KERNEL_H_
