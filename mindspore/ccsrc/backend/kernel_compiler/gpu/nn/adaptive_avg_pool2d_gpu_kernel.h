/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/adaptive_avg_pool2d_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class AdaptiveAvgPool2DKernel : public GpuKernel {
 public:
  AdaptiveAvgPool2DKernel()
      : input_size_(0),
        output_size_(0),
        len(0),
        input_height(0),
        input_width(0),
        output_height(0),
        output_width(0),
        size(0),
        is_null_input_(false) {}
  ~AdaptiveAvgPool2DKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

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
    auto shape_addr = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "output_size");
    if (shape_addr.size() == 1) {
      output_height = shape_addr[0];
      output_width = shape_addr[0];
    } else if (shape_addr.size() == 2) {
      output_height = static_cast<uint>(shape_addr[0]);
      output_width = static_cast<uint>(shape_addr[1]);
    } else {
      MS_LOG(ERROR) << "Input Error.";
      return false;
    }

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but adaptive_avg_pool2d needs 1 inputs.";
      return false;
    }

    input_size_ = sizeof(T);
    output_size_ = sizeof(T);

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'AdaptiveAvgPool2dGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    len = static_cast<uint>(input_shape.size());

    if (len < 2) {
      MS_LOG(ERROR) << "The input should have rank at least 2.";
      return false;
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

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEAVGPOOL2D_GPU_KERNEL_H_
