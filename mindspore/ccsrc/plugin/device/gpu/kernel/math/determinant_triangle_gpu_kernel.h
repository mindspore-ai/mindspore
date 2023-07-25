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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_DETRMINANT_TRIANGLE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_DETRMINANT_TRIANGLE_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/determinant_triangle_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class DetTriangleGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  DetTriangleGpuKernelMod() : input_size_(sizeof(T)), output_size_(sizeof(T)) {}
  ~DetTriangleGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    bool host_error_res = false;
    auto status = CheckTriangle(input_addr, fill_mode_, matrix_n_, outputs[0]->size / sizeof(T),
                                reinterpret_cast<cudaStream_t>(stream_ptr), &host_error_res);
    CHECK_CUDA_STATUS(status, kernel_name_);
    if (!host_error_res) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the elements in the upper half of the matrix should be all 0, fill mode is: " << fill_mode_;
      return false;
    }
    DetTriangle(input_addr, output_addr, matrix_n_, outputs[0]->size / sizeof(T),
                reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    kernel_node_ = kernel_node;
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_size_ *= SizeOf(input_shape);

    if (input_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 2, but got "
                        << input_shape.size();
    }

    matrix_n_ = LongToSizeClipNeg(input_shape[input_shape.size() - 1]);
    output_size_ *= SizeOf(output_shape);
    if (matrix_n_ == 0 || output_size_ != input_size_ / matrix_n_ / matrix_n_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of output should be "
                        << (input_size_ / matrix_n_ / matrix_n_) << ", but got " << output_size_;
    }
    if (input_shape[input_shape.size() - 2] != input_shape[input_shape.size() - 1]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of input should be square matrix";
    }
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    fill_mode_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("fill_mode")));
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
  size_t matrix_n_ = 0;
  int fill_mode_ = 0;
  bool is_null_input_ = false;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_DETRMINANT_TRIANGLE_GPU_KERNEL_H_
