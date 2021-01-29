/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARYOP_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARYOP_GRAD_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/unary_op_grad_impl.cuh"

namespace mindspore {
namespace kernel {
enum UnaryGradOptype {
  UNARY_OP_SQRT_GRAD = 0,
  UNARY_OP_RSQRT_GRAD = 1,
  UNARY_OP_ASIN_GRAD = 2,
  UNARY_OP_ACOS_GRAD = 3,
  UNARY_OP_ATAN_GRAD = 4,
  UNARY_OP_ASINH_GRAD = 5,
  UNARY_OP_ACOSH_GRAD = 6,
  UNARY_OP_RECIPROCAL_GRAD = 7,
  UNARY_OP_GRAD_INVALID_TYPE = 255
};
static const std::map<std::string, UnaryGradOptype> kUnaryGradOpTypeMap = {
  {"SqrtGrad", UNARY_OP_SQRT_GRAD},   {"RsqrtGrad", UNARY_OP_RSQRT_GRAD},
  {"AsinGrad", UNARY_OP_ASIN_GRAD},   {"ACosGrad", UNARY_OP_ACOS_GRAD},
  {"AtanGrad", UNARY_OP_ATAN_GRAD},   {"AsinhGrad", UNARY_OP_ASINH_GRAD},
  {"AcoshGrad", UNARY_OP_ACOSH_GRAD}, {"ReciprocalGrad", UNARY_OP_RECIPROCAL_GRAD}};

template <typename T>
class UnaryGradOpGpuKernel : public GpuKernel {
 public:
  UnaryGradOpGpuKernel()
      : unary_grad_op_type_(UNARY_OP_GRAD_INVALID_TYPE),
        input_size_(sizeof(T)),
        dx_size_(sizeof(T)),
        output_size_(sizeof(T)),
        workspace_size_(0),
        is_null_input_(false) {}
  ~UnaryGradOpGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input_x_addr = GetDeviceAddress<T>(inputs, 0);
    T *input_dx_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_y_addr = GetDeviceAddress<T>(outputs, 0);

    switch (unary_grad_op_type_) {
      case UNARY_OP_SQRT_GRAD: {
        SqrtGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                 reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_ASIN_GRAD: {
        AsinGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                 reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_ACOS_GRAD: {
        ACosGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                 reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_ATAN_GRAD: {
        AtanGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                 reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_ASINH_GRAD: {
        AsinhGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                  reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_ACOSH_GRAD: {
        AcoshGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                  reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_RSQRT_GRAD: {
        RsqrtGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                  reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_RECIPROCAL_GRAD: {
        ReciprocalGrad(input_x_addr, input_dx_addr, output_y_addr, inputs[0]->size / sizeof(T),
                       reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "Unary grad operation " << unary_grad_op_type_ << " is not supported.";
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kUnaryGradOpTypeMap.find(kernel_name);
    if (iter == kUnaryGradOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Unary grad operation " << kernel_name << " is not supported.";
    } else {
      unary_grad_op_type_ = iter->second;
    }
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but unary grad op needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but unary grad op needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "UnaryGradOpGpuKernel input 0 is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    auto dx_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(dx_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "UnaryGradOpGpuKernel input 1 is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < dx_shape.size(); i++) {
      dx_size_ *= dx_shape[i];
    }
    if (input_size_ != dx_size_) {
      MS_LOG(WARNING) << "UnaryGradOpGpuKernel inputs should be same, but got " << input_size_ << " and " << dx_size_;
      InitSizeLists();
      return true;
    }
    output_size_ = input_size_;
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(dx_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  UnaryGradOptype unary_grad_op_type_;
  size_t input_size_;
  size_t dx_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARYOP_GRAD_GPU_KERNEL_H_
