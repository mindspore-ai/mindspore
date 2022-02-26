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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARY_OP_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARY_OP_GRAD_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_grad_impl.cuh"

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
class UnaryGradOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  UnaryGradOpGpuKernelMod()
      : unary_grad_op_type_(UNARY_OP_GRAD_INVALID_TYPE),
        input_size_(sizeof(T)),
        dx_size_(sizeof(T)),
        output_size_(sizeof(T)),
        workspace_size_(0),
        is_null_input_(false) {}
  ~UnaryGradOpGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
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
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: SqrtGrad, RsqrtGrad, AsinGrad, "
                          << "ACosGrad, AtanGrad, AsinhGrad, AcoshGrad, ReciprocalGrad currently, but got "
                          << unary_grad_op_type_;
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    auto iter = kUnaryGradOpTypeMap.find(kernel_name);
    if (iter == kUnaryGradOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << ", only support these types: SqrtGrad, RsqrtGrad, AsinGrad, "
                        << "ACosGrad, AtanGrad, AsinhGrad, AcoshGrad, ReciprocalGrad currently, but got "
                        << kernel_name;
    }
    unary_grad_op_type_ = iter->second;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    auto dx_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_SHAPE_NULL(dx_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < dx_shape.size(); i++) {
      dx_size_ *= dx_shape[i];
    }
    if (input_size_ != dx_size_) {
      MS_LOG(WARNING) << "For '" << kernel_name << "', both inputs should be equal, but got " << input_size_ << " and "
                      << dx_size_;
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
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARY_OP_GRAD_GPU_KERNEL_H_
