/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_UNARYOP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_UNARYOP_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/unary_op_impl.cuh"

namespace mindspore {
namespace kernel {
enum UnaryOptype {
  UNARY_OP_EXP = 0,
  UNARY_OP_LOG,
  UNARY_OP_NEG,
  UNARY_OP_RECIPROCAL,
  UNARY_OP_ZEROSLIKE,
  UNARY_OP_INVALID_TYPE = 255
};
const std::map<std::string, UnaryOptype> kUnaryOpTypeMap = {{"Exp", UNARY_OP_EXP},
                                                            {"Log", UNARY_OP_LOG},
                                                            {"Neg", UNARY_OP_NEG},
                                                            {"Reciprocal", UNARY_OP_RECIPROCAL},
                                                            {"ZerosLike", UNARY_OP_ZEROSLIKE}};
template <typename T>
class UnaryOpGpuKernel : public GpuKernel {
 public:
  UnaryOpGpuKernel()
      : unary_op_type_(UNARY_OP_INVALID_TYPE), input_size_(sizeof(T)), output_size_(sizeof(T)), workspace_size_(0) {}
  ~UnaryOpGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    switch (unary_op_type_) {
      case UNARY_OP_EXP: {
        Exponential(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_LOG: {
        Logarithm(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_NEG: {
        Negative(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_RECIPROCAL: {
        Reciprocal(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case UNARY_OP_ZEROSLIKE: {
        return true;
      }
      default: {
        MS_LOG(EXCEPTION) << "Unary operation " << unary_op_type_ << " is not supported.";
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kUnaryOpTypeMap.find(kernel_name);
    if (iter == kUnaryOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Unary operation " << kernel_name << " is not supported.";
    } else {
      unary_op_type_ = iter->second;
    }
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but negative op needs 1 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but negative op needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    output_size_ = input_size_;
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  UnaryOptype unary_op_type_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_UNARYOP_GPU_KERNEL_H_
