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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARY_COMPLEX_OP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARY_COMPLEX_OP_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/math/unary_op_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
template <typename T>
class UnaryOpComplexGpuKernel : public GpuKernel {
 public:
  UnaryOpComplexGpuKernel() { ResetResource(); }
  ~UnaryOpComplexGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    Complex<T> *input_addr = GetDeviceAddress<Complex<T>>(inputs, 0);
    if (is_c2r_op_) {
      T *output_addr = GetDeviceAddress<T>(outputs, 0);
      switch (unary_op_type_) {
        case UNARY_OP_REAL: {
          Real(input_addr, output_addr, inputs[0]->size / sizeof(Complex<T>),
               reinterpret_cast<cudaStream_t>(stream_ptr));
          break;
        }
        case UNARY_OP_IMAG: {
          Imag(input_addr, output_addr, inputs[0]->size / sizeof(Complex<T>),
               reinterpret_cast<cudaStream_t>(stream_ptr));
          break;
        }
        default: {
          MS_LOG(EXCEPTION) << "Unary operation " << unary_op_type_ << " is not supported.";
        }
      }
    } else {
      Complex<T> *output_addr = GetDeviceAddress<Complex<T>>(outputs, 0);
      switch (unary_op_type_) {
        case UNARY_OP_CONJ: {
          Conj(input_addr, output_addr, inputs[0]->size / sizeof(Complex<T>),
               reinterpret_cast<cudaStream_t>(stream_ptr));
          break;
        }
        default: {
          MS_LOG(EXCEPTION) << "Unary operation " << unary_op_type_ << " is not supported.";
        }
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    GetOpType(kernel_node);
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but unary op needs 1 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but unary op needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "UnaryOpGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
      output_size_ *= input_shape[i];
    }
    if (is_c2r_op_) {
      output_size_ /= 2;
    }
    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    input_size_ = sizeof(Complex<T>);
    output_size_ = sizeof(Complex<T>);
    workspace_size_ = 0;
    is_null_input_ = false;
    is_c2r_op_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  void GetOpType(const CNodePtr &kernel_node) {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    static std::map<std::string, UnaryOptype> kComplexSupportedC2RTypeMap = {{"Real", UNARY_OP_REAL},
                                                                             {"Imag", UNARY_OP_IMAG}};
    auto iter = kComplexSupportedC2RTypeMap.find(kernel_name);
    if (iter != kComplexSupportedC2RTypeMap.end()) {
      unary_op_type_ = iter->second;
      is_c2r_op_ = true;
      return;
    }
    static std::map<std::string, UnaryOptype> kComplexSupportedC2CTypeMap = {{"Conj", UNARY_OP_CONJ}};
    iter = kComplexSupportedC2CTypeMap.find(kernel_name);
    if (iter != kComplexSupportedC2RTypeMap.end()) {
      unary_op_type_ = iter->second;
      is_c2r_op_ = false;
      return;
    }

    MS_LOG(EXCEPTION) << "operation " << kernel_name << " is not supported.";
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
  bool is_c2r_op_;
  UnaryOptype unary_op_type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARY_COMPLEX_OP_GPU_KERNEL_H_
