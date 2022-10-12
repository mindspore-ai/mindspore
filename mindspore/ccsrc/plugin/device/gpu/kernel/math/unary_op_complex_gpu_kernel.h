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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARY_COMPLEX_OP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARY_COMPLEX_OP_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/unary_helper.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class UnaryOpComplexGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  UnaryOpComplexGpuKernelMod() { ResetResource(); }
  ~UnaryOpComplexGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);

    S *output_addr = GetDeviceAddress<S>(outputs, 0);
    switch (unary_op_type_) {
      case cukernel::UNARY_OP_REAL: {
        if constexpr (!std::is_same<S, utils::Complex<float>>::value &&
                      !std::is_same<S, utils::Complex<double>>::value) {
          Real(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        }
        break;
      }
      case cukernel::UNARY_OP_IMAG: {
        if constexpr (!std::is_same<S, utils::Complex<float>>::value &&
                      !std::is_same<S, utils::Complex<double>>::value) {
          Imag(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        }
        break;
      }
      case cukernel::UNARY_OP_CONJ: {
        if constexpr (std::is_same<T, S>::value && !std::is_same<T, bool>::value) {
          Conj(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        }
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: Real, Imag, Conj currently, "
                          << "but got " << unary_op_type_;
      }
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    GetOpType(kernel_node);
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= static_cast<size_t>(input_shape[i]);
      output_size_ *= static_cast<size_t>(input_shape[i]);
    }
    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    input_size_ = 1;
    output_size_ = 1;
    workspace_size_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(S));
  }

 private:
  void GetOpType(const CNodePtr &kernel_node) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    static std::map<std::string, cukernel::UnaryOptype> kComplexSupportedTypeMap = {
      {"Real", cukernel::UNARY_OP_REAL}, {"Imag", cukernel::UNARY_OP_IMAG}, {"Conj", cukernel::UNARY_OP_CONJ}};
    auto iter = kComplexSupportedTypeMap.find(kernel_name);
    if (iter != kComplexSupportedTypeMap.end()) {
      unary_op_type_ = iter->second;
      return;
    }
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: Real, Imag, Conj currently, but got "
                      << kernel_name;
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
  cukernel::UnaryOptype unary_op_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARY_COMPLEX_OP_GPU_KERNEL_H_
