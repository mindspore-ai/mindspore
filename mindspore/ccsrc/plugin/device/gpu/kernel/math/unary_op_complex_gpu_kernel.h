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
constexpr auto kInputOutputNum = 1;
template <typename T, typename S>
class UnaryOpComplexGpuKernelMod : public NativeGpuKernelMod {
 public:
  UnaryOpComplexGpuKernelMod() {}
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

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    MS_ERROR_IF_NULL(base_operator);
    kernel_name_ = base_operator->name();
    GetOpType(base_operator->GetPrim()->name());
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputOutputNum, kernel_name_);
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputOutputNum, kernel_name_);
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    auto ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    auto input_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return KRET_OK;
    }
    return KRET_OK;
  }

 private:
  void InitSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(S));
  }

  void GetOpType(const std::string kernel_name) {
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
  size_t input_size_{kInputOutputNum};
  size_t output_size_{kInputOutputNum};
  size_t workspace_size_;
  bool is_null_input_{false};
  cukernel::UnaryOptype unary_op_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNARY_COMPLEX_OP_GPU_KERNEL_H_
