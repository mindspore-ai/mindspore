/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/unary_op_cpu_kernel.h"
#include <map>
#include <string>
#include <vector>
namespace mindspore {
namespace kernel {
template <typename T, typename S>
void Real(const T *input, S *output, size_t start, size_t end) {
  if (!std::is_same<S, complex64>::value && !std::is_same<S, complex128>::value) {
    for (size_t i = start; i < end; ++i) {
      output[i] = static_cast<S>(std::real(input[i]));
    }
  } else {
    MS_LOG(EXCEPTION) << "For Real, it's output data type only support these types: float or double";
  }
}

template <typename T, typename S>
void Imag(const T *input, S *output, size_t start, size_t end) {
  if constexpr (!std::is_same<S, std::complex<float>>::value && !std::is_same<S, std::complex<double>>::value) {
    for (size_t i = start; i < end; ++i) {
      output[i] = static_cast<S>(std::imag(input[i]));
    }
  } else {
    MS_LOG(EXCEPTION) << "For Imag, it's output data type only support these types: float or double";
  }
}

template <typename T, typename S>
void Conj(const T *input, S *output, size_t start, size_t end) {
  if constexpr (std::is_same<T, S>::value &&
                (std::is_same<T, complex64>::value || std::is_same<T, complex128>::value)) {
    for (size_t i = start; i < end; ++i) {
      output[i] = static_cast<S>(std::conj(input[i]));
    }
  } else {
    MS_LOG(EXCEPTION) << "For Conj, it's output data type only support these types: complex<float> or complex<double>";
  }
}

template <typename T, typename S>
void UnaryOpCPUKernel<T, S>::GetUnaryOpFunc() {
  if constexpr (std::is_same<T, complex64>::value || std::is_same<T, complex128>::value) {
    static std::map<std::string, UnaryOpFunc> kComplexSupportedTypeMap = {{prim::kPrimReal->name(), &Real<T, S>},
                                                                          {prim::kPrimImag->name(), &Imag<T, S>},
                                                                          {prim::kPrimConj->name(), &Conj<T, S>}};
    auto iter = kComplexSupportedTypeMap.find(kernel_name_);
    if (iter != kComplexSupportedTypeMap.end()) {
      unary_op_func_ = iter->second;
      return;
    }
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: Real, Imag, Conj currently, but got "
                      << kernel_name_;
  }
}

template <typename T, typename S>
void UnaryOpCPUKernel<T, S>::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  GetUnaryOpFunc();
}

template <typename T, typename S>
bool UnaryOpCPUKernel<T, S>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  auto input = inputs.front();
  auto output = outputs.front();
  const auto input_addr = reinterpret_cast<T *>(input->addr);
  auto output_addr = reinterpret_cast<S *>(output->addr);
  if (unary_op_func_ != nullptr) {
    ParallelLaunchAutoSearch(
      std::bind(unary_op_func_, input_addr, output_addr, std::placeholders::_1, std::placeholders::_2),
      output->size / sizeof(S), this, &parallel_search_info_);
  } else {
    (void)memcpy_s(output_addr, output->size, input_addr, input->size);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
