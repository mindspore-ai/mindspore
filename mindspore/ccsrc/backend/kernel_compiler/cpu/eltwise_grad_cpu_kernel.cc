/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/eltwise_grad_cpu_kernel.h"
#include <string>
#include <map>
#include "common/thread_pool.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "nnacl/fp32_grad/activation_grad.h"
#include "nnacl/fp32_grad/arithmetic_grad.h"
#include "nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
template <typename T>
void EltWiseGradCPUKernel<T>::ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "ReLUGrad only support float";
  }

  int ret = ::ReluGrad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "ReLUGrad execute failed.";
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "ReLU6Grad only support float";
  }

  int ret = ::Relu6Grad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "ReLU6Grad execute failed.";
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "AbsGrad only support float";
  }

  int ret = ::ElementAbsGrad(input1 + start, input2 + start, out + start, end - start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "AbsGrad execute failed.";
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "SigmoidGrad only support float";
  }

  int ret = ::SigmoidGrad(input2 + start, input1 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "SigmoidGrad execute failed.";
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    out[i] = input2[i] / (input1[i] * 2);
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "TanhGrad only support float";
  }

  int ret = ::TanhGrad(input2 + start, input1 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "TanhGrad execute failed.";
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T x = input2[i];
    auto double_x = static_cast<T>(x);
    T tanh_res = (T)std::tanh(0.7978845608 * (double_x + 0.044715 * double_x * double_x * double_x));
    T mul_right = (T)(0.7978845608 + 0.1070322244 * double_x * double_x);
    T y_res = (((T)1.0 + tanh_res) + x * ((T)1.0 - tanh_res * tanh_res) * mul_right) / (T)2.0;
    out[i] = input1[i] * y_res;
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::AsinGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = sqrt(1 - input1[i] * input1[i]);
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::ACosGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = -input2[i];
    T divisor = sqrt(1 - input1[i] * input1[i]);
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::AtanGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = 1 + input1[i] * input1[i];
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::AsinhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = cosh(input1[i]);
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::AcoshGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = sinh(input1[i]);
    if (divisor == 0) {
      if (dividend == 0) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::SoftplusGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "SoftplusGrad only support float";
  }

  int ret = ::SoftplusGrad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "SoftplusGrad execute failed.";
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
}

template <typename T>
bool EltWiseGradCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  static const std::map<std::string,
                        std::function<void(EltWiseGradCPUKernel *, const T *, const T *, T *, size_t, size_t)>>
    elt_map{{prim::kPrimReluGrad->name(), &EltWiseGradCPUKernel<T>::ReluGrad},
            {prim::kPrimRelu6Grad->name(), &EltWiseGradCPUKernel<T>::ReLU6Grad},
            {prim::kPrimSigmoidGrad->name(), &EltWiseGradCPUKernel<T>::SigmoidGrad},
            {prim::kPrimAbsGrad->name(), &EltWiseGradCPUKernel<T>::AbsGrad},
            {prim::kPrimTanhGrad->name(), &EltWiseGradCPUKernel<T>::TanhGrad},
            {prim::kPrimSqrtGrad->name(), &EltWiseGradCPUKernel<T>::SqrtGrad},
            {prim::kPrimGeLUGrad->name(), &EltWiseGradCPUKernel<T>::GeluGrad},
            {prim::kPrimAsinGrad->name(), &EltWiseGradCPUKernel<T>::AsinGrad},
            {prim::kPrimACosGrad->name(), &EltWiseGradCPUKernel<T>::ACosGrad},
            {prim::kPrimAtanGrad->name(), &EltWiseGradCPUKernel<T>::AtanGrad},
            {prim::kPrimAsinhGrad->name(), &EltWiseGradCPUKernel<T>::AsinhGrad},
            {prim::kPrimAcoshGrad->name(), &EltWiseGradCPUKernel<T>::AcoshGrad},
            {prim::kPrimSoftplusGrad->name(), &EltWiseGradCPUKernel<T>::SoftplusGrad}};
  if (inputs.size() < kInputMinNum || outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << kernel_name_ << " requires at least 2 inputs and 1 output, but got " << inputs.size()
                  << " inputs and " << outputs.size() << " output.";
    return false;
  }
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << kernel_name_ << " output memory size should be greater than 0, but got 0.";
    return true;
  }
  const auto input0 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto input1 = reinterpret_cast<T *>(inputs[1]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  ParallelLaunchAutoSearch(
    std::bind(elt_map.at(kernel_name_), this, input0, input1, output, std::placeholders::_1, std::placeholders::_2),
    outputs[0]->size / sizeof(T), this, &parallel_search_info_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
