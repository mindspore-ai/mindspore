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

#include "plugin/device/cpu/kernel/eltwise_grad_cpu_kernel.h"
#include <string>
#include <map>
#include "common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "nnacl/fp32_grad/activation_grad.h"
#include "nnacl/fp32_grad/arithmetic_grad.h"
#include "nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
template <typename T>
void EltWiseGradCpuKernelMod<T>::ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "For 'ReLUGrad', the dtype of input should be float.";
  }

  int ret = ::ReluGrad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "ReLUGrad execute failed. Error no: " << ret;
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "For 'ReLU6Grad', the dtype of input should be float.";
  }

  int ret = ::Relu6Grad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "ReLU6Grad execute failed. Error no: " << ret;
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (std::is_same<T, float>::value) {
    int ret = ::ElementAbsGrad(input1 + start, input2 + start, out + start, end - start);
    if (ret == NNACL_ERR) {
      MS_LOG(EXCEPTION) << "AbsGrad execute failed. Error no: " << ret;
    }
  } else {
    for (size_t i = start; i < end; i++) {
      out[i] = (input1[i] < 0) ? -input2[i] : ((input1[i] > 0) ? input2[i] : 0);
    }
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "For 'SigmoidGrad', the dtype of input should be float.";
  }

  int ret = ::SigmoidGrad(input2 + start, input1 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "SigmoidGrad execute failed. Error no: " << ret;
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    out[i] = input2[i] / (input1[i] * 2);
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "For 'TanhGrad', the dtype of input should be float.";
  }

  int ret = ::TanhGrad(input2 + start, input1 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "TanhGrad execute failed. Error no: " << ret;
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
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
void EltWiseGradCpuKernelMod<T>::AsinGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
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
void EltWiseGradCpuKernelMod<T>::ACosGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
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
void EltWiseGradCpuKernelMod<T>::AtanGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
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
void EltWiseGradCpuKernelMod<T>::AsinhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
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
void EltWiseGradCpuKernelMod<T>::AcoshGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
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
void EltWiseGradCpuKernelMod<T>::ComplexAcoshGrad(const T *input1, const T *input2, T *out, size_t start,
                                                  size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = std::conj(sinh(input1[i]));
    if (divisor == static_cast<T>(0)) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::SoftplusGrad(const T *input1, const T *input2, T *out, size_t start,
                                              size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "For 'SoftplusGrad', the dtype of input should be float.";
  }

  int ret = ::SoftplusGrad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "SoftplusGrad execute failed. Error no: " << ret;
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::InitComputeFunc() {
  if constexpr (std::is_same_v<T, double>) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuKernelMod *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimSqrtGrad->name(), &EltWiseGradCpuKernelMod<T>::SqrtGrad},
              {prim::kPrimGeLUGrad->name(), &EltWiseGradCpuKernelMod<T>::GeluGrad},
              {prim::kPrimAsinGrad->name(), &EltWiseGradCpuKernelMod<T>::AsinGrad},
              {prim::kPrimACosGrad->name(), &EltWiseGradCpuKernelMod<T>::ACosGrad},
              {prim::kPrimAtanGrad->name(), &EltWiseGradCpuKernelMod<T>::AtanGrad},
              {prim::kPrimAsinhGrad->name(), &EltWiseGradCpuKernelMod<T>::AsinhGrad},
              {prim::kPrimAcoshGrad->name(), &EltWiseGradCpuKernelMod<T>::AcoshGrad},
              {prim::kPrimAbsGrad->name(), &EltWiseGradCpuKernelMod<T>::AbsGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "EltWiseGradCpuKernelMod does not support " << kernel_name_ << " with double as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
    return;
  }
  if constexpr (std::is_same_v<T, float>) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuKernelMod *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimReluGrad->name(), &EltWiseGradCpuKernelMod<T>::ReluGrad},
              {prim::kPrimRelu6Grad->name(), &EltWiseGradCpuKernelMod<T>::ReLU6Grad},
              {prim::kPrimSigmoidGrad->name(), &EltWiseGradCpuKernelMod<T>::SigmoidGrad},
              {prim::kPrimAbsGrad->name(), &EltWiseGradCpuKernelMod<T>::AbsGrad},
              {prim::kPrimTanhGrad->name(), &EltWiseGradCpuKernelMod<T>::TanhGrad},
              {prim::kPrimSqrtGrad->name(), &EltWiseGradCpuKernelMod<T>::SqrtGrad},
              {prim::kPrimGeLUGrad->name(), &EltWiseGradCpuKernelMod<T>::GeluGrad},
              {prim::kPrimAsinGrad->name(), &EltWiseGradCpuKernelMod<T>::AsinGrad},
              {prim::kPrimACosGrad->name(), &EltWiseGradCpuKernelMod<T>::ACosGrad},
              {prim::kPrimAtanGrad->name(), &EltWiseGradCpuKernelMod<T>::AtanGrad},
              {prim::kPrimAsinhGrad->name(), &EltWiseGradCpuKernelMod<T>::AsinhGrad},
              {prim::kPrimAcoshGrad->name(), &EltWiseGradCpuKernelMod<T>::AcoshGrad},
              {prim::kPrimSoftplusGrad->name(), &EltWiseGradCpuKernelMod<T>::SoftplusGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "EltWiseGradCpuKernelMod does not support " << kernel_name_ << " with float as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
    return;
  }
  if constexpr (std::is_same_v<T, int>) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuKernelMod *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimAbsGrad->name(), &EltWiseGradCpuKernelMod<T>::AbsGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "EltWiseGradCpuKernelMod does not support " << kernel_name_ << " with int as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
  }
  if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuKernelMod *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimAcoshGrad->name(), &EltWiseGradCpuKernelMod<T>::ComplexAcoshGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "EltWiseGradCpuKernelMod does not support " << kernel_name_;
    }
    compute_func_ = elt_map.at(kernel_name_);
  }
}

template <typename T>
void EltWiseGradCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  InitComputeFunc();
}

template <typename T>
bool EltWiseGradCpuKernelMod<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < kInputMinNum || outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it requires at least 2 inputs and 1 output, but got "
                  << inputs.size() << " input(s) and " << outputs.size() << " output(s).";
    return false;
  }
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the memory size of output should be greater than 0, but got 0.";
    return true;
  }
  const auto input0 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto input1 = reinterpret_cast<T *>(inputs[1]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);

  ParallelLaunchAutoSearch(
    std::bind(compute_func_, this, input0, input1, output, std::placeholders::_1, std::placeholders::_2),
    outputs[0]->size / sizeof(T), this, &parallel_search_info_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
