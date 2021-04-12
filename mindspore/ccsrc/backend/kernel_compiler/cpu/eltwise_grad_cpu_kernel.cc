/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <map>
#include "backend/kernel_compiler/cpu/eltwise_grad_cpu_kernel.h"
#include "common/thread_pool.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {

template <typename T>
void EltWiseGradCPUKernel<T>::ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    if (input2[i] > 0) {
      out[i] = input1[i];
    } else {
      out[i] = 0;
    }
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    if (input2[i] > 0 && input2[i] <= 6) {
      out[i] = input1[i];
    } else {
      out[i] = 0;
    }
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    if (input1[i] > 0) {
      out[i] = input2[i];
    } else if (input1[i] < 0) {
      out[i] = -input2[i];
    } else {
      out[i] = 0;
    }
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = input2[i] * input1[i] * (1 - input1[i]);
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = input2[i] / (input1[i] * 2);
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    T tmp = input1[i] * input1[i];
    out[i] = input2[i] * (1 - tmp);
  }
}

template <typename T>
void EltWiseGradCPUKernel<T>::GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
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
void EltWiseGradCPUKernel<T>::AsinGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
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
void EltWiseGradCPUKernel<T>::ACosGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
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
void EltWiseGradCPUKernel<T>::AtanGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    const T divisor = 1 + input1[i] * input1[i];
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
void EltWiseGradCPUKernel<T>::AsinhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = sqrt(1 + input1[i] * input1[i]);
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
void EltWiseGradCPUKernel<T>::AcoshGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = sqrt(input1[i] * input1[i] - 1);
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
void EltWiseGradCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
}

template <typename T>
bool EltWiseGradCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> & /*workspace*/,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  static const std::map<std::string,
                        std::function<void(EltWiseGradCPUKernel *, const T *, const T *, T *, size_t, size_t)>>
    elt_map{{"ReluGrad", &EltWiseGradCPUKernel<T>::ReluGrad},       {"ReLU6Grad", &EltWiseGradCPUKernel<T>::ReLU6Grad},
            {"SigmoidGrad", &EltWiseGradCPUKernel<T>::SigmoidGrad}, {"AbsGrad", &EltWiseGradCPUKernel<T>::AbsGrad},
            {"TanhGrad", &EltWiseGradCPUKernel<T>::TanhGrad},       {"SqrtGrad", &EltWiseGradCPUKernel<T>::SqrtGrad},
            {"GeLUGrad", &EltWiseGradCPUKernel<T>::GeluGrad},       {"AsinGrad", &EltWiseGradCPUKernel<T>::AsinGrad},
            {"ACosGrad", &EltWiseGradCPUKernel<T>::ACosGrad},       {"AtanGrad", &EltWiseGradCPUKernel<T>::AtanGrad},
            {"AsinhGrad", &EltWiseGradCPUKernel<T>::AsinhGrad},     {"AcoshGrad", &EltWiseGradCPUKernel<T>::AcoshGrad}};
  T *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);

  size_t count = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  const float block_size = 128.0;
  size_t thread_num = count < block_size * max_thread_num ? std::ceil(count / block_size) : max_thread_num;
  std::vector<common::Task> tasks;
  size_t start = 0;
  size_t once_compute_size = (count + thread_num - 1) / thread_num;
  while (start < count) {
    size_t end = (start + once_compute_size) > count ? count : (start + once_compute_size);
    auto block = [&, start, end]() {
      elt_map.at(kernel_name_)(this, input1, input2, output, start, end);
      return common::SUCCESS;
    };
    tasks.emplace_back(block);
    start += once_compute_size;
  }
  common::ThreadPool::GetInstance().SyncRun(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
