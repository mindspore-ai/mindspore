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

#include "plugin/device/cpu/kernel/arithmetic_self_cpu_kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <thread>
#include <unordered_map>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr float kMaxNegSerialSize = 5000.0f;
constexpr float kMaxSquareSerialSize = 5000.0f;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;

template <typename T>
void Square(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = in[i] * in[i];
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sign(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if (in[i] < 0) {
        out[i] = -1;
      } else if (in[i] > 0) {
        out[i] = 1;
      } else {
        out[i] = 0;
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Neg(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = -in[i];
    }
  };
  ParallelLaunch(task, size, kMaxNegSerialSize);
}

void LogicalNot(ArithmeticSelfCpuKernelMod *content, const bool *in, bool *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = !in[i];
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Floor(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(floor(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Rint(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(rint(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Round(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(nearbyint(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Reciprocal(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(1.0 / in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Gelu(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    auto factor_a = static_cast<T>(0.7978845608);
    auto factor_b = static_cast<T>(0.044715);
    for (size_t i = start; i < end; i++) {
      T x = in[i];
      auto double_x = static_cast<T>(x);
      T tanh_res = static_cast<T>(std::tanh(factor_a * (double_x + factor_b * double_x * double_x * double_x)));
      out[i] = x * (static_cast<T>(1.0) + tanh_res) / static_cast<T>(2.0);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Asin(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asin(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ACos(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Atan(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atan(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sin(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sin(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Cos(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Tan(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(tan(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sinh(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Cosh(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Asinh(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexAcosh(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acosh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Acosh(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Atanh(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atanh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Abs(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = abs(in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sqrt(ArithmeticSelfCpuKernelMod *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sqrt(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Identity(const T *in, T *out, size_t size) {
  (void)std::copy(in, in + size, out);
}
}  // namespace

void ArithmeticSelfCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool ArithmeticSelfCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchKernelComplex<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchKernelComplex<std::complex<double>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32 || dtype_ == kNumberTypeInt16) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    LaunchLogicalNot(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'x' should be float16, float32, float64, int16, int32, int64, or bool, "
                         "but got "
                      << TypeIdLabel(dtype_);
  }
  return true;
}

void ArithmeticSelfCpuKernelMod::LaunchLogicalNot(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &outputs) {
  auto *input = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *output = reinterpret_cast<bool *>(outputs[0]->addr);
  size_t lens = outputs[0]->size / sizeof(bool);
  LogicalNot(this, input, output, lens);
}

template <typename T>
void ArithmeticSelfCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t lens = outputs[0]->size / sizeof(T);
  static const std::unordered_map<std::string,
                                  std::function<void(ArithmeticSelfCpuKernelMod *, const T *, T *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimSquare->name(), Square<T>},
                          {prim::kPrimSign->name(), Sign<T>},
                          {prim::kPrimNeg->name(), Neg<T>},
                          {prim::kPrimAtanh->name(), Atanh<T>},
                          {prim::kPrimAcosh->name(), Acosh<T>},
                          {prim::kPrimFloor->name(), Floor<T>},
                          {prim::kPrimSin->name(), Sin<T>},
                          {prim::kPrimGeLU->name(), Gelu<T>},
                          {prim::kPrimCos->name(), Cos<T>},
                          {prim::kPrimTan->name(), Tan<T>},
                          {prim::kPrimAsin->name(), Asin<T>},
                          {prim::kPrimACos->name(), ACos<T>},
                          {prim::kPrimAtan->name(), Atan<T>},
                          {prim::kPrimSinh->name(), Sinh<T>},
                          {prim::kPrimCosh->name(), Cosh<T>},
                          {prim::kPrimAsinh->name(), Asinh<T>},
                          {prim::kPrimReciprocal->name(), Reciprocal<T>},
                          {prim::kPrimRint->name(), Rint<T>},
                          {prim::kPrimRound->name(), Round<T>},
                          {prim::kPrimAbs->name(), Abs<T>},
                          {prim::kPrimSqrt->name(), Sqrt<T>}};

  const auto func_pair = arithmeticSelfFuncMap.find(kernel_name_);
  if (arithmeticSelfFuncMap.find(kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For 'Arithmetic', only supports operators in " << Unorderedmap2Str(arithmeticSelfFuncMap)
                      << ", but got " << kernel_name_;
  }
  func_pair->second(this, input, output, lens);
}

template <typename T>
void ArithmeticSelfCpuKernelMod::LaunchKernelComplex(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t lens = outputs[0]->size / sizeof(T);
  static const std::unordered_map<std::string,
                                  std::function<void(ArithmeticSelfCpuKernelMod *, const T *, T *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimSquare->name(), Square<T>},
                          {prim::kPrimAcosh->name(), ComplexAcosh<T>},
                          {prim::kPrimNeg->name(), Neg<T>}};
  const auto func_pair = arithmeticSelfFuncMap.find(kernel_name_);
  if (arithmeticSelfFuncMap.find(kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "ArithmeticSelfCpuKernelMod does not support " << kernel_name_;
  }
  func_pair->second(this, input, output, lens);
}

template <typename T>
bool IdentityCpuKernelMod<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  Identity<T>(input, output, lens);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
