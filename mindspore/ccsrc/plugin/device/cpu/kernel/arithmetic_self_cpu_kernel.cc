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
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/mkldnn/eltwise_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

constexpr float kMaxNegSerialSize = 5000.0f;
constexpr float kMaxSquareSerialSize = 5000.0f;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;

constexpr auto kSquare = "Square";
constexpr auto kNeg = "Neg";
constexpr auto kSign = "Sign";
constexpr auto kFloor = "Floor";
constexpr auto kRint = "Rint";
constexpr auto kRound = "Round";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kGeLU = "GeLU";
constexpr auto kLogicalNot = "LogicalNot";
constexpr auto kAsin = "Asin";
constexpr auto kACos = "ACos";
constexpr auto kAtan = "Atan";
constexpr auto kSin = "Sin";
constexpr auto kCos = "Cos";
constexpr auto kTan = "Tan";
constexpr auto kSinh = "Sinh";
constexpr auto kCosh = "Cosh";
constexpr auto kAsinh = "Asinh";
constexpr auto kAcosh = "Acosh";
constexpr auto kAtanh = "Atanh";
constexpr auto kAbs = "Abs";
constexpr auto kSqrt = "Sqrt";
constexpr auto kRsqrt = "Rsqrt";

class ArithmeticSelfCpuKernelFunc : public CpuKernelFunc {
 public:
  ArithmeticSelfCpuKernelFunc() = default;
  ~ArithmeticSelfCpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  void LaunchLogicalNot(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void LaunchKernelComplex(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  TypeId dtype_{kTypeUnknown};
  std::string kernel_name_;
};

template <typename T>
void Square(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = in[i] * in[i];
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sign(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
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
void Neg(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = -in[i];
    }
  };
  ParallelLaunch(task, size, kMaxNegSerialSize);
}

void LogicalNot(ArithmeticSelfCpuKernelFunc *content, const bool *in, bool *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = !in[i];
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Floor(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(floor(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Rint(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(rint(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Round(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(nearbyint(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Reciprocal(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(1.0 / in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Gelu(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
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
void Asin(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asin(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ACos(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Atan(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atan(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sin(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sin(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Cos(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexSin(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sin(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexSinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sinh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexCos(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cos(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexCosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cosh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Tan(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(tan(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Cosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexAsinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asinh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Asinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexAcosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acosh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Acosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Atanh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atanh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Abs(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = abs(in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sqrt(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sqrt(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Rsqrt(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(1) / sqrt(in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Identity(const T *in, T *out, size_t size) {
  (void)std::copy(in, in + size, out);
}

template <typename T>
bool IdentityCpuFunc(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  Identity<T>(input, output, lens);
  return true;
}

static std::vector<std::pair<KernelAttr, LaunchFunc>> identity_kernel_attr_lists = {
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64), IdentityCpuFunc<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), IdentityCpuFunc<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32), IdentityCpuFunc<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), IdentityCpuFunc<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16), IdentityCpuFunc<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), IdentityCpuFunc<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), IdentityCpuFunc<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), IdentityCpuFunc<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), IdentityCpuFunc<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), IdentityCpuFunc<complex128>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), IdentityCpuFunc<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), IdentityCpuFunc<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), IdentityCpuFunc<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), IdentityCpuFunc<bool>}};

void ArithmeticSelfCpuKernelFunc::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool ArithmeticSelfCpuKernelFunc::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
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

void ArithmeticSelfCpuKernelFunc::LaunchLogicalNot(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &outputs) {
  auto *input = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *output = reinterpret_cast<bool *>(outputs[0]->addr);
  size_t lens = outputs[0]->size / sizeof(bool);
  LogicalNot(this, input, output, lens);
}

template <typename T>
void ArithmeticSelfCpuKernelFunc::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t lens = outputs[0]->size / sizeof(T);
  static const std::unordered_map<std::string,
                                  std::function<void(ArithmeticSelfCpuKernelFunc *, const T *, T *, size_t)>>
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
                          {prim::kPrimSqrt->name(), Sqrt<T>},
                          {prim::kPrimRsqrt->name(), Rsqrt<T>}};

  const auto func_pair = arithmeticSelfFuncMap.find(kernel_name_);
  if (arithmeticSelfFuncMap.find(kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For 'Arithmetic', only supports operators in " << Unorderedmap2Str(arithmeticSelfFuncMap)
                      << ", but got " << kernel_name_;
  }
  func_pair->second(this, input, output, lens);
}

template <typename T>
void ArithmeticSelfCpuKernelFunc::LaunchKernelComplex(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t lens = outputs[0]->size / sizeof(T);
  static const std::unordered_map<std::string,
                                  std::function<void(ArithmeticSelfCpuKernelFunc *, const T *, T *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimSquare->name(), Square<T>},      {prim::kPrimAcosh->name(), ComplexAcosh<T>},
                          {prim::kPrimAsinh->name(), ComplexAsinh<T>}, {prim::kPrimNeg->name(), Neg<T>},
                          {prim::kPrimSinh->name(), ComplexSinh<T>},   {prim::kPrimCosh->name(), ComplexCosh<T>},
                          {prim::kPrimSin->name(), ComplexSin<T>},     {prim::kPrimCos->name(), ComplexCos<T>},
                          {prim::kPrimRsqrt->name(), Rsqrt<T>}};
  const auto func_pair = arithmeticSelfFuncMap.find(kernel_name_);
  if (arithmeticSelfFuncMap.find(kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "ArithmeticSelfCpuKernelFunc does not support " << kernel_name_;
  }
  func_pair->second(this, input, output, lens);
}

// MKLDNN Sqrt
class SqrtMKLKernelFunc : public CpuKernelFunc, private EltWiseCpuKernelMod {
 public:
  SqrtMKLKernelFunc() : EltWiseCpuKernelMod(kSqrt) {}
  ~SqrtMKLKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name != kSqrt) {
      MS_LOG(EXCEPTION) << "Should be " << kSqrt << ", but got " << kernel_name;
    }
    EltWiseCpuKernelMod::InitKernel(kernel_node);
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override {
    return EltWiseCpuKernelMod::Launch(inputs, workspace, outputs);
  }
};

std::shared_ptr<CpuKernelFunc> CreateArithSelfFunc() { return std::make_shared<ArithmeticSelfCpuKernelFunc>(); }
using ArithFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, ArithFuncCreator>>> arith_kernel_attr_list_map = {
  {kRsqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kSquare,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kNeg,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kSign,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kFloor,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kRint,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kRound,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kReciprocal,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kGeLU, {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc}}},
  {kLogicalNot, {{KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), CreateArithSelfFunc}}},
  {kAsin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kACos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kAtan,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kSin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kCos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kTan,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kSinh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kCosh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kAsinh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kAcosh,
   {{KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kAtanh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kAbs,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kSqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<SqrtMKLKernelFunc>(); }}}}};
}  // namespace

void ArithmeticSelfCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "ArithmeticSelf cpu does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = arith_kernel_attr_list_map[kernel_name_][index].second();
  func_obj_->InitFunc(kernel_node);
}

std::vector<KernelAttr> ArithmeticSelfCpuKernelMod::GetOpSupport() {
  auto iter = arith_kernel_attr_list_map.find(kernel_type_);
  if (iter == arith_kernel_attr_list_map.end()) {
    MS_LOG(EXCEPTION) << "Arithmetic self cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ArithFuncCreator> &pair) { return pair.first; });

  return support_list;
}

void IdentityCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Identity cpu does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = identity_kernel_attr_lists[index].second;
}

std::vector<KernelAttr> IdentityCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> kernel_attr_list;
  (void)std::transform(identity_kernel_attr_lists.begin(), identity_kernel_attr_lists.end(),
                       std::back_inserter(kernel_attr_list),
                       [](const std::pair<KernelAttr, LaunchFunc> &pair) { return pair.first; });
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Rsqrt,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kRsqrt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Square,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSquare); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Neg,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kNeg); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sign,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSign); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Floor,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kFloor); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Rint,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kRint); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Round,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kRound); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Reciprocal,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kReciprocal); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, GeLU,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kGeLU); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LogicalNot,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kLogicalNot); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Asin,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAsin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ACos,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kACos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Atan,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAtan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sin,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Cos,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kCos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Tan,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kTan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sinh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSinh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Cosh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kCosh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Asinh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAsinh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Acosh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAcosh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Atanh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAtanh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Abs,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAbs); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sqrt,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSqrt); });

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Identity, IdentityCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
