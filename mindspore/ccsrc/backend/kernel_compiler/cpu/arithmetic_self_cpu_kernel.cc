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
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <map>
#include "backend/kernel_compiler/cpu/arithmetic_self_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
void Square(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = in[i] * in[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, size, MAX_SQUARE_SERIAL_SIZE);
}

template <typename T>
void Sign(const T *in, T *out, size_t size) {
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
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Neg(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = -in[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, size, MAX_NEG_SERIAL_SIZE);
}

template <typename T>
void LogicalNot(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = !in[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void OnesLike(const T *, T *out, size_t size) {
  auto task = [&out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(1);
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ZerosLike(const T *, T *out, size_t size) {
  auto task = [&out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(0);
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Floor(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(floor(in[i]));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Rint(const T *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(rint(in[i]));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Round(const T *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(nearbyint(in[i]));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Reciprocal(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(1.0 / in[i]);
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Gelu(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      T x = in[i];
      auto double_x = static_cast<T>(x);
      T tanh_res = static_cast<T>(std::tanh(0.7978845608 * (double_x + 0.044715 * double_x * double_x * double_x)));
      out[i] = x * (static_cast<T>(1.0) + tanh_res) / static_cast<T>(2.0);
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Asin(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asin(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void ACos(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acos(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Atan(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atan(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Sin(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sin(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Cos(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cos(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Tan(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(tan(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Sinh(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sinh(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Cosh(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cosh(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Asinh(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asinh(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Acosh(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acosh(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Atanh(const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atanh(static_cast<double>(in[i])));
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename T>
void Identity(const T *in, T *out, size_t size) {
  std::copy(in, in + size, out);
}
}  // namespace

static const std::map<std::string, OperateType> kArithmeticOpTypeMap = {{prim::kPrimNeg->name(), NEG},
                                                                        {prim::kPrimSquare->name(), SQUARE},
                                                                        {prim::kPrimOnesLike->name(), ONESLIKE},
                                                                        {prim::kPrimZerosLike->name(), ZEROSLIKE},
                                                                        {prim::kPrimLogicalNot->name(), LOGICALNOT},
                                                                        {prim::kPrimSign->name(), SIGN},
                                                                        {prim::kPrimFloor->name(), FLOOR},
                                                                        {prim::kPrimRint->name(), RINT},
                                                                        {prim::kPrimRound->name(), ROUND},
                                                                        {prim::kPrimReciprocal->name(), RECIPROCAL},
                                                                        {prim::kPrimGeLU->name(), GELU},
                                                                        {prim::kPrimAsin->name(), ASIN},
                                                                        {prim::kPrimACos->name(), ACOS},
                                                                        {prim::kPrimAtan->name(), ATAN},
                                                                        {prim::kPrimSin->name(), SIN},
                                                                        {prim::kPrimCos->name(), COS},
                                                                        {prim::kPrimTan->name(), TAN},
                                                                        {prim::kPrimSinh->name(), SINH},
                                                                        {prim::kPrimCosh->name(), COSH},
                                                                        {prim::kPrimAsinh->name(), ASINH},
                                                                        {prim::kPrimAcosh->name(), ACOSH},
                                                                        {prim::kPrimAtanh->name(), ATANH},
                                                                        {prim::kPrimIdentityMath->name(), IDENTITY}};

void ArithmeticSelfCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  operate_type_ = kArithmeticOpTypeMap.at(kernel_name);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool ArithmeticSelfCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32 || dtype_ == kNumberTypeInt16) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    LaunchKernelLogic<bool>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

template <typename T>
void ArithmeticSelfCPUKernel::LaunchKernelLogic(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  LogicalNot<T>(input, output, lens);
  return;
}

template <typename T>
void ArithmeticSelfCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  static const std::map<OperateType, std::function<void(const T *in, T *out, size_t size)>> kArithmeticOpFuncMap = {
    {SQUARE, Square<T>},     {SIGN, Sign<T>},
    {NEG, Neg<T>},           {LOGICALNOT, LogicalNot<T>},
    {ONESLIKE, OnesLike<T>}, {ZEROSLIKE, ZerosLike<T>},
    {FLOOR, Floor<T>},       {RECIPROCAL, Reciprocal<T>},
    {GELU, Gelu<T>},         {SIN, Sin<T>},
    {COS, Cos<T>},           {TAN, Tan<T>},
    {ASIN, Asin<T>},         {ACOS, ACos<T>},
    {ATAN, Atan<T>},         {SINH, Sinh<T>},
    {COSH, Cosh<T>},         {ASINH, Asinh<T>},
    {ACOSH, Acosh<T>},       {ATANH, Atanh<T>},
    {RINT, Rint<T>},         {ROUND, Round<T>}};
  if (kArithmeticOpFuncMap.find(operate_type_) != kArithmeticOpFuncMap.end()) {
    kArithmeticOpFuncMap.at(operate_type_)(input, output, lens);
  } else {
    MS_LOG(EXCEPTION) << "Not support " << operate_type_;
  }
}

template <typename T>
bool IdentityCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  Identity<T>(input, output, lens);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
