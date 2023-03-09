/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/scalar_arithmetic_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kScalarAdd = "ScalarAdd";
constexpr auto kScalarSub = "ScalarSub";
constexpr auto kScalarMul = "ScalarMul";
constexpr auto kScalarDiv = "ScalarDiv";
constexpr auto kScalarFloordiv = "ScalarFloordiv";
constexpr auto kScalarMod = "ScalarMod";
constexpr auto kScalarGt = "scalar_gt";
constexpr auto kScalarGe = "scalar_ge";
constexpr auto kScalarLt = "scalar_lt";
constexpr auto kScalarLe = "scalar_le";
constexpr auto kScalarEq = "scalar_eq";
constexpr size_t kInputNum = 2;
constexpr size_t kInputx = 0;
constexpr size_t kInputy = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

template <typename T, typename S, typename N>
void AddImpl(const T *in_x, const S *in_y, N *out) {
  N x = static_cast<N>(*in_x);
  N y = static_cast<N>(*in_y);
#ifndef _MSC_VER
  if constexpr (std::is_integral<N>::value && std::is_signed<N>::value) {
    if (__builtin_add_overflow(x, y, out)) {
      MS_EXCEPTION(ValueError) << "For prim ScalarAdd Overflow of the sum of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return;
  }
#endif
  *out = x + y;
}

template <typename T, typename S, typename N>
void SubImpl(const T *in_x, const S *in_y, N *out) {
  N x = static_cast<N>(*in_x);
  N y = static_cast<N>(*in_y);
#ifndef _MSC_VER
  if constexpr (std::is_integral<N>::value && std::is_signed<N>::value) {
    if (__builtin_sub_overflow(x, y, out)) {
      MS_EXCEPTION(ValueError) << "For prim ScalarSub Overflow of the sub of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return;
  }
#endif
  *out = x - y;
}

template <typename T, typename S, typename N>
void MulImpl(const T *in_x, const S *in_y, N *out) {
  N x = static_cast<N>(*in_x);
  N y = static_cast<N>(*in_y);

#ifndef _MSC_VER
  if constexpr (std::is_integral<N>::value && std::is_signed<N>::value) {
    if (__builtin_mul_overflow(x, y, out)) {
      MS_EXCEPTION(ValueError) << "For prim ScalarMul Overflow of the mul of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return;
  }
#endif
  auto res = static_cast<double>(x) * static_cast<double>(y);
  *out = static_cast<N>(res);
}

template <typename T, typename S, typename N>
void DivImpl(const T *in_x, const S *in_y, N *out) {
  N x = static_cast<N>(*in_x);
  N y = static_cast<N>(*in_y);
  N zero = 0;
  if (y == zero) {
    MS_EXCEPTION(ValueError) << "The divisor could not be zero. But the divisor is zero now.";
  }
  if constexpr (std::is_integral<N>::value && std::is_signed<N>::value) {
    if (x == std::numeric_limits<N>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "For prim ScalarDiv Overflow of the div of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  *out = x / y;
}

template <typename T, typename S, typename N>
void FloorDivImpl(const T *in_x, const S *in_y, N *out) {
  N x = static_cast<N>(*in_x);
  N y = static_cast<N>(*in_y);
  N zero = 0;
  if (y == zero) {
    MS_EXCEPTION(ValueError) << "The divisor could not be zero. But the divisor is zero now.";
  }
  if constexpr (std::is_integral<N>::value && std::is_signed<N>::value) {
    if (x == std::numeric_limits<N>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "For prim ScalarDiv Overflow of the div of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  T n = std::floor(static_cast<float>(x) / static_cast<float>(y));
  auto mod = x - n * y;
  *out = (x - mod) / y;
}

template <typename T, typename S, typename N>
void ModImpl(const T *in_x, const S *in_y, N *out) {
  N x = static_cast<N>(*in_x);
  N y = static_cast<N>(*in_y);
  N zero = 0;
  if (y == zero) {
    MS_EXCEPTION(ValueError) << "Cannot perform modulo operation on zero.";
  }
  if constexpr (std::is_integral<N>::value && std::is_signed<N>::value) {
    if (x == std::numeric_limits<N>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "For prim ScalarDiv Overflow of the div of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  T n = std::floor(static_cast<float>(x) / static_cast<float>(y));
  *out = x - n * y;
}

template <typename T, typename S, typename N>
void EqImpl(const T *in_x, const S *in_y, N *out) {
  double x = static_cast<double>(*in_x);
  double y = static_cast<double>(*in_y);
  if (std::isinf(static_cast<double>(x)) && std::isinf(static_cast<double>(y))) {
    *out = static_cast<N>((x > 0 && y > 0) || (x < 0 && y < 0));
    return;
  }
  double error_abs = fabs(x - y);
  *out = static_cast<N>(error_abs < DBL_EPSILON);
}

template <typename T, typename S, typename N>
void LtImpl(const T *in_x, const S *in_y, N *out) {
  double x = static_cast<double>(*in_x);
  double y = static_cast<double>(*in_y);
  *out = static_cast<N>(x < y);
}

template <typename T, typename S, typename N>
void LeImpl(const T *in_x, const S *in_y, N *out) {
  double x = static_cast<double>(*in_x);
  double y = static_cast<double>(*in_y);
  *out = static_cast<N>(x <= y);
}

template <typename T, typename S, typename N>
void GtImpl(const T *in_x, const S *in_y, N *out) {
  double x = static_cast<double>(*in_x);
  double y = static_cast<double>(*in_y);
  *out = static_cast<N>(x > y);
}

template <typename T, typename S, typename N>
void GeImpl(const T *in_x, const S *in_y, N *out) {
  double x = static_cast<double>(*in_x);
  double y = static_cast<double>(*in_y);
  *out = static_cast<N>(x >= y);
}

bool ScalarArithmeticCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For kernel '" << kernel_type_ << "' input_num must be 2, but got " << inputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  if (kernel_type_ == kScalarDiv) {
    kernel_func_ = div_func_list_[index].second;
  } else if (is_logic_ops_) {
    kernel_func_ = logic_func_list_[index].second;
  } else {
    kernel_func_ = math_func_list_[index].second;
  }
  return true;
}

int ScalarArithmeticCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  return KRET_OK;
}

template <typename T, typename S, typename N>
bool ScalarArithmeticCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);

  using MathImplFunc = std::function<void(const T *x, const S *y, N *out)>;
  std::unordered_map<std::string, MathImplFunc> func_map = {{kScalarAdd, AddImpl<T, S, N>},
                                                            {kScalarSub, SubImpl<T, S, N>},
                                                            {kScalarMul, MulImpl<T, S, N>},
                                                            {kScalarDiv, DivImpl<T, S, N>},
                                                            {kScalarMod, ModImpl<T, S, N>},
                                                            {kScalarEq, EqImpl<T, S, N>},
                                                            {kScalarGt, GtImpl<T, S, N>},
                                                            {kScalarLt, LtImpl<T, S, N>},
                                                            {kScalarGe, GeImpl<T, S, N>},
                                                            {kScalarLe, LeImpl<T, S, N>},
                                                            {kScalarFloordiv, FloorDivImpl<T, S, N>}};
  auto iter = func_map.find(kernel_name_);
  if (iter == func_map.end()) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "' don't support. Only support [Add, Sub, Mul, Div, Mod, Eq, Le, Ge, Lt, Gt]";
  }
  MathImplFunc compute_func = iter->second;

  T *input_x = GetDeviceAddress<T>(inputs, kInputx);
  S *input_y = GetDeviceAddress<S>(inputs, kInputy);
  N *output = GetDeviceAddress<N>(outputs, 0);
  compute_func(input_x, input_y, output);
  return true;
}

#define ADD_KERNEL(x_dtype, y_dtype, out_dtype, x_type, y_type, out_type)   \
  {                                                                         \
    KernelAttr()                                                            \
      .AddInputAttr(kObjectTypeNumber, kNumberType##x_dtype)                \
      .AddInputAttr(kObjectTypeNumber, kNumberType##y_dtype)                \
      .AddOutputAttr(kObjectTypeNumber, kNumberType##out_dtype),            \
      &ScalarArithmeticCpuKernelMod::LaunchKernel<x_type, y_type, out_type> \
  }

std::vector<std::pair<KernelAttr, ScalarArithmeticCpuKernelMod::ScalarArithmeticFunc>>
  ScalarArithmeticCpuKernelMod::math_func_list_ = {
    ADD_KERNEL(Float32, Float32, Float32, float, float, float),
    ADD_KERNEL(Float32, Float64, Float64, float, double, double),
    ADD_KERNEL(Float32, Int32, Float32, float, int32_t, float),
    ADD_KERNEL(Float32, Int64, Float32, float, int64_t, float),
    ADD_KERNEL(Float32, Bool, Float32, float, bool, float),
    ADD_KERNEL(Float64, Float64, Float64, double, double, double),
    ADD_KERNEL(Float64, Float32, Float64, double, float, double),
    ADD_KERNEL(Float64, Int64, Float64, double, int64_t, double),
    ADD_KERNEL(Float64, Int32, Float64, double, int32_t, double),
    ADD_KERNEL(Float64, Bool, Float64, double, bool, double),
    ADD_KERNEL(Int32, Float32, Float32, int32_t, float, float),
    ADD_KERNEL(Int32, Float64, Float64, int32_t, double, double),
    ADD_KERNEL(Int32, Int32, Int32, int32_t, int32_t, int32_t),
    ADD_KERNEL(Int32, Int64, Int64, int32_t, int64_t, int64_t),
    ADD_KERNEL(Int32, Bool, Int32, int32_t, bool, int32_t),
    ADD_KERNEL(Int64, Float64, Float64, int64_t, double, double),
    ADD_KERNEL(Int64, Float32, Float32, int64_t, float, float),
    ADD_KERNEL(Int64, Int64, Int64, int64_t, int64_t, int64_t),
    ADD_KERNEL(Int64, Int32, Int64, int64_t, int32_t, int64_t),
    ADD_KERNEL(Int64, Bool, Int64, int64_t, bool, int64_t),
    ADD_KERNEL(Bool, Float32, Float32, bool, float, float),
    ADD_KERNEL(Bool, Float64, Float64, bool, double, double),
    ADD_KERNEL(Bool, Int32, Int32, bool, int32_t, int32_t),
    ADD_KERNEL(Bool, Int64, Int64, bool, int64_t, int64_t),
    ADD_KERNEL(Bool, Bool, Int32, bool, bool, int32_t),
};

std::vector<std::pair<KernelAttr, ScalarArithmeticCpuKernelMod::ScalarArithmeticFunc>>
  ScalarArithmeticCpuKernelMod::div_func_list_ = {
    ADD_KERNEL(Float32, Float32, Float32, float, float, float),
    ADD_KERNEL(Float32, Float64, Float32, float, double, float),
    ADD_KERNEL(Float32, Int32, Float32, float, int32_t, float),
    ADD_KERNEL(Float32, Int64, Float32, float, int64_t, float),
    ADD_KERNEL(Float32, Bool, Float32, float, bool, float),
    ADD_KERNEL(Float64, Float64, Float32, double, double, float),
    ADD_KERNEL(Float64, Float32, Float32, double, float, float),
    ADD_KERNEL(Float64, Int64, Float32, double, int64_t, float),
    ADD_KERNEL(Float64, Int32, Float32, double, int32_t, float),
    ADD_KERNEL(Float64, Bool, Float32, double, bool, float),
    ADD_KERNEL(Int32, Float32, Float32, int32_t, float, float),
    ADD_KERNEL(Int32, Float64, Float32, int32_t, double, float),
    ADD_KERNEL(Int32, Int32, Float32, int32_t, int32_t, float),
    ADD_KERNEL(Int32, Int64, Float32, int32_t, int64_t, float),
    ADD_KERNEL(Int32, Bool, Float32, int32_t, bool, float),
    ADD_KERNEL(Int64, Float64, Float32, int64_t, double, float),
    ADD_KERNEL(Int64, Float32, Float32, int64_t, float, float),
    ADD_KERNEL(Int64, Int64, Float32, int64_t, int64_t, float),
    ADD_KERNEL(Int64, Int32, Float32, int64_t, int32_t, float),
    ADD_KERNEL(Int64, Bool, Float32, int64_t, bool, float),
    ADD_KERNEL(Bool, Float64, Float32, bool, double, float),
    ADD_KERNEL(Bool, Float32, Float32, bool, float, float),
    ADD_KERNEL(Bool, Int64, Float32, bool, int64_t, float),
    ADD_KERNEL(Bool, Int32, Float32, bool, int32_t, float),
    ADD_KERNEL(Bool, Bool, Float32, bool, bool, float),
};

std::vector<std::pair<KernelAttr, ScalarArithmeticCpuKernelMod::ScalarArithmeticFunc>>
  ScalarArithmeticCpuKernelMod::logic_func_list_ = {ADD_KERNEL(Float32, Float32, Bool, float, float, bool),
                                                    ADD_KERNEL(Float32, Float64, Bool, float, double, bool),
                                                    ADD_KERNEL(Float32, Int32, Bool, float, int32_t, bool),
                                                    ADD_KERNEL(Float32, Int64, Bool, float, int64_t, bool),
                                                    ADD_KERNEL(Float32, Bool, Bool, float, bool, bool),
                                                    ADD_KERNEL(Float64, Bool, Bool, double, bool, bool),
                                                    ADD_KERNEL(Float64, Float64, Bool, double, double, bool),
                                                    ADD_KERNEL(Float64, Float32, Bool, double, float, bool),
                                                    ADD_KERNEL(Float64, Int64, Bool, double, int64_t, bool),
                                                    ADD_KERNEL(Float64, Int32, Bool, double, int32_t, bool),
                                                    ADD_KERNEL(Int32, Float32, Bool, int32_t, float, bool),
                                                    ADD_KERNEL(Int32, Float64, Bool, int32_t, double, bool),
                                                    ADD_KERNEL(Int32, Int32, Bool, int32_t, int32_t, bool),
                                                    ADD_KERNEL(Int32, Int64, Bool, int32_t, int64_t, bool),
                                                    ADD_KERNEL(Int32, Bool, Bool, int32_t, bool, bool),
                                                    ADD_KERNEL(Int64, Bool, Bool, int64_t, bool, bool),
                                                    ADD_KERNEL(Int64, Float64, Bool, int64_t, double, bool),
                                                    ADD_KERNEL(Int64, Float32, Bool, int64_t, float, bool),
                                                    ADD_KERNEL(Int64, Int64, Bool, int64_t, int64_t, bool),
                                                    ADD_KERNEL(Int64, Int32, Bool, int64_t, int32_t, bool),
                                                    ADD_KERNEL(Bool, Float64, Bool, bool, double, bool),
                                                    ADD_KERNEL(Bool, Float32, Bool, bool, float, bool),
                                                    ADD_KERNEL(Bool, Int64, Bool, bool, int64_t, bool),
                                                    ADD_KERNEL(Bool, Int32, Bool, bool, int32_t, bool),
                                                    ADD_KERNEL(Bool, Bool, Bool, bool, bool, bool)};

std::vector<KernelAttr> ScalarArithmeticCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::set<std::string> logic_ops = {kScalarEq, kScalarGe, kScalarGt, kScalarLt, kScalarLe};
  auto iter = logic_ops.find(kernel_type_);
  if (kernel_type_ == kScalarDiv) {
    (void)std::transform(div_func_list_.begin(), div_func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, ScalarArithmeticFunc> &item) { return item.first; });
  } else if (iter != logic_ops.end()) {
    is_logic_ops_ = true;
    (void)std::transform(logic_func_list_.begin(), logic_func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, ScalarArithmeticFunc> &item) { return item.first; });
  } else {
    (void)std::transform(math_func_list_.begin(), math_func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, ScalarArithmeticFunc> &item) { return item.first; });
  }
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarAdd,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarAdd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarSub,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarSub); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarMul,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarMul); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarDiv,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarDiv); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarFloordiv,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarFloordiv); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScalarMod,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarMod); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, scalar_eq,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarEq); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, scalar_gt,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarGt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, scalar_ge,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarGe); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, scalar_lt,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarLt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, scalar_le,
                                 []() { return std::make_shared<ScalarArithmeticCpuKernelMod>(kScalarLe); });
}  // namespace kernel
}  // namespace mindspore
