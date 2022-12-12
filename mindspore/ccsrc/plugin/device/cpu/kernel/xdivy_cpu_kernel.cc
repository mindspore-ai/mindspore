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
#include "plugin/device/cpu/kernel/xdivy_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <limits>
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "Eigen/Eigen"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
static constexpr size_t INPUT_NUM = 2;
static constexpr size_t OUTPUT_NUM = 1;
static constexpr int MAX_DIMS = 7;
static constexpr size_t PARALLEL_THRESHOLD = 4096;
template <typename T>
T GetDivZeroVal(const T &v) {
  auto zero = static_cast<T>(0.0);
  return v > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
}

template <>
complex128 GetDivZeroVal(const complex128 &) {
  return std::numeric_limits<complex128>::quiet_NaN();
}

template <>
complex64 GetDivZeroVal(const complex64 &) {
  return std::numeric_limits<complex64>::quiet_NaN();
}

template <class T>
bool isZero(const T &val) {
  return val == T(0.0f);
}

template <>
bool isZero(const float &val) {
  return std::fpclassify(val) == FP_ZERO;
}

template <>
bool isZero(const double &val) {
  return std::fpclassify(val) == FP_ZERO;
}

template <typename T>
void SameShapeTask(T *x_addr, T *y_addr, T *output_addr, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    auto dividend = x_addr[i];
    auto divisor = y_addr[i];
    if (isZero(divisor)) {
      if (isZero(dividend)) {
        output_addr[i] = static_cast<T>(0.0);
        continue;
      }
      output_addr[i] = GetDivZeroVal(dividend);
      continue;
    }
    output_addr[i] = dividend / divisor;
  }
}

template <>
void SameShapeTask(float *x_addr, float *y_addr, float *output_addr, size_t start, size_t end) {
  Eigen::Map<Eigen::ArrayXf> x_v(x_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXf> y_v(y_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXf> o_v(output_addr + start, end - start);
  o_v = (x_v == 0).select(o_v, x_v / y_v);
}

template <>
void SameShapeTask(double *x_addr, double *y_addr, double *output_addr, size_t start, size_t end) {
  Eigen::Map<Eigen::ArrayXd> x_v(x_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXd> y_v(y_addr + start, end - start);
  Eigen::Map<Eigen::ArrayXd> o_v(output_addr + start, end - start);
  o_v = (x_v == 0).select(o_v, x_v / y_v);
}

template <typename T>
bool XdivyCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  if (has_null_input_) {
    return true;
  }
  auto x_addr = static_cast<T *>(inputs[0]->addr);
  auto y_addr = static_cast<T *>(inputs[1]->addr);
  auto output_addr = static_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T);
  auto sameShapeTask = [&x_addr, &y_addr, &output_addr](size_t start, size_t end) {
    SameShapeTask(x_addr, y_addr, output_addr, start, end);
  };
  auto diffShapeTask = [this, &x_addr, &y_addr, &output_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto idxX = index_listx_[i];
      auto idxY = index_listy_[i];
      auto dividend = x_addr[idxX];
      auto divisor = y_addr[idxY];
      auto zero = static_cast<T>(0);
      if (divisor == zero) {
        if (dividend == zero) {
          output_addr[i] = zero;
          continue;
        }
        output_addr[i] = GetDivZeroVal(dividend);
        continue;
      }
      output_addr[i] = dividend / divisor;
    }
  };

  CTask task = is_need_broadcast_ ? CTask(diffShapeTask) : CTask(sameShapeTask);
  if (output_size < PARALLEL_THRESHOLD) {
    task(0, output_size);
  } else {
    ParallelLaunch(task, output_size, PARALLEL_THRESHOLD, this, pool_);
  }
  return true;
}

bool XdivyCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

bool XdivyCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto x_type = inputs[0]->GetDtype();
  auto y_type = inputs[1]->GetDtype();
  auto out_type = outputs[0]->GetDtype();
  if (!(x_type == y_type && x_type == out_type)) {
    MS_LOG(ERROR) << "Xdivy need same input and output data type, but got X type:" << x_type << " Y type:" << y_type
                  << " out type:" << out_type;
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int XdivyCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  ResetResource();
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto x_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto y_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());

  // while has null input, xdivy result is null too
  has_null_input_ = CheckNullInput(x_shape) || CheckNullInput(y_shape);
  if (has_null_input_) {
    return 0;
  }

  auto out_shape = LongVecToSizeVec(outputs.at(kIndex0)->GetShapeVector());
  if (out_shape.size() > MAX_DIMS || out_shape.size() < x_shape.size() || out_shape.size() < y_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", and output dimension can't less than input; but got x_shape dimension:" << x_shape.size()
                      << " ,y_shape dimension:" << y_shape.size() << " ,out_shape dimension:" << out_shape.size();
  }
  is_need_broadcast_ = x_shape != y_shape;
  if (is_need_broadcast_) {
    GetBroadCastIndex(x_shape, out_shape, &index_listx_);
    GetBroadCastIndex(y_shape, out_shape, &index_listy_);
  }
  return 0;
}

const std::vector<std::pair<KernelAttr, XdivyCpuKernelMod::KernelRunFunc>> &XdivyCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, XdivyCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &XdivyCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &XdivyCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &XdivyCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &XdivyCpuKernelMod::LaunchKernel<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &XdivyCpuKernelMod::LaunchKernel<complex128>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Xdivy, XdivyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
