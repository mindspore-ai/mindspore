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
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
static constexpr size_t INPUT_NUM = 2;
static constexpr size_t OUTPUT_NUM = 1;
template <typename T>
T GetDivZeroVal(const T &v) {
  auto zero = static_cast<T>(0.0);
  if (std::numeric_limits<T>::has_infinity) {
    return v > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
  } else {
    return v > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
  }
}

template <>
complex128 GetDivZeroVal(const complex128 &v) {
  return std::numeric_limits<complex128>::quiet_NaN();
}

template <>
complex64 GetDivZeroVal(const complex64 &v) {
  return std::numeric_limits<complex64>::quiet_NaN();
}

template <typename T>
bool XdivyCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto y_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T);
  BroadcastIterator base_iter(x_shape_, y_shape_, out_shape_);
  auto task = [&x_addr, &y_addr, &output_addr, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto dividend = x_addr[iter.GetInputPosA()];
      auto divisor = y_addr[iter.GetInputPosB()];
      iter.GenNextPos();
      auto zero = (T)0;
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
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

bool XdivyCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs) {
  const size_t kInputsNum = 2;
  const size_t kOutputsNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
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

  auto iter = func_map_.find(x_type);
  if (iter == func_map_.end()) {
    MS_LOG(ERROR) << "Xdivy only support tensor with data type float16, float32, "
                     "float64, Complex64, Complex128, but got typeid:"
                  << x_type;
    return false;
  }
  kernel_func_ = iter->second;
  return true;
}

int XdivyCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_NUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  ResetResource();
  int ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  x_shape_ = inputs[0]->GetShapeVector();
  y_shape_ = inputs[1]->GetShapeVector();
  out_shape_ = outputs[0]->GetShapeVector();
  if (out_shape_.empty()) {
    out_shape_.emplace_back(1);
  }
  auto x_shape_len = x_shape_.size();
  for (size_t i = 0; i < out_shape_.size() - x_shape_len; ++i) {
    (void)x_shape_.insert(x_shape_.begin(), 1);
  }
  auto y_shape_len = y_shape_.size();
  for (size_t i = 0; i < out_shape_.size() - y_shape_len; ++i) {
    (void)y_shape_.insert(y_shape_.begin(), 1);
  }
  return KRET_OK;
}

std::vector<KernelAttr> XdivyCpuKernelMod::GetOpSupport() { return support_ops_; }

std::vector<KernelAttr> XdivyCpuKernelMod::support_ops_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64)},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128)},
};

std::map<mindspore::TypeId, XdivyCpuKernelMod::XdivyFunc> XdivyCpuKernelMod::func_map_ = {
  {kNumberTypeFloat16, &XdivyCpuKernelMod::LaunchKernel<float16>},
  {kNumberTypeFloat32, &XdivyCpuKernelMod::LaunchKernel<float>},
  {kNumberTypeFloat64, &XdivyCpuKernelMod::LaunchKernel<double>},
  {kNumberTypeComplex64, &XdivyCpuKernelMod::LaunchKernel<complex64>},
  {kNumberTypeComplex128, &XdivyCpuKernelMod::LaunchKernel<complex128>}};
}  // namespace kernel
}  // namespace mindspore
