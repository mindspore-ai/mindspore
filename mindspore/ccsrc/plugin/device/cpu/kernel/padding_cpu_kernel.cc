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

#include "plugin/device/cpu/kernel/padding_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>
#include <complex>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/padding.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool PaddingCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimPadding->name()) {
    MS_LOG(ERROR) << "For 'Padding', the kernel name must be 'Padding', but got " << kernel_name_;
    return false;
  }
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::Padding>(base_operator->GetPrim());
  pad_dim_size_ = LongToSize(kernel_ptr->get_pad_dim_size());

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int PaddingCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  shapes_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(shapes_), LongToSize);
  input_element_num_ = std::accumulate(shapes_.begin(), shapes_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num_ == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  // The input_shape size have been checked in Padding C++ primitive.
  x_last_dim_ = shapes_[shapes_.size() - kDim1];
  if (x_last_dim_ != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the last dimension of 'x' must be 1, but got: " << x_last_dim_
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  output_outer_size_ = 1;
  for (size_t i = 0; i < shapes_.size() - kDim1; i++) {
    output_outer_size_ *= shapes_[i];
  }
  output_element_num_ = output_outer_size_ * pad_dim_size_;
  return KRET_OK;
}

template <typename T>
bool PaddingCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->addr);

  auto memset_errno = memset_s(output_ptr, output_element_num_ * sizeof(T), 0, output_element_num_ * sizeof(T));
  if (memset_errno != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it's memset failed. Error no: " << memset_errno;
  }
  auto task = [this, input_ptr, output_ptr](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output_ptr[i * pad_dim_size_] = input_ptr[i];
    }
  };
  ParallelLaunchAutoSearch(task, output_outer_size_, this, &parallel_search_info_, pool_);
  return true;
}

const std::vector<std::pair<KernelAttr, PaddingCpuKernelMod::KernelRunFunc>> &PaddingCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, PaddingCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &PaddingCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &PaddingCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &PaddingCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &PaddingCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &PaddingCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &PaddingCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &PaddingCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &PaddingCpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &PaddingCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &PaddingCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &PaddingCpuKernelMod::LaunchKernel<complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &PaddingCpuKernelMod::LaunchKernel<complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &PaddingCpuKernelMod::LaunchKernel<bool>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Padding, PaddingCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
