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

#include "kernel/cpu/native/renorm_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <map>
#include <complex>

#include "mindspore/ops/infer/renorm.h"

#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace renorm_cpu {
namespace {
constexpr size_t kRenormInputsNum = 1;
constexpr size_t kRenormOutputsNum = 1;
}  // namespace

bool RenormCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kRenormInputsNum || outputs.size() != kRenormOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output tensor number must be " << kRenormInputsNum
                  << " and " << kRenormOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "Renorm does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  return true;
}

int RenormCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(inputs, outputs)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " resize failed.";
    return ret;
  }
  x_shape_ = inputs[kIndex0]->GetShapeVector();
  axis_ = GetValue<int64_t>(primitive_->GetAttr("dim"));
  p_ = GetValue<float>(primitive_->GetAttr("p"));
  max_norm_ = GetValue<float>(primitive_->GetAttr("maxnorm"));
  return 0;
}

void RenormCpuKernelMod::CheckAndInitParams() {
  if (p_ <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the attribute norm 'p' must be positive, but got " << p_;
  }
  if (max_norm_ < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the attribute 'maxnorm' must be non-negative, but got "
                      << max_norm_;
  }

  auto x_rank = x_shape_.size();
  if (x_rank == 0) {
    if (axis_ != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << ", the input is a scala, and the attribute 'dim' must be 0, but got " << axis_;
    }
  } else if (axis_ < -SizeToLong(x_rank) || axis_ >= SizeToLong(x_rank)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the attribute 'dim' must be in range [" << -SizeToLong(x_rank)
                      << ", " << x_rank << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += SizeToLong(x_rank);
  }

  stride_size_ = 1;
  inner_size_ = 1;
  axis_size_ = 1;
  total_size_ = 1;
  for (size_t i = 0; i < x_rank; ++i) {
    if (SizeToLong(i) == axis_) {
      axis_size_ *= LongToSize(x_shape_[i]);
    } else if (SizeToLong(i) < axis_) {
      stride_size_ *= LongToSize(x_shape_[i]);
    } else {
      inner_size_ *= LongToSize(x_shape_[i]);
    }
    total_size_ *= LongToSize(x_shape_[i]);
  }
}

template <typename T>
bool RenormCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                      const std::vector<kernel::KernelTensor *> &,
                                      const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRenormInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRenormOutputsNum, kernel_name_);
  auto *x = GetDeviceAddress<T>(inputs, kIndex0);
  auto *output = GetDeviceAddress<T>(outputs, kIndex0);
  CheckAndInitParams();

  auto axis_size = axis_size_;      // maximum parallel number
  auto inner_size = inner_size_;    // continuous number
  auto stride_size = stride_size_;  // stride number
  auto total_size = total_size_;    // total number
  auto p = static_cast<double>(p_);
  auto maxnorm = static_cast<double>(max_norm_);

  auto pnorm = std::make_unique<double[]>(axis_size);
  auto task = [&](const size_t start, const size_t end) {
    for (size_t ith = start; ith < end; ++ith) {
      double single_sum = static_cast<double>(0.0);
      size_t step_len = total_size / stride_size;
      for (size_t pos_ith = ith * inner_size; pos_ith < total_size; pos_ith += step_len) {
        for (size_t j = 0; j < inner_size; ++j) {
          size_t index = pos_ith + j;
          single_sum += pow(static_cast<double>(abs(x[index])), p);
        }
      }
      pnorm[ith] = pow(single_sum, static_cast<double>(1.0) / p);

      for (size_t pos_ith = ith * inner_size; pos_ith < total_size; pos_ith += step_len) {
        for (size_t j = 0; j < inner_size; ++j) {
          size_t index = pos_ith + j;
          if (pnorm[ith] > maxnorm) {
            output[index] = x[index] / static_cast<T>(pnorm[ith]) * static_cast<T>(maxnorm);
          } else {
            output[index] = x[index];
          }
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, axis_size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, RenormCpuKernelMod::RenormFunc>> RenormCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &RenormCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &RenormCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &RenormCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &RenormCpuKernelMod::LaunchKernel<std::complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &RenormCpuKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> RenormCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RenormFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Renorm, RenormCpuKernelMod);
}  // namespace renorm_cpu
}  // namespace kernel
}  // namespace mindspore
