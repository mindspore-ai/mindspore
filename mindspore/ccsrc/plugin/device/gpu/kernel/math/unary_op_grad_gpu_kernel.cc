/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <functional>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/math/unary_op_grad_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t INPUT_NUM = 2;
constexpr size_t OUTPUT_NUM = 1;
constexpr auto kSqrtGrad = "SqrtGrad";
constexpr auto kRsqrtGrad = "RsqrtGrad";
constexpr auto kAsinGrad = "AsinGrad";
constexpr auto kACosGrad = "ACosGrad";
constexpr auto kAtanGrad = "AtanGrad";
constexpr auto kAsinhGrad = "AsinhGrad";
constexpr auto kAcoshGrad = "AcoshGrad";
constexpr auto kReciprocalGrad = "ReciprocalGrad";
constexpr auto kInvGrad = "InvGrad";
}  // namespace

std::map<std::string, std::vector<std::pair<KernelAttr, UnaryGradOpGpuKernelMod::UnaryOpGradFunc>>>
  UnaryGradOpGpuKernelMod::kernel_attr_map_ = {
    {kSqrtGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kRsqrtGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<double>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>}}},
    {kAsinGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kACosGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kAtanGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>}}},
    {kAsinhGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kAcoshGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kReciprocalGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>}}},
    {kInvGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryGradOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryGradOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryGradOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex64)
         .AddInputAttr(kNumberTypeComplex64)
         .AddOutputAttr(kNumberTypeComplex64),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeComplex128)
         .AddInputAttr(kNumberTypeComplex128)
         .AddOutputAttr(kNumberTypeComplex128),
       &UnaryGradOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}}};

bool UnaryGradOpGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "For 'UnaryGrad op', the kernel name must be in"
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, UnaryGradOpGpuKernelMod::UnaryOpGradFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return false;
  }
  size_t input_num = inputs.size();
  if (input_num != INPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    return false;
  }
  size_t output_num = outputs.size();
  if (output_num != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;
  return true;
}

int UnaryGradOpGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  size_t input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  return KRET_OK;
}

std::vector<KernelAttr> UnaryGradOpGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'UnaryGrad', it cannot support " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UnaryOpGradFunc> &item) { return item.first; });
  return support_list;
}

template <typename T>
bool UnaryGradOpGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  std::map<std::string, std::function<void(const T *, const T *, T *, const size_t, cudaStream_t)>> func_map;
  const bool is_t_complex = (std::is_same_v<T, utils::Complex<float>>) || (std::is_same_v<T, utils::Complex<double>>);
  if constexpr (is_t_complex) {
    std::map<std::string, std::function<void(const T *, const T *, T *, const size_t, cudaStream_t)>> func_map_complex =
      {{kReciprocalGrad, ReciprocalGrad<T>},
       {kInvGrad, InvGrad<T>},
       {kACosGrad, ACosGrad<T>},
       {kAcoshGrad, AcoshGrad<T>},
       {kAsinGrad, AsinGrad<T>},
       {kAsinhGrad, AsinhGrad<T>},
       {kSqrtGrad, SqrtGrad<T>},
       {kRsqrtGrad, RsqrtGrad<T>}};
    copy(func_map_complex.begin(), func_map_complex.end(), inserter(func_map, func_map.begin()));
  } else {
    std::map<std::string, std::function<void(const T *, const T *, T *, const size_t, cudaStream_t)>> func_map_normal =
      {{kSqrtGrad, SqrtGrad<T>},   {kRsqrtGrad, RsqrtGrad<T>},
       {kAsinGrad, AsinGrad<T>},   {kACosGrad, ACosGrad<T>},
       {kAtanGrad, AtanGrad<T>},   {kAsinhGrad, AsinhGrad<T>},
       {kAcoshGrad, AcoshGrad<T>}, {kReciprocalGrad, ReciprocalGrad<T>},
       {kInvGrad, InvGrad<T>}};
    copy(func_map_normal.begin(), func_map_normal.end(), inserter(func_map, func_map.begin()));
  }
  auto iter = func_map.find(kernel_name_);
  if (iter == func_map.end()) {
    MS_LOG(ERROR)
      << "For 'UnaryGrad', only support these types: "
      << kernel::Map2Str<std::map, std::function<void(const T *, const T *, T *, const size_t, cudaStream_t)>>(func_map)
      << " currently, but got " << kernel_name_;
    return false;
  }
  auto input_x_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto input_dx_addr = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto output_y_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  iter->second(input_x_addr, input_dx_addr, output_y_addr, input_size_list_[0] / sizeof(T),
               reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SqrtGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kSqrtGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, RsqrtGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kRsqrtGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AsinGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAsinGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ACosGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kACosGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AtanGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAtanGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AsinhGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAsinhGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AcoshGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kAcoshGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ReciprocalGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kReciprocalGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, InvGrad,
                                 []() { return std::make_shared<UnaryGradOpGpuKernelMod>(kInvGrad); });
}  // namespace kernel
}  // namespace mindspore
