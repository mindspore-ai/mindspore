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
#include <algorithm>
#include <complex>
#include "plugin/device/cpu/kernel/segment_arithmetic_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kSegmentsThreshold = 2 * 1024;
const size_t kDataSizeThreshold = 2 * 1024;
}  // namespace

template <typename T>
T ComputeProd(const T num_1, const T num_2) {
  T res;
  auto a = num_1.real();
  auto b = num_1.imag();
  auto x = num_2.real();
  auto y = num_2.imag();
  auto real_res = a * x - b * y;
  auto imag_res = b * x + a * y;
  res.real(real_res);
  res.imag(imag_res);
  return res;
}

template <typename T>
void ComputeFuncSum(void *output_addr, void *input_addr) {
  T *output_ptr = static_cast<T *>(output_addr);
  T *input_ptr = static_cast<T *>(input_addr);
  auto a = *output_ptr;
  auto b = *input_ptr;
  *output_ptr = a + b;
}

template <typename T>
void ComputeFuncProd(void *output_addr, void *input_addr) {
  T *output_ptr = static_cast<T *>(output_addr);
  T *input_ptr = static_cast<T *>(input_addr);
  T a = *output_ptr;
  T b = *input_ptr;
  T prod_value;
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    prod_value = ComputeProd(a, b);
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    prod_value = ComputeProd(a, b);
  } else {
    prod_value = a * b;
  }
  *output_ptr = prod_value;
}

template <typename T>
T SegmentArithmeticCPUKernelMod::GetInitValue() {
  static const std::map<std::string, T> SegmentArithmeticInitValueMap{
    {prim::kPrimSegmentProd->name(), static_cast<T>(1.0)},
    {prim::kPrimSegmentSum->name(), static_cast<T>(0.0)},
  };
  if (SegmentArithmeticInitValueMap.find(kernel_name_) == SegmentArithmeticInitValueMap.end()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the current operator does not support this operation.";
  }
  T init_value = SegmentArithmeticInitValueMap.at(kernel_name_);
  return init_value;
}

template <typename T>
bool SegmentArithmeticCPUKernelMod::GetComputeFunc() {
  static const std::map<std::string, SegmentComputeFunc> ComputeFuncList = {
    {prim::kPrimSegmentSum->name(), ComputeFuncSum<T>},
    {prim::kPrimSegmentProd->name(), ComputeFuncProd<T>},
  };
  if (ComputeFuncList.find(kernel_name_) == ComputeFuncList.end()) {
    MS_LOG(ERROR) << "Invalid '" << kernel_name_ << "'.";
  }
  compute_func_ = ComputeFuncList.at(kernel_name_);
  return true;
}

bool SegmentArithmeticCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  input_x_dtype_ = inputs.at(kIndex0)->GetDtype();
  segment_ids_dtype_ = inputs.at(kIndex1)->GetDtype();
  output_dtype_ = outputs.at(kIndex0)->GetDtype();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SegmentMax does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SegmentArithmeticCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  segment_ids_shape_ = inputs.at(kIndex1)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  input_x_num_ = SizeOf(input_x_shape_);
  segment_ids_num_ = SizeOf(segment_ids_shape_);
  output_num_ = SizeOf(output_shape_);
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, SegmentArithmeticCPUKernelMod::SegmentArithmeticFunc>>
  SegmentArithmeticCPUKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<float16, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<std::complex<float>, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<std::complex<double>, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<float16, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<std::complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &SegmentArithmeticCPUKernelMod::LaunchKernel<std::complex<double>, int64_t>}};

std::vector<KernelAttr> SegmentArithmeticCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SegmentArithmeticFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T1, typename T2>
bool SegmentArithmeticCPUKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &workspace,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  if (kernel_name_ == prim::kPrimSegmentMax->name() || kernel_name_ == prim::kPrimSegmentMin->name()) {
    if constexpr (std::is_same_v<T1, std::complex<float>>) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x types can not be complex64.";
    } else if constexpr (std::is_same_v<T1, std::complex<double>>) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x types can not be complex128.";
    }
  }
  if (auto ret = GetComputeFunc<T1>(); !ret) {
    return ret;
  }
  T1 init_value = GetInitValue<T1>();
  auto input_x_data_addr = static_cast<T1 *>(inputs[0]->addr);
  auto segment_ids_data_addr = static_cast<T2 *>(inputs[1]->addr);
  auto output_data_addr = static_cast<T1 *>(outputs[0]->addr);
  std::vector<int64_t> segments = CPUKernelUtils::CalcSegmentIds(segment_ids_data_addr, segment_ids_num_);
  for (size_t i = 0; i < output_num_; ++i) {
    output_data_addr[i] = init_value;
  }
  if (input_x_shape_[0] == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x_shape_[0] can not be 0";
  }
  const size_t num_compare_per = input_x_num_ / LongToSize(input_x_shape_[0]);
  const size_t num_segments = segments.size();
  if (num_segments < kSegmentsThreshold) {
    for (size_t i = 0; i < num_segments; ++i) {
      const size_t count = static_cast<size_t>(segments[i]);
      int64_t count_no = 0;
      for (size_t j = 0; j < i; ++j) {
        count_no += segments[j];
      }
      size_t input_addr_base = LongToSize(count_no) * num_compare_per;
      auto task = [&](size_t start, size_t end) {
        for (size_t j = start; j < end; ++j) {
          size_t res_init_addr = input_addr_base + j;
          T1 res_value = input_x_data_addr[res_init_addr];
          for (size_t k = 1; k < count; ++k) {
            int cmp_addr = res_init_addr + k * num_compare_per;
            compute_func_(static_cast<void *>(&res_value), static_cast<void *>(input_x_data_addr + cmp_addr));
          }
          output_data_addr[segment_ids_data_addr[LongToSize(count_no)] * num_compare_per + j] = res_value;
        }
      };
      if (num_compare_per < kDataSizeThreshold) {
        task(0, num_compare_per);
      } else {
        CPUKernelUtils::ParallelFor(task, num_compare_per);
      }
    }
  } else {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        const size_t count = static_cast<size_t>(segments[i]);
        int64_t count_no = 0;
        for (size_t j = 0; j < i; ++j) {
          count_no += segments[j];
        }
        size_t input_addr_base = LongToSize(count_no) * num_compare_per;
        for (size_t j = 0; j < num_compare_per; ++j) {
          size_t res_init_addr = input_addr_base + j;
          T1 res_value = input_x_data_addr[res_init_addr];
          for (size_t k = 1; k < count; ++k) {
            int cmp_addr = res_init_addr + k * num_compare_per;
            compute_func_(static_cast<void *>(&res_value), static_cast<void *>(input_x_data_addr + cmp_addr));
          }
          output_data_addr[segment_ids_data_addr[LongToSize(count_no)] * num_compare_per + j] = res_value;
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, num_segments);
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SegmentSum, SegmentArithmeticCPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SegmentProd, SegmentArithmeticCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
