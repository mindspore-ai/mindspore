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

#include "kernel/cpu/native/split_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>

#include "include/utils/thread_pool.h"
#include "infer/ops_func_impl/split.h"

namespace mindspore {
namespace kernel {
namespace split_cpu {
namespace {
constexpr size_t kSplitInputsNum = 3;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool SplitCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  axis_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  output_num_ = IntToSize(inputs[kIndex2]->GetValueWithCheck<int64_t>());
  if (output_num_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'output_num' must be positive int, but got 0.";
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::tuple<KernelAttr, SplitFunc, InitIOFunc> &tuple_item) { return std::get<0>(tuple_item); });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Split does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = std::get<1>(func_list_[index]);
  const size_t kTwoIdx = 2;
  init_io_func_ = std::get<kTwoIdx>(func_list_[index]);
  return true;
}

int SplitCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSplitInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num_, kernel_name_);
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_.clear();
  auto input_shape = inputs[0]->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_),
                       [](const int64_t &value) { return LongToInt(value); });
  CheckParam();
  init_io_func_(this);
  return ret;
}

template <typename T>
void SplitCpuKernelMod::InitIOSize() {
  (void)workspace_size_list_.emplace_back((sizeof(T *) * LongToSize(output_num_)));
}

template <typename T>
void SplitCpuKernelMod::LaunchSplit(T *input, T **output, size_t /* size */) {
  SplitParameter param;
  param.num_split_ = SizeToInt(output_num_);
  param.split_dim_ = LongToInt(axis_);
  param.strides_[input_shape_.size() - 1] = 1;
  for (int i = SizeToInt(input_shape_.size()) - 2; i >= 0; i--) {  // from -2 to 0 dim
    param.strides_[i] = param.strides_[i + 1] * input_shape_[i + 1];
  }
  auto split_sizes = std::make_unique<int[]>(IntToSize(param.num_split_));
  param.split_sizes_ = split_sizes.get();
  int split_size = input_shape_[param.split_dim_] / SizeToInt(output_num_);
  for (int i = 0; i < param.num_split_; i++) {
    param.split_sizes_[i] = split_size;
  }
  param.split_count_ = 1;
  for (size_t i = 0; i < static_cast<size_t>(axis_); ++i) {
    param.split_count_ *= input_shape_[i];
  }
  auto task = [this, &input, &output, &param](size_t start, size_t end) {
    (void)DoSplit(input, reinterpret_cast<void **>(output), &input_shape_[0], SizeToInt(start), SizeToInt(end - start),
                  &param, SizeToInt(sizeof(T)));
  };
  ParallelLaunchAutoSearch(task, IntToSize(param.split_count_ * param.num_split_), this, &parallel_search_info_);
  return;
}

template <typename T>
bool SplitCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  T **output = reinterpret_cast<T **>(workspace[0]->device_ptr());
  for (size_t i = 0; i < outputs.size(); i++) {
    output[i] = reinterpret_cast<T *>(outputs[i]->device_ptr());
  }
  size_t size = static_cast<size_t>(inputs[0]->size() / sizeof(T));
  LaunchSplit(input, output, size);
  return true;
}

void SplitCpuKernelMod::CheckParam() {
  int64_t dims = SizeToLong(input_shape_.size());
  if (dims == 0 || dims > SPLIT_STRIDES_SIZE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input tensor must be in range [1, "
                      << SPLIT_STRIDES_SIZE << "], but got " << dims;
  }
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in range [" << -dims << ", " << dims
                      << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += SizeToLong(input_shape_.size());
  }
  if (output_num_ > IntToSize(input_shape_[LongToUlong(axis_)])) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'output_num' must be less than or equal to "
                      << input_shape_[axis_] << ", but got " << output_num_;
  }
}

bool SplitCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSplitInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num_, kernel_name_);
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::tuple<KernelAttr, SplitCpuKernelMod::SplitFunc, SplitCpuKernelMod::InitIOFunc>>
  SplitCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SplitCpuKernelMod::LaunchKernel<float>, &SplitCpuKernelMod::InitIOSize<float>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SplitCpuKernelMod::LaunchKernel<float16>, &SplitCpuKernelMod::InitIOSize<float16>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SplitCpuKernelMod::LaunchKernel<double>, &SplitCpuKernelMod::InitIOSize<double>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &SplitCpuKernelMod::LaunchKernel<int8_t>, &SplitCpuKernelMod::InitIOSize<int8_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &SplitCpuKernelMod::LaunchKernel<int16_t>, &SplitCpuKernelMod::InitIOSize<int16_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SplitCpuKernelMod::LaunchKernel<int32_t>, &SplitCpuKernelMod::InitIOSize<int32_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SplitCpuKernelMod::LaunchKernel<int64_t>, &SplitCpuKernelMod::InitIOSize<int64_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &SplitCpuKernelMod::LaunchKernel<uint8_t>, &SplitCpuKernelMod::InitIOSize<uint8_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &SplitCpuKernelMod::LaunchKernel<uint16_t>, &SplitCpuKernelMod::InitIOSize<uint16_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &SplitCpuKernelMod::LaunchKernel<uint32_t>, &SplitCpuKernelMod::InitIOSize<uint32_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &SplitCpuKernelMod::LaunchKernel<uint64_t>, &SplitCpuKernelMod::InitIOSize<uint64_t>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SplitCpuKernelMod::LaunchKernel<complex64>, &SplitCpuKernelMod::InitIOSize<complex64>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &SplitCpuKernelMod::LaunchKernel<complex128>, &SplitCpuKernelMod::InitIOSize<complex128>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool),
     &SplitCpuKernelMod::LaunchKernel<bool>, &SplitCpuKernelMod::InitIOSize<bool>},
    {KernelAttr()
       .AddAllSameAttr(true)
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBFloat16),
     &SplitCpuKernelMod::LaunchKernel<bfloat16>, &SplitCpuKernelMod::InitIOSize<bfloat16>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Split, SplitCpuKernelMod);
}  // namespace split_cpu
}  // namespace kernel
}  // namespace mindspore
