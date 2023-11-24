/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/concat_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ops/op_utils.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr size_t kConcatOutputsNum = 1;
}  // namespace

bool ConcatCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int ConcatCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  MS_CHECK_VALUE(inputs.size() > 1, CheckAndConvertUtils::FormatCheckIntegerMsg(kernel_name_, inputs.size(),
                                                                                kGreaterThan, 1, primitive()));
  input_tensor_num_ = inputs.size() - 1;
  inputs_shape_.clear();
  for (size_t i = 0; i < input_tensor_num_; ++i) {
    inputs_shape_.push_back(inputs[i]->GetShapeVector());
  }

  axis_ = LongToInt(inputs[input_tensor_num_]->GetValueWithCheck<int64_t>());
  auto rank = SizeToInt(inputs_shape_[0].size());
  MS_CHECK_VALUE(-rank <= axis_ && axis_ < rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_, kIncludeLeft, {-rank, rank}, primitive()));
  if (axis_ < 0) {
    axis_ = axis_ + rank;
  }

  input_flat_shape_list_.clear();
  for (size_t i = 0; i < input_tensor_num_; i++) {
    auto input_shape_i = inputs_shape_[i];
    auto flat_shape = CPUKernelUtils::FlatShapeByAxis(input_shape_i, axis_);
    (void)input_flat_shape_list_.emplace_back(flat_shape);
  }

  output_dim_ = 0;
  offset_.clear();
  for (size_t j = 0; j < input_tensor_num_; ++j) {
    offset_.push_back(output_dim_);
    output_dim_ += LongToSize(input_flat_shape_list_[j][1]);
  }

  return KRET_OK;
}

template <typename T>
bool ConcatCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &,
                                      const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_tensor_num_ + 1, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kConcatOutputsNum, kernel_name_);

  auto *output_addr = reinterpret_cast<T *>(outputs[0]->device_ptr());
  std::vector<T *> input_addr_list;
  for (size_t j = 0; j < input_tensor_num_; ++j) {
    auto *tmp_addr = reinterpret_cast<T *>(inputs[j]->device_ptr());
    (void)input_addr_list.emplace_back(tmp_addr);
  }
  if (input_flat_shape_list_.size() == 0 || input_flat_shape_list_[0].size() == 0) {
    return true;
  }

  auto concat_times = LongToSize(input_flat_shape_list_[0][0]) * input_tensor_num_;
  auto task = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; ++pos) {
      size_t i = pos / input_tensor_num_;
      size_t j = pos % input_tensor_num_;

      if (input_flat_shape_list_[j][1] == 0) {
        continue;
      }
      auto copy_num = LongToSize(input_flat_shape_list_[j][1]);
      auto copy_size = copy_num * sizeof(T);
      auto offset = copy_num * i;
      auto output_ptr = output_addr + i * output_dim_ + offset_[j];
      auto ret = memcpy_s(output_ptr, copy_size, input_addr_list[j] + offset, copy_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s failed. Error no: " << ret;
      }
    }
  };
  ParallelLaunchAutoSearch(task, concat_times, this, &parallel_search_info_);
  return true;
}

#define CONCAT_CPU_KERNEL_ATTR(input_type, real_type)    \
  {                                                      \
    KernelAttr()                                         \
      .AddAllSameAttr(true, 1)                           \
      .AddInputAttr(input_type)                          \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
      .AddOutputAttr(input_type),                        \
      &ConcatCpuKernelMod::LaunchKernel<real_type>       \
  }

const std::vector<std::pair<KernelAttr, ConcatCpuKernelMod::KernelRunFunc>> &ConcatCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ConcatCpuKernelMod::KernelRunFunc>> func_list = {
    CONCAT_CPU_KERNEL_ATTR(kNumberTypeFloat16, float16),       CONCAT_CPU_KERNEL_ATTR(kNumberTypeFloat32, float),
    CONCAT_CPU_KERNEL_ATTR(kNumberTypeFloat64, double),        CONCAT_CPU_KERNEL_ATTR(kNumberTypeInt8, int8_t),
    CONCAT_CPU_KERNEL_ATTR(kNumberTypeInt16, int16_t),         CONCAT_CPU_KERNEL_ATTR(kNumberTypeInt32, int32_t),
    CONCAT_CPU_KERNEL_ATTR(kNumberTypeInt64, int64_t),         CONCAT_CPU_KERNEL_ATTR(kNumberTypeUInt8, uint8_t),
    CONCAT_CPU_KERNEL_ATTR(kNumberTypeUInt16, uint16_t),       CONCAT_CPU_KERNEL_ATTR(kNumberTypeUInt32, uint32_t),
    CONCAT_CPU_KERNEL_ATTR(kNumberTypeUInt64, uint64_t),       CONCAT_CPU_KERNEL_ATTR(kNumberTypeComplex64, complex64),
    CONCAT_CPU_KERNEL_ATTR(kNumberTypeComplex128, complex128), CONCAT_CPU_KERNEL_ATTR(kNumberTypeBool, bool)};

  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Concat, ConcatCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
