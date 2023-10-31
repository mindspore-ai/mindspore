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

#include "plugin/device/cpu/kernel/sequence/sequence_concat_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/add_fp32.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kOutputsNum = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool SequenceConcatCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SequenceConcatCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  tuple_shape_ = inputs[0]->GetShapeVector();
  if (tuple_shape_.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input tuple size must greater 0";
  }
  std::vector<int64_t> shape_vec_item;
  (void)std::copy(tuple_shape_.begin() + 1, tuple_shape_.end(), std::back_inserter(shape_vec_item));

  input_num_ = LongToSize(tuple_shape_[0]);
  inputs_shape_.clear();
  for (size_t i = 0; i < input_num_; ++i) {
    inputs_shape_.push_back(shape_vec_item);
  }

  axis_ = LongToInt(inputs[kIndex1]->GetValueWithCheck<int64_t>());
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(inputs_shape_[0].size());
  }

  input_flat_shape_list_.clear();
  for (size_t i = 0; i < input_num_; i++) {
    auto input_shape_i = inputs_shape_[i];
    auto flat_shape = CPUKernelUtils::FlatShapeByAxis(input_shape_i, axis_);
    (void)input_flat_shape_list_.emplace_back(flat_shape);
  }

  output_dim_ = 0;
  offset_.clear();
  for (size_t j = 0; j < input_num_; ++j) {
    offset_.push_back(output_dim_);
    output_dim_ += LongToSize(input_flat_shape_list_[j][1]);
  }

  return KRET_OK;
}

template <typename T>
bool SequenceConcatCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &,
                                              const std::vector<KernelTensor *> &outputs) {
  const auto input_addr = GetDeviceAddress<T>(inputs, 0);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->device_ptr());

  size_t element_index_size =
    LongToSize(std::accumulate(tuple_shape_.begin() + 1, tuple_shape_.end(), 1, std::multiplies<int64_t>()));

  std::vector<T *> input_addr_list;
  for (size_t j = 0; j < input_num_; ++j) {
    auto *tmp_addr = reinterpret_cast<T *>(input_addr + j * element_index_size);
    (void)input_addr_list.emplace_back(tmp_addr);
  }
  if (input_flat_shape_list_.empty() || input_flat_shape_list_[0].empty()) {
    return true;
  }

  auto concat_times = LongToSize(input_flat_shape_list_[0][0]) * input_num_;
  auto task = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; ++pos) {
      size_t i = pos / input_num_;
      size_t j = pos % input_num_;

      if (input_flat_shape_list_[j][1] == 0) {
        continue;
      }
      auto copy_num = LongToSize(input_flat_shape_list_[j][1]);
      auto copy_size = copy_num * sizeof(T);
      auto offset = copy_num * i;
      auto output_ptr = output_addr + i * output_dim_ + offset_[j];
      auto ret = memcpy_s(reinterpret_cast<void *>(output_ptr), copy_size, input_addr_list[j] + offset, copy_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s failed. Error no: " << ret;
      }
    }
  };
  ParallelLaunchAutoSearch(task, concat_times, this, &parallel_search_info_);
  return true;
}

#define SEQUENCE_CONCAT_REG(ms_type, builtin_type)            \
  {                                                           \
    KernelAttr()                                              \
      .AddInputAttr(kObjectTypeTuple, ms_type)                \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)      \
      .AddOutputAttr(ms_type),                                \
      &SequenceConcatCpuKernelMod::LaunchKernel<builtin_type> \
  }

const SequenceConcatCpuKernelMod::FuncList &SequenceConcatCpuKernelMod::GetFuncList() const {
  static const FuncList func_list = {
    SEQUENCE_CONCAT_REG(kNumberTypeInt8, int8_t),           SEQUENCE_CONCAT_REG(kNumberTypeInt16, int16_t),
    SEQUENCE_CONCAT_REG(kNumberTypeInt32, int32_t),         SEQUENCE_CONCAT_REG(kNumberTypeInt64, int64_t),
    SEQUENCE_CONCAT_REG(kNumberTypeUInt8, uint8_t),         SEQUENCE_CONCAT_REG(kNumberTypeUInt16, uint16_t),
    SEQUENCE_CONCAT_REG(kNumberTypeUInt32, uint32_t),       SEQUENCE_CONCAT_REG(kNumberTypeUInt64, uint64_t),
    SEQUENCE_CONCAT_REG(kNumberTypeFloat16, float16),       SEQUENCE_CONCAT_REG(kNumberTypeFloat32, float),
    SEQUENCE_CONCAT_REG(kNumberTypeFloat64, double),        SEQUENCE_CONCAT_REG(kNumberTypeComplex64, complex64),
    SEQUENCE_CONCAT_REG(kNumberTypeComplex128, complex128), SEQUENCE_CONCAT_REG(kNumberTypeBool, bool)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceConcat, SequenceConcatCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
