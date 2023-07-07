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

#include "plugin/device/cpu/kernel/sequence/sequence_unstack_cpu_kernel.h"
#include <map>
#include <tuple>
#include <utility>
#include "ops/sequence_unstack.h"
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSequenceUnstackInputsNum = 1;
constexpr size_t kSequenceUnstackOutputsMinNum = 1;
constexpr size_t kSequenceUnstackWorkspaceMinNum = 1;
constexpr size_t kMaxDataSize = 2147483648;  // 2GB
}  // namespace

bool SequenceUnstackCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SequenceUnstack>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "cast sequence unstack ops failed!";
    return false;
  }
  sequence_unstack_param_.axis_ = kernel_ptr->get_axis();
  origin_axis_ = sequence_unstack_param_.axis_;
  sequence_unstack_param_.pre_dims_ = 1;
  sequence_unstack_param_.axis_dim_ = 1;
  sequence_unstack_param_.after_dims_ = 1;
  input_size_ = 1;

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SequenceUnstackFunc> &tuple_item) { return std::get<0>(tuple_item); });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(ERROR) << "SequenceUnstack does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = std::get<1>(func_list_[index]);
  return true;
}

int SequenceUnstackCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  MS_EXCEPTION_IF_NULL(base_operator);
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }

  input_size_ = 1;
  sequence_unstack_param_.pre_dims_ = 1;
  sequence_unstack_param_.axis_dim_ = 1;
  sequence_unstack_param_.after_dims_ = 1;
  sequence_unstack_param_.axis_ = origin_axis_;

  input_shape_ = inputs[0]->GetShapeVector();
  int32_t shape_size = SizeToInt(input_shape_.size());
  if (sequence_unstack_param_.axis_ < -shape_size || sequence_unstack_param_.axis_ >= shape_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the `axis` should be in [" << -shape_size << ", " << shape_size
                      << "), but got " << sequence_unstack_param_.axis_;
  }
  if (sequence_unstack_param_.axis_ < 0) {
    sequence_unstack_param_.axis_ += input_shape_.size();
  }
  output_num_ = input_shape_[sequence_unstack_param_.axis_];
  sequence_unstack_param_.num_ = SizeToInt(output_num_);

  for (size_t i = 0; i < input_shape_.size(); i++) {
    if (i < IntToSize(sequence_unstack_param_.axis_)) {
      sequence_unstack_param_.pre_dims_ *= LongToInt(input_shape_[i]);
    } else if (i > IntToSize(sequence_unstack_param_.axis_)) {
      sequence_unstack_param_.after_dims_ *= LongToInt(input_shape_[i]);
    } else {
      sequence_unstack_param_.axis_dim_ = LongToInt(input_shape_[i]);
    }
    input_size_ *= LongToSize(input_shape_[i]);
  }
  return KRET_OK;
}

bool SequenceUnstackCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &workspace,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSequenceUnstackInputsNum, kernel_name_);
  return kernel_func_(this, inputs, workspace, outputs);
}

template <typename T>
bool SequenceUnstackCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs) {
  const auto input = GetDeviceAddress<T>(inputs, 0);
  auto *outputs_host = reinterpret_cast<T *>(outputs[0]->addr);

  size_t total_size = input_size_ * sizeof(T);
  if (total_size >= kMaxDataSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input data size cannot larger than 2GB, but got "
                      << total_size << " bytes";
  }
  int data_size = SizeToInt(sizeof(T));
  int copy_size = sequence_unstack_param_.after_dims_ * data_size;
  int cp_ret = EOK;

  auto task = [this, input, outputs_host, data_size, copy_size, &cp_ret](size_t start, size_t end) {
    int sequence_cell_size = input_size_ / sequence_unstack_param_.axis_dim_;
    for (size_t i = start; i < end; i++) {
      int n = SizeToInt(i) / sequence_unstack_param_.axis_dim_;
      int c = SizeToInt(i) % sequence_unstack_param_.axis_dim_;
      int in_offset = n * sequence_unstack_param_.axis_dim_ * sequence_unstack_param_.after_dims_ +
                      c * sequence_unstack_param_.after_dims_;
      int out_offset = n * sequence_unstack_param_.after_dims_;
      int c_out_offset = c * sequence_cell_size + out_offset;
      auto ret = memcpy_s(outputs_host + c_out_offset, copy_size, input + in_offset, copy_size);
      if (ret != EOK && cp_ret == EOK) {
        cp_ret = ret;
      }
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(sequence_unstack_param_.num_ * sequence_unstack_param_.pre_dims_), this,
                           &parallel_search_info_);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", memcpy error, errorno: " << cp_ret;
  }
  return true;
}

#define SEQUENCE_STACK_REG(ms_type, builtin_type)                                \
  {                                                                              \
    KernelAttr().AddInputAttr(ms_type).AddOutputAttr(kObjectTypeTuple, ms_type), \
      &SequenceUnstackCpuKernelMod::LaunchKernel<builtin_type>                   \
  }

std::vector<std::pair<KernelAttr, SequenceUnstackCpuKernelMod::SequenceUnstackFunc>>
  SequenceUnstackCpuKernelMod::func_list_ = {
    SEQUENCE_STACK_REG(kNumberTypeInt8, int8_t),           SEQUENCE_STACK_REG(kNumberTypeInt16, int16_t),
    SEQUENCE_STACK_REG(kNumberTypeInt32, int32_t),         SEQUENCE_STACK_REG(kNumberTypeInt64, int64_t),
    SEQUENCE_STACK_REG(kNumberTypeUInt8, uint8_t),         SEQUENCE_STACK_REG(kNumberTypeUInt16, uint16_t),
    SEQUENCE_STACK_REG(kNumberTypeUInt32, uint32_t),       SEQUENCE_STACK_REG(kNumberTypeUInt64, uint64_t),
    SEQUENCE_STACK_REG(kNumberTypeFloat16, float16),       SEQUENCE_STACK_REG(kNumberTypeFloat32, float),
    SEQUENCE_STACK_REG(kNumberTypeFloat64, double),        SEQUENCE_STACK_REG(kNumberTypeComplex64, complex64),
    SEQUENCE_STACK_REG(kNumberTypeComplex128, complex128), SEQUENCE_STACK_REG(kNumberTypeBool, bool)};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SequenceUnstack, SequenceUnstackCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
