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
#include "plugin/device/cpu/kernel/bincount_cpu_kernel.h"

#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace kernel {
bool BincountCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  dt_arr_ = inputs[kIndex0]->GetDtype();
  dt_weights_ = inputs[kIndex2]->GetDtype();
  return true;
}

int BincountCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_arr_sizes_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_size_sizes_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  input_weights_sizes_ = inputs[kIndex2]->GetDeviceShapeAdaptively();
  output_sizes_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename T_in, typename T_out>
void BincountTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                  const std::vector<AddressPtr> &outputs, const std::vector<int64_t> &input_arr_sizes, int32_t num_bins,
                  const std::vector<int64_t> &input_weights_sizes, const std::vector<int64_t> &) {
  auto bin_array = static_cast<T_in *>(inputs[0]->addr);
  auto output_data = static_cast<T_out *>(outputs[0]->addr);
  const size_t data_num = SizeOf(input_arr_sizes);
  for (int32_t i = 0; i < num_bins; i++) {
    output_data[i] = 0;
  }
  if (input_weights_sizes.size() != 0 && input_weights_sizes[0] == 0) {
    for (size_t i = 0; i < data_num; i++) {
      T_in value = bin_array[i];
      if (value < num_bins) {
        output_data[value] += T_out(1);
      }
    }
  } else {
    auto bin_weights = static_cast<T_out *>(inputs[2]->addr);
    for (size_t i = 0; i < data_num; i++) {
      T_in value = bin_array[i];
      if (value < num_bins) {
        output_data[value] += bin_weights[i];
      }
    }
  }
}

void BincountCpuKernelMod::SetMap() {
  calls_[kNumberTypeInt32][kNumberTypeFloat32] = BincountTask<int32_t, float>;
  calls_[kNumberTypeInt32][kNumberTypeInt32] = BincountTask<int32_t, int32_t>;
  calls_[kNumberTypeInt32][kNumberTypeInt64] = BincountTask<int32_t, int64_t>;
  calls_[kNumberTypeInt32][kNumberTypeFloat64] = BincountTask<int32_t, double>;
}

bool BincountCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                                  const std::vector<AddressPtr> &outputs) {
  const size_t array_num = SizeOf(input_arr_sizes_);
  const size_t weights_num = SizeOf(input_weights_sizes_);
  if (array_num != weights_num) {
    MS_LOG(EXCEPTION) << "For Bincount, the size of input_weights " << Vector2Str(input_arr_sizes_)
                      << " need be the same with input_arr " << Vector2Str(input_weights_sizes_);
  }
  MS_EXCEPTION_IF_NULL(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(inputs[1]->addr);
  if (input_size_sizes_.size() != 0) {
    MS_LOG(EXCEPTION) << "For Bincount, input_size should be a scalar, but got rank " << input_size_sizes_.size();
  }
  auto num_bins_ptr = static_cast<int32_t *>(inputs[1]->addr);
  if (*num_bins_ptr < 0) {
    MS_LOG(EXCEPTION) << "For Bincount, input size should be nonnegative, but got" << *num_bins_ptr;
  }
  int32_t num_bins = *num_bins_ptr;

  // check input_arr nonnegative
  auto bin_array = static_cast<int32_t *>(inputs[0]->addr);
  for (size_t i = 0; i < array_num; i++) {
    if (bin_array[i] < 0) {
      MS_LOG(EXCEPTION) << "For Bincount, input array should be nonnegative, but got " << bin_array[i];
    }
  }
  SetMap();
  calls_[dt_arr_][dt_weights_](inputs, workspaces, outputs, input_arr_sizes_, num_bins, input_weights_sizes_,
                               output_sizes_);
  calls_.clear();

  return true;
}

std::vector<KernelAttr> BincountCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Bincount, BincountCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
