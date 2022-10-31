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

#include "plugin/device/cpu/kernel/unpack_cpu_kernel.h"
#include <map>
#include <tuple>
#include "ops/unstack.h"
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUnpackInputsNum = 1;
constexpr size_t kUnpackOutputsMinNum = 1;
constexpr size_t kUnpackWorkspaceMinNum = 1;
constexpr size_t kMaxDataSize = 2147483648;  // 2GB
}  // namespace

bool UnpackCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Unstack>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "cast unstack ops failed!";
    return false;
  }
  unstack_param_.axis_ = kernel_ptr->get_axis();
  origin_axis_ = unstack_param_.axis_;
  unstack_param_.pre_dims_ = 1;
  unstack_param_.axis_dim_ = 1;
  unstack_param_.after_dims_ = 1;
  input_size_ = 1;

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::tuple<KernelAttr, UnstackFunc, InitIOFunc> &tuple_item) { return std::get<0>(tuple_item); });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(ERROR) << "Unstack does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = std::get<1>(func_list_[index]);
  const size_t kTwoIdx = 2;
  init_io_func_ = std::get<kTwoIdx>(func_list_[index]);
  return true;
}

int UnpackCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  MS_EXCEPTION_IF_NULL(base_operator);
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }

  input_size_ = 1;
  unstack_param_.pre_dims_ = 1;
  unstack_param_.axis_dim_ = 1;
  unstack_param_.after_dims_ = 1;
  unstack_param_.axis_ = origin_axis_;

  auto input_shape = inputs[0]->GetShapeVector();
  if (unstack_param_.axis_ < 0) {
    unstack_param_.axis_ += input_shape.size();
  }
  output_num_ = input_shape[unstack_param_.axis_];
  unstack_param_.num_ = SizeToInt(output_num_);

  for (size_t i = 0; i < input_shape.size(); i++) {
    if (i < IntToSize(unstack_param_.axis_)) {
      unstack_param_.pre_dims_ *= LongToInt(input_shape[i]);
    } else if (i > IntToSize(unstack_param_.axis_)) {
      unstack_param_.after_dims_ *= LongToInt(input_shape[i]);
    } else {
      unstack_param_.axis_dim_ = LongToInt(input_shape[i]);
    }
    input_size_ *= LongToSize(input_shape[i]);
  }
  init_io_func_(this);
  return KRET_OK;
}

template <typename T>
void UnpackCpuKernelMod::InitIOSize() {
  (void)workspace_size_list_.emplace_back(sizeof(T *) * output_num_);
}

bool UnpackCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUnpackInputsNum, kernel_name_);
  if (outputs.size() < kUnpackOutputsMinNum || workspace.size() < kUnpackWorkspaceMinNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the number of outputs and workspaces must be at least 1, but got the number of outputs: "
                      << outputs.size() << " and the number of workspaces: " << workspace.size();
  }
  return kernel_func_(this, inputs, workspace, outputs);
}

template <typename T>
bool UnpackCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  const void *input = reinterpret_cast<void *>(inputs[0]->addr);
  void **outputs_host = reinterpret_cast<void **>(workspace[0]->addr);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs_host[i] = reinterpret_cast<T *>(outputs[i]->addr);
  }

  size_t total_size = input_size_ * sizeof(T);
  if (total_size >= kMaxDataSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the input data size cannot larger than 2GB, but got "
                      << total_size << " bytes";
  }
  int data_size = SizeToInt(sizeof(T));
  Unstack(input, outputs_host, &unstack_param_, data_size);
  return true;
}

std::vector<std::tuple<KernelAttr, UnpackCpuKernelMod::UnstackFunc, UnpackCpuKernelMod::InitIOFunc>>
  UnpackCpuKernelMod::func_list_ = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &UnpackCpuKernelMod::LaunchKernel<int8_t>, &UnpackCpuKernelMod::InitIOSize<int8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &UnpackCpuKernelMod::LaunchKernel<int16_t>, &UnpackCpuKernelMod::InitIOSize<int16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &UnpackCpuKernelMod::LaunchKernel<int>, &UnpackCpuKernelMod::InitIOSize<int>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &UnpackCpuKernelMod::LaunchKernel<int64_t>, &UnpackCpuKernelMod::InitIOSize<int64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &UnpackCpuKernelMod::LaunchKernel<bool>, &UnpackCpuKernelMod::InitIOSize<bool>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &UnpackCpuKernelMod::LaunchKernel<uint8_t>, &UnpackCpuKernelMod::InitIOSize<uint8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &UnpackCpuKernelMod::LaunchKernel<uint16_t>, &UnpackCpuKernelMod::InitIOSize<uint16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &UnpackCpuKernelMod::LaunchKernel<uint32_t>, &UnpackCpuKernelMod::InitIOSize<uint32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &UnpackCpuKernelMod::LaunchKernel<uint64_t>, &UnpackCpuKernelMod::InitIOSize<uint64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &UnpackCpuKernelMod::LaunchKernel<complex64>, &UnpackCpuKernelMod::InitIOSize<complex64>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &UnpackCpuKernelMod::LaunchKernel<complex128>, &UnpackCpuKernelMod::InitIOSize<complex128>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &UnpackCpuKernelMod::LaunchKernel<float16>, &UnpackCpuKernelMod::InitIOSize<float16>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &UnpackCpuKernelMod::LaunchKernel<float>, &UnpackCpuKernelMod::InitIOSize<float>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &UnpackCpuKernelMod::LaunchKernel<double>, &UnpackCpuKernelMod::InitIOSize<double>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Unstack, UnpackCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
