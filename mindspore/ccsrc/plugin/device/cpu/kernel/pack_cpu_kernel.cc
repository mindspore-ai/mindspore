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

#include "plugin/device/cpu/kernel/pack_cpu_kernel.h"
#include <thread>
#include <algorithm>
#include <string>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPackOutputsNum = 1;

template <typename T>
class PackFwdCpuKernelFunc : public CpuKernelFunc {
 public:
  PackFwdCpuKernelFunc() = default;
  ~PackFwdCpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  void PackTensor(T *output, size_t start, size_t end) const;

  int axis_{0};
  size_t input_num_{1};
  size_t output_size_{0};
  size_t dims_behind_axis_{1};
  std::unique_ptr<T *[]> inputs_host_ { nullptr };
  std::string kernel_name_;
};

template <typename T>
void PackFwdCpuKernelFunc<T>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
  axis_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  if (axis_ < 0) {
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    axis_ += (SizeToInt(input_shape.size()) + 1);
  }

  dims_behind_axis_ = 1;
  // calculate elements while dim >= axis
  auto first_input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  for (size_t i = IntToSize(axis_); i < first_input_shape.size(); i++) {
    dims_behind_axis_ *= first_input_shape[i];
  }

  auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  output_size_ = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_size_ *= output_shape[i];
  }
}

template <typename T>
bool PackFwdCpuKernelFunc<T>::RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPackOutputsNum, kernel_name_);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  inputs_host_ = std::make_unique<T *[]>(input_num_);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs_host_[i] = reinterpret_cast<T *>(inputs[i]->addr);
  }

  // multi-threading
  size_t input_size = output_size_;
  auto task = [this, &output](size_t start, size_t end) { PackTensor(output, start, end); };
  ParallelLaunchAutoSearch(task, input_size, this, &parallel_search_info_);
  return true;
}

template <typename T>
void PackFwdCpuKernelFunc<T>::PackTensor(T *output, size_t start, size_t end) const {
  for (size_t pos = start; pos < end; ++pos) {
    size_t cur_input_index = pos / dims_behind_axis_ % input_num_;
    size_t cycle_len = input_num_ * dims_behind_axis_;
    size_t local_index = pos / cycle_len * dims_behind_axis_ + pos % cycle_len % dims_behind_axis_;
    output[pos] = inputs_host_[cur_input_index][local_index];
  }
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializePackFwdFunc() {
  return std::make_shared<PackFwdCpuKernelFunc<T>>();
}
using SpecializePackFwdFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
std::vector<std::pair<KernelAttr, SpecializePackFwdFuncCreator>> func_class_list = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   SpecializePackFwdFunc<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   SpecializePackFwdFunc<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   SpecializePackFwdFunc<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   SpecializePackFwdFunc<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   SpecializePackFwdFunc<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   SpecializePackFwdFunc<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   SpecializePackFwdFunc<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   SpecializePackFwdFunc<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   SpecializePackFwdFunc<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   SpecializePackFwdFunc<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   SpecializePackFwdFunc<bool>}};
}  // namespace

void PackFwdCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_class_list.begin(), func_class_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpecializePackFwdFuncCreator> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = func_class_list[index].second();
  func_obj_->InitFunc(kernel_node);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Stack, PackFwdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
