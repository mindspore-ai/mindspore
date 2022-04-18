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

#include "plugin/device/cpu/kernel/relu_grad_v2_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore::kernel {
constexpr auto kReluGradV2 = "ReluGradV2";
constexpr const size_t kReluGradV2InputsNum = 2;
constexpr const size_t kReluGradV2OutputsNum = 1;
constexpr size_t kFirstInputIndex = 0;
constexpr size_t kSecondInputIndex = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kInputDims = 4;

template <typename T>
class ReluGradV2CpuKernelFunc : public CpuKernelFunc {
 public:
  ReluGradV2CpuKernelFunc() = default;
  ~ReluGradV2CpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  std::string kernel_name_;
};

template <typename T>
void ReluGradV2CpuKernelFunc<T>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  MS_EXCEPTION_IF_CHECK_FAIL(
    input_shape.size() == kInputDims,
    "The input shape dims of ReluGradV2 should be 4, but got : " + std::to_string(input_shape.size()));
}

template <typename T>
bool ReluGradV2CpuKernelFunc<T>::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReluGradV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReluGradV2OutputsNum, kernel_name_);
  auto *dy = reinterpret_cast<T *>(inputs[kFirstInputIndex]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dy, false);
  auto *mask = reinterpret_cast<uint8_t *>(inputs[kSecondInputIndex]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(mask, false);
  auto *dx = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(dx, false);

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [dy, mask, dx](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      dx[i] = (mask[i] == 1) ? dy[i] : static_cast<T>(0);
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeReluGradV2CpuKernelFunc() {
  return std::make_shared<ReluGradV2CpuKernelFunc<T>>();
}
using ReluGradV2FuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::vector<std::pair<KernelAttr, ReluGradV2FuncCreator>> func_list = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16),
   SpecializeReluGradV2CpuKernelFunc<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
   SpecializeReluGradV2CpuKernelFunc<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64),
   SpecializeReluGradV2CpuKernelFunc<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt8),
   SpecializeReluGradV2CpuKernelFunc<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt16),
   SpecializeReluGradV2CpuKernelFunc<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   SpecializeReluGradV2CpuKernelFunc<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   SpecializeReluGradV2CpuKernelFunc<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReluGradV2CpuKernelFunc<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt16),
   SpecializeReluGradV2CpuKernelFunc<uint16_t>}};

std::vector<KernelAttr> ReluGradV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list.begin(), func_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReluGradV2FuncCreator> &pair) { return pair.first; });
  return support_list;
}

void ReluGradV2CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(EXCEPTION) << "ReluGradV2 does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = func_list[pair.second].second();
  func_obj_->InitFunc(kernel_node);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReluGradV2,
                                 []() { return std::make_shared<ReluGradV2CpuKernelMod>(kReluGradV2); });
}  // namespace mindspore::kernel
