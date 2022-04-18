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

#include "plugin/device/cpu/kernel/relu_v2_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore::kernel {
constexpr auto kReLUV2 = "ReLUV2";
constexpr const size_t kReLUV2InputsNum = 1;
constexpr const size_t kReLUV2OutputsNum = 2;
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;
constexpr size_t kMaskIndex = 1;
constexpr size_t kInputDims = 4;

template <typename T>
class ReLUV2CpuKernelFunc : public CpuKernelFunc {
 public:
  ReLUV2CpuKernelFunc() = default;
  ~ReLUV2CpuKernelFunc() override = default;
  void InitFunc(const CNodePtr &kernel_node) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  std::string kernel_name_;
};

template <typename T>
void ReLUV2CpuKernelFunc<T>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  MS_EXCEPTION_IF_CHECK_FAIL(
    input_shape.size() == kInputDims,
    "The input shape dims of ReluGradV2 should be 4, but got : " + std::to_string(input_shape.size()));
}

template <typename T>
bool ReLUV2CpuKernelFunc<T>::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<AddressPtr> &workspace,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReLUV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReLUV2OutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[kInputIndex]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto *output = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);
  auto *mask = reinterpret_cast<uint8_t *>(outputs[kMaskIndex]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(mask, false);

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [input, mask, output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      T v = input[i];
      bool p = v > static_cast<T>(0);
      mask[i] = static_cast<uint8_t>(p);
      output[i] = p ? v : static_cast<T>(0);
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeReLUV2CpuKernelFunc() {
  return std::make_shared<ReLUV2CpuKernelFunc<T>>();
}
using ReLUV2FuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::vector<std::pair<KernelAttr, ReLUV2FuncCreator>> func_list = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8),
   SpecializeReLUV2CpuKernelFunc<uint16_t>}};

std::vector<KernelAttr> ReLUV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list.begin(), func_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReLUV2FuncCreator> &pair) { return pair.first; });
  return support_list;
}

void ReLUV2CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(EXCEPTION) << "ReLUV2 does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = func_list[pair.second].second();
  func_obj_->InitFunc(kernel_node);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLUV2,
                                 []() { return std::make_shared<ReLUV2CpuKernelMod>(kReLUV2); });
}  // namespace mindspore::kernel
