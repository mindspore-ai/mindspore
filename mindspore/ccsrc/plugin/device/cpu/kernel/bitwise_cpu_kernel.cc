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

#include "plugin/device/cpu/kernel/bitwise_cpu_kernel.h"

#include <string>
#include <vector>
#include <cmath>
#include <type_traits>
#include <unordered_map>
#include <memory>
#include <map>
#include <functional>
#include <algorithm>
#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kBitwiseInputsNum = 2;
const size_t kBitwiseOutputsNum = 1;

template <typename T>
class BitwiseCpuTypeFunc : public CpuKernelFunc {
 public:
  BitwiseCpuTypeFunc() = default;
  ~BitwiseCpuTypeFunc() override = default;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
               const std::vector<AddressPtr> &outputs) override {
    const auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
    const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
    auto *output = reinterpret_cast<T *>(outputs[0]->addr);
    compute_func_(this, input1, input2, output);
    return true;
  }

  void InitFunc(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    CHECK_KERNEL_INPUTS_NUM(input_num, kBitwiseInputsNum, common::AnfAlgo::GetCNodeName(kernel_node));
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    CHECK_KERNEL_OUTPUTS_NUM(output_num, kBitwiseOutputsNum, common::AnfAlgo::GetCNodeName(kernel_node));
    input_type_1_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    input_type_2_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
    if (input_type_1_ != input_type_2_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', input1 and input2 must have the same type. But got input1 type " << input_type_1_
                        << ", input2 type " << input_type_2_;
    }
    input_shape_1_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_shape_2_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);

    static const std::unordered_map<std::string, TypeComputeFunc> bitwise_func_map{
      {prim::kPrimBitwiseAnd->name(), &BitwiseCpuTypeFunc<T>::BitwiseCompute},
      {prim::kPrimBitwiseOr->name(), &BitwiseCpuTypeFunc<T>::BitwiseCompute},
      {prim::kPrimBitwiseXor->name(), &BitwiseCpuTypeFunc<T>::BitwiseCompute}};
    if (bitwise_func_map.find(kernel_name_) == bitwise_func_map.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', only supports operators in "
                        << Unorderedmap2Str(bitwise_func_map) << ", but got " << kernel_name_;
    }
    compute_func_ = bitwise_func_map.at(kernel_name_);
  }

 private:
  std::string kernel_name_;
  TypeId input_type_1_{kTypeUnknown};
  TypeId input_type_2_{kTypeUnknown};
  std::vector<size_t> input_shape_1_;
  std::vector<size_t> input_shape_2_;
  std::vector<size_t> output_shape_;

  void BitwiseCompute(const T *input1, const T *input2, T *output);

  using TypeComputeFunc = std::function<void(BitwiseCpuTypeFunc *, const T *, const T *, T *)>;
  TypeComputeFunc compute_func_{nullptr};
};

template <typename T>
void BitwiseCpuTypeFunc<T>::BitwiseCompute(const T *input1, const T *input2, T *output) {
  if (output_shape_.size() == 0) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }
  size_t output_size_ = 1;
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ *= output_shape_[i];
  }
  BroadcastIterator base_iter(input_shape_1_, input_shape_2_, output_shape_);
  auto task = [this, &input1, &input2, &output, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      T y_val = (input2[iter.GetInputPosB()]);
      T bit_val = static_cast<T>(sizeof(T) * 8 - 1);
      if (y_val > bit_val) {
        y_val = bit_val;
      }
      if (this->kernel_name_.compare(prim::kPrimBitwiseAnd->name()) == 0) {
        output[i] = static_cast<T>(input1[iter.GetInputPosA()] & y_val);
      } else if (this->kernel_name_.compare(prim::kPrimBitwiseOr->name()) == 0) {
        output[i] = static_cast<T>(input1[iter.GetInputPosA()] | y_val);
      } else if (this->kernel_name_.compare(prim::kPrimBitwiseXor->name()) == 0) {
        output[i] = static_cast<T>(input1[iter.GetInputPosA()] ^ y_val);
      } else {
        MS_LOG(EXCEPTION) << "For '" << this->kernel_name_ << "', kernel name should be '" << this->kernel_name_
                          << "', but got " << this->kernel_name_;
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeBitwiseFunc() {
  return std::make_shared<BitwiseCpuTypeFunc<T>>();
}
using BitwiseCpuFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, BitwiseCpuFuncCreator>>> kernel_attr_lists = {
  {prim::kPrimBitwiseAnd->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     SpecializeBitwiseFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     SpecializeBitwiseFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeBitwiseFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeBitwiseFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     SpecializeBitwiseFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     SpecializeBitwiseFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     SpecializeBitwiseFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     SpecializeBitwiseFunc<uint64_t>}}},
  {prim::kPrimBitwiseOr->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     SpecializeBitwiseFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     SpecializeBitwiseFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeBitwiseFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeBitwiseFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     SpecializeBitwiseFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     SpecializeBitwiseFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     SpecializeBitwiseFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     SpecializeBitwiseFunc<uint64_t>}}},
  {prim::kPrimBitwiseXor->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     SpecializeBitwiseFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     SpecializeBitwiseFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeBitwiseFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeBitwiseFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     SpecializeBitwiseFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     SpecializeBitwiseFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     SpecializeBitwiseFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     SpecializeBitwiseFunc<uint64_t>}}}};
}  // namespace

void BitwiseCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);

  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', kernel type should be '" << kernel_name_ << "', but got "
                      << kernel_type_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_lists[kernel_name_][index].second();
  func_obj_->InitFunc(kernel_node);
}

std::vector<KernelAttr> BitwiseCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_lists.find(kernel_type_);
  if (iter == kernel_attr_lists.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', kernel type should be '" << kernel_name_ << "', but got "
                      << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BitwiseCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BitwiseAnd,
                                 []() { return std::make_shared<BitwiseCpuKernelMod>(prim::kPrimBitwiseAnd->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BitwiseOr,
                                 []() { return std::make_shared<BitwiseCpuKernelMod>(prim::kPrimBitwiseOr->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, BitwiseXor,
                                 []() { return std::make_shared<BitwiseCpuKernelMod>(prim::kPrimBitwiseXor->name()); });
}  // namespace kernel
}  // namespace mindspore
