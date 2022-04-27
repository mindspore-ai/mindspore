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

#include "plugin/device/cpu/kernel/ger_cpu_kernel.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kGerInputsNum = 2;
const size_t kGerOutputsNum = 1;

template <typename T>
class GerCpuTypeFunc : public CpuKernelFunc {
 public:
  GerCpuTypeFunc() = default;
  ~GerCpuTypeFunc() override = default;
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
    CHECK_KERNEL_INPUTS_NUM(input_num, kGerInputsNum, common::AnfAlgo::GetCNodeName(kernel_node));
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    CHECK_KERNEL_OUTPUTS_NUM(output_num, kGerOutputsNum, common::AnfAlgo::GetCNodeName(kernel_node));
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

    compute_func_ = &GerCpuTypeFunc<T>::Compute;
  }

 private:
  std::string kernel_name_;
  TypeId input_type_1_{kTypeUnknown};
  TypeId input_type_2_{kTypeUnknown};
  std::vector<size_t> input_shape_1_;
  std::vector<size_t> input_shape_2_;
  std::vector<size_t> output_shape_;

  void Compute(const T *input1, const T *input2, T *output);

  using TypeComputeFunc = std::function<void(GerCpuTypeFunc *, const T *, const T *, T *)>;
  TypeComputeFunc compute_func_{nullptr};
};

template <typename T>
void GerCpuTypeFunc<T>::Compute(const T *input1, const T *input2, T *output) {
  if (output_shape_.size() == 0) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }
  size_t output_size_ = 1;
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ *= output_shape_[i];
  }
  size_t input2_size_ = input_shape_2_[0];
  auto task = [&input1, &input2, &output, input2_size_](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t input1_index = static_cast<size_t>(i / input2_size_);
      size_t input2_index = static_cast<size_t>(i % input2_size_);
      output[i] = static_cast<T>(input1[input1_index] * input2[input2_index]);
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeGerFunc() {
  return std::make_shared<GerCpuTypeFunc<T>>();
}
using GerCpuFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::vector<std::pair<KernelAttr, GerCpuFuncCreator>> kernel_attr_lists = {
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    SpecializeGerFunc<float16>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    SpecializeGerFunc<float>}}};
}  // namespace

void GerCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);

  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', kernel type must be '" << kernel_name_ << "', but got "
                      << kernel_type_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_lists[index].second();
  func_obj_->InitFunc(kernel_node);
}

std::vector<KernelAttr> GerCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr_lists.begin(), kernel_attr_lists.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GerCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Ger,
                                 []() { return std::make_shared<GerCpuKernelMod>(prim::kPrimGer->name()); });
}  // namespace kernel
}  // namespace mindspore
