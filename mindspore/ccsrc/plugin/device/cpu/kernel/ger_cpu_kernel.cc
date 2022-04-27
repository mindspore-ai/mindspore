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
}  // namespace

bool GerCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  if (!base_operator) {
    MS_LOG(ERROR) << "For " << kernel_type_ << ", cast " << kernel_type_ << " ops failed!";
    return false;
  }
  kernel_name_ = base_operator->name();
  if (inputs.size() != kGerInputsNum || outputs.size() != kGerOutputsNum) {
    MS_LOG(ERROR) << "For" << kernel_name_ << ": input and output size should be " << kGerInputsNum << " and "
                  << kGerOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  input_type_1_ = inputs[0]->GetDtype();
  input_type_2_ = inputs[1]->GetDtype();
  if (input_type_1_ != input_type_2_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input1 and input2 must have the same type. But got input1 type "
                  << input_type_1_ << ", input2 type " << input_type_2_;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GerCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs,
                            const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  std::vector<int64_t> input_shape_1 = inputs[0]->GetShapeVector();
  std::vector<int64_t> input_shape_2 = inputs[1]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  auto in_shape_size_1 = input_shape_1.size();
  auto in_shape_size_2 = input_shape_2.size();
  if (in_shape_size_1 != in_shape_size_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input1 shape size should be the same as input2 shape size, but got"
                  << " input1 shape size " << in_shape_size_1 << " input2 shape size " << in_shape_size_2;
    return KRET_RESIZE_FAILED;
  }

  (void)std::transform(input_shape_1.begin(), input_shape_1.end(), std::back_inserter(input_shape_1_), LongToSize);
  (void)std::transform(input_shape_2.begin(), input_shape_2.end(), std::back_inserter(input_shape_2_), LongToSize);
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToSize);

  batches_ = 1;
  for (size_t shape_index = 0; shape_index < input_shape_1_.size() - 1; shape_index++) {
    if (input_shape_1_[shape_index] != input_shape_2_[shape_index]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the " << shape_index << "th dimension of shape size of input1 and"
                    << " input2 should be the same, but got input1 shape size " << input_shape_1_[shape_index]
                    << " input2 shape size " << input_shape_2_[shape_index];
      return KRET_RESIZE_FAILED;
    }
    batches_ *= input_shape_1_[shape_index];
  }

  return KRET_OK;
}

template <typename T>
bool GerCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGerInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGerOutputsNum, kernel_name_);
  T *input1 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  if (output_shape_.size() == 0) {
    (void)output_shape_.insert(output_shape_.begin(), 1);
  }
  size_t output_size_ = 1;
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ *= output_shape_[i];
  }
  size_t input1_size = input_shape_1_[input_shape_1_.size() - 1];
  size_t input2_size = input_shape_2_[input_shape_2_.size() - 1];
  size_t output_size = input1_size * input2_size;
  auto task = [&input1, &input2, &output, input1_size, input2_size, output_size](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t batch_index = i / output_size;
      size_t input1_index = (i % output_size) / input2_size + batch_index * input1_size;
      size_t input2_index = (i % output_size) % input2_size + batch_index * input2_size;
      output[i] = static_cast<T>(input1[input1_index] * input2[input2_index]);
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, GerCpuKernelMod::GerLaunchFunc>> GerCpuKernelMod::func_list_ = {
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    &GerCpuKernelMod::LaunchKernel<float16>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    &GerCpuKernelMod::LaunchKernel<float>}}};

std::vector<KernelAttr> GerCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GerLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Ger,
                                 []() { return std::make_shared<GerCpuKernelMod>(prim::kPrimGer->name()); });
}  // namespace kernel
}  // namespace mindspore
