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

#include "plugin/device/gpu/kernel/math/gcd_lcm_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kGcd = "Gcd";
constexpr auto kLcm = "Lcm";

template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateGcdLcmKernelPtr(const std::string &kernel_name,
                                                                     const uint32_t &device_id) {
  return std::make_unique<cukernel::GcdLcmHelperGpuKernel<T>>(kernel_name, device_id);
}
using GcdLcmPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::map<std::string, std::vector<std::pair<KernelAttr, GcdLcmPtrCreatorFunc>>> kernel_attr_map = {
  {kGcd,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     CreateGcdLcmKernelPtr<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     CreateGcdLcmKernelPtr<int64_t>}}},
  {kLcm,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     CreateGcdLcmKernelPtr<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     CreateGcdLcmKernelPtr<int64_t>}}}};
}  // namespace

bool GcdLcmGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool GcdLcmGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr_map.at(kernel_type_)[index].second(kernel_name_, device_id_));

  Resize(base_operator, inputs, outputs);
  return true;
}

int GcdLcmGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> inpx1_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<int64_t> inpx2_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<int64_t> y_shape = outputs.at(kIndex0)->GetShapeVector();
  input_shapes.emplace_back(inpx1_shape);
  input_shapes.emplace_back(inpx2_shape);
  output_shapes.emplace_back(y_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> GcdLcmGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map.find(kernel_type_);
  if (iter == kernel_attr_map.end()) {
    MS_LOG(ERROR) << "For 'GcdLcmOp', only support these types: " << kernel::Map2Str(kernel_attr_map)
                  << " currently, but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GcdLcmPtrCreatorFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Gcd, []() { return std::make_shared<GcdLcmGpuKernelMod>(kGcd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Lcm, []() { return std::make_shared<GcdLcmGpuKernelMod>(kLcm); });
}  // namespace kernel
}  // namespace mindspore
