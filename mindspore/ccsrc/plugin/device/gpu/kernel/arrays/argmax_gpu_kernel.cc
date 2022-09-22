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

#include "plugin/device/gpu/kernel/arrays/argmax_gpu_kernel.h"
#include <utility>
namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateArgMaxKernelPtr(const std::string &kernel_name,
                                                                     const uint32_t &device_id) {
  return std::make_unique<cukernel::ArgMaxHelperGpuKernel<T, S>>(kernel_name, device_id);
}
using ArgMaxPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, ArgMaxPtrCreatorFunc>> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32), CreateArgMaxKernelPtr<half, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32), CreateArgMaxKernelPtr<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
   CreateArgMaxKernelPtr<uint64_t, int32_t>},

  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64), CreateArgMaxKernelPtr<half, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64), CreateArgMaxKernelPtr<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
   CreateArgMaxKernelPtr<uint64_t, int64_t>}};
}  // namespace

bool ArgmaxGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool ArgmaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Argmax>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  attr_ptr_->axis = kernel_ptr->get_axis();
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);

  Resize(base_operator, inputs, outputs);
  return true;
}

int ArgmaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> inp_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[0]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> ArgmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ArgMaxPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Argmax, ArgmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
