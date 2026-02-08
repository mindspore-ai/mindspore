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

#include "kernel/gpu/cuda/nn/maxunpool3d_gpu_kernel.h"
#include <utility>
namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateMaxUnpool3DKernelPtr(const std::string &kernel_name,
                                                                          const uint32_t &device_id) {
  return std::make_unique<cukernel::MaxUnpool3DHelperGpuKernel<T, S>>(kernel_name, device_id);
}
using MaxUnpool3DPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, MaxUnpool3DPtrCreatorFunc>> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   CreateMaxUnpool3DKernelPtr<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   CreateMaxUnpool3DKernelPtr<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   CreateMaxUnpool3DKernelPtr<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   CreateMaxUnpool3DKernelPtr<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   CreateMaxUnpool3DKernelPtr<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   CreateMaxUnpool3DKernelPtr<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   CreateMaxUnpool3DKernelPtr<uint64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   CreateMaxUnpool3DKernelPtr<uint64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   CreateMaxUnpool3DKernelPtr<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   CreateMaxUnpool3DKernelPtr<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   CreateMaxUnpool3DKernelPtr<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   CreateMaxUnpool3DKernelPtr<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   CreateMaxUnpool3DKernelPtr<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   CreateMaxUnpool3DKernelPtr<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   CreateMaxUnpool3DKernelPtr<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   CreateMaxUnpool3DKernelPtr<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   CreateMaxUnpool3DKernelPtr<half, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   CreateMaxUnpool3DKernelPtr<half, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   CreateMaxUnpool3DKernelPtr<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   CreateMaxUnpool3DKernelPtr<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   CreateMaxUnpool3DKernelPtr<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   CreateMaxUnpool3DKernelPtr<double, int64_t>}};
}  // namespace

bool MaxUnpool3DGPUKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool MaxUnpool3DGPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  attr_ptr_->data_format = GetValue<std::string>(primitive_->GetAttr(ops::kFormat));
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);
  return true;
}

int MaxUnpool3DGPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_maxunpool3d_shapes;
  std::vector<std::vector<int64_t>> output_maxunpool3d_shapes;
  std::vector<int64_t> inp_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> indices_shape = inputs[1]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[0]->GetShapeVector();
  input_maxunpool3d_shapes.emplace_back(inp_shape);
  input_maxunpool3d_shapes.emplace_back(indices_shape);
  output_maxunpool3d_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_maxunpool3d_shapes, output_maxunpool3d_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> MaxUnpool3DGPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxUnpool3DPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxUnpool3D, MaxUnpool3DGPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
