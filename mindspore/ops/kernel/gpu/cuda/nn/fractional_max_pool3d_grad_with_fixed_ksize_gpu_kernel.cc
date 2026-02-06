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

#include "kernel/gpu/cuda/nn/fractional_max_pool3d_grad_with_fixed_ksize_gpu_kernel.h"
#include <utility>
#include <iostream>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOriginInputIndex = 0;
constexpr size_t kOutBackpropIndex = 1;
constexpr size_t kArgmaxIndex = 2;
constexpr size_t kOutputIndex = 0;

template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr(
  const std::string &kernel_name, const uint32_t &device_id) {
  return std::make_unique<cukernel::FractionalMaxPool3DGradWithFixedKsizeHelperGpuKernel<T, S>>(kernel_name, device_id);
}
using FractionalMaxPool3DGradWithFixedKsizePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;
const std::vector<std::pair<KernelAttr, FractionalMaxPool3DGradWithFixedKsizePtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DGradWithFixedKsizeKernelPtr<int64_t, int64_t>}};
}  // namespace

bool FractionalMaxPool3DGradWithFixedKsizeGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                               const std::vector<KernelTensor *> &workspace,
                                                               const std::vector<KernelTensor *> &outputs,
                                                               void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool FractionalMaxPool3DGradWithFixedKsizeGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                             const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  attr_ptr_->data_format = GetValue<std::string>(primitive_->GetAttr(ops::kFormat));
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);

  Resize(inputs, outputs);
  return true;
}

int FractionalMaxPool3DGradWithFixedKsizeGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                              const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> origin_input_shape = inputs[kOriginInputIndex]->GetShapeVector();
  std::vector<int64_t> out_backprop_shape = inputs[kOutBackpropIndex]->GetShapeVector();
  std::vector<int64_t> argmax_shape = inputs[kArgmaxIndex]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[kOutputIndex]->GetShapeVector();
  input_shapes.emplace_back(origin_input_shape);
  input_shapes.emplace_back(out_backprop_shape);
  input_shapes.emplace_back(argmax_shape);
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> FractionalMaxPool3DGradWithFixedKsizeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, FractionalMaxPool3DGradWithFixedKsizePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FractionalMaxPool3DGradWithFixedKsize,
                      FractionalMaxPool3DGradWithFixedKsizeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
