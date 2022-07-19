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

#include "plugin/device/gpu/kernel/nn/fractional_max_pool3d_with_fixed_ksize_gpu_kernel.h"
#include <utility>
#include <iostream>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kRandomSamplesIndex = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kArgmaxIndex = 1;

template <typename T, typename S, typename G>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateFractionalMaxPool3DWithFixedKsizeKernelPtr(
  const std::string &kernel_name, const uint32_t &device_id) {
  return std::make_unique<cukernel::FractionalMaxPool3DWithFixedKsizeHelperGpuKernel<T, S, G>>(kernel_name, device_id);
}
using FractionalMaxPool3DWithFixedKsizePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;
const std::vector<std::pair<KernelAttr, FractionalMaxPool3DWithFixedKsizePtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<half, half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<half, float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<half, double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<half, half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<half, float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<half, double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<float, half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<float, float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<float, double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<float, half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<float, float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<float, double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<double, half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<double, float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<double, double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<double, half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<double, float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<double, double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int32_t, half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int32_t, float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int32_t, double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int32_t, half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int32_t, float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int32_t, double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int64_t, half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int64_t, float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int64_t, double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int64_t, half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int64_t, float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalMaxPool3DWithFixedKsizeKernelPtr<int64_t, double, int32_t>}};
}  // namespace

bool FractionalMaxPool3DWithFixedKsizeGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                           const std::vector<AddressPtr> &workspace,
                                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool FractionalMaxPool3DWithFixedKsizeGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                         const std::vector<KernelTensorPtr> &inputs,
                                                         const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FractionalMaxPool3DWithFixedKsize>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  attr_ptr_->ksize = kernel_ptr->get_ksize();
  attr_ptr_->output_shape = kernel_ptr->get_output_shape();
  attr_ptr_->data_format = kernel_ptr->get_data_format();
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);

  Resize(base_operator, inputs, outputs);
  return true;
}

int FractionalMaxPool3DWithFixedKsizeGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                          const std::vector<KernelTensorPtr> &inputs,
                                                          const std::vector<KernelTensorPtr> &outputs,
                                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> inp_shape = inputs[kInputIndex]->GetShapeVector();
  std::vector<int64_t> random_samples_shape = inputs[kRandomSamplesIndex]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[kOutputIndex]->GetShapeVector();
  std::vector<int64_t> argmax_shape = outputs[kArgmaxIndex]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);
  input_shapes.emplace_back(random_samples_shape);
  output_shapes.emplace_back(out_shape);
  output_shapes.emplace_back(argmax_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> FractionalMaxPool3DWithFixedKsizeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, FractionalMaxPool3DWithFixedKsizePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FractionalMaxPool3DWithFixedKsize,
                      FractionalMaxPool3DWithFixedKsizeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
