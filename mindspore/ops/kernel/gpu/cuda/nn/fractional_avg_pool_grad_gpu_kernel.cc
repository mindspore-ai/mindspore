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

#include "kernel/gpu/cuda/nn/fractional_avg_pool_grad_gpu_kernel.h"
#include <utility>
#include <iostream>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOrigInputIndex = 0;
constexpr size_t kOutBackpropIndex = 1;
constexpr size_t kInputRowPoolingSequenceIndex = 2;
constexpr size_t kInputColPoolingSequenceIndex = 3;
constexpr size_t kOutputIndex = 0;

template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateFractionalAvgPoolGradKernelPtr(const std::string &kernel_name,
                                                                                    const uint32_t &device_id) {
  return std::make_unique<cukernel::FractionalPoolGradHelperGpuKernel<T>>(kernel_name, device_id);
}
using FractionalAvgPoolGradPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;
const std::vector<std::pair<KernelAttr, FractionalAvgPoolGradPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateFractionalAvgPoolGradKernelPtr<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateFractionalAvgPoolGradKernelPtr<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateFractionalAvgPoolGradKernelPtr<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateFractionalAvgPoolGradKernelPtr<int64_t>}};
}  // namespace

bool FractionalAvgPoolGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
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

bool FractionalAvgPoolGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  attr_ptr_->overlapping = GetValue<bool>(primitive_->GetAttr("overlapping"));
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);

  Resize(inputs, outputs);
  return true;
}

int FractionalAvgPoolGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> orig_input_shape = inputs[kOrigInputIndex]->GetShapeVector();
  std::vector<int64_t> out_backprop_shape = inputs[kOutBackpropIndex]->GetShapeVector();
  std::vector<int64_t> row_shape = inputs[kInputRowPoolingSequenceIndex]->GetShapeVector();
  std::vector<int64_t> col_shape = inputs[kInputColPoolingSequenceIndex]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[kOutputIndex]->GetShapeVector();

  input_shapes.emplace_back(orig_input_shape);
  input_shapes.emplace_back(out_backprop_shape);
  input_shapes.emplace_back(row_shape);
  input_shapes.emplace_back(col_shape);
  output_shapes.emplace_back(out_shape);

  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> FractionalAvgPoolGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, FractionalAvgPoolGradPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FractionalAvgPoolGrad, FractionalAvgPoolGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
