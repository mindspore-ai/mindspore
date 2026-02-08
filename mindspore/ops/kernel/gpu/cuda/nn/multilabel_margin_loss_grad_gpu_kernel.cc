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

#include "kernel/gpu/cuda/nn/multilabel_margin_loss_grad_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "mindspore/ops/infer/grad/multilabel_margin_loss_grad.h"
#include "kernel/gpu/cuda_impl/cuda_ops/multilabel_margin_loss_grad_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateMultilabelMarginLossGradKernelPtr(const std::string &kernel_name,
                                                                                       const uint32_t &device_id) {
  return std::make_unique<cukernel::MultilabelMarginLossGradHelperGpuKernel<T>>(kernel_name, device_id);
}
using MultilabelMarginLossGradPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, MultilabelMarginLossGradPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateMultilabelMarginLossGradKernelPtr<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateMultilabelMarginLossGradKernelPtr<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateMultilabelMarginLossGradKernelPtr<double>}};
}  // namespace

bool MultilabelMarginLossGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
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

bool MultilabelMarginLossGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  static std::map<std::string, int64_t> kReductionModeMap{{"none", 0}, {"mean", 1}, {"sum", 2}};
  string reduc_str = GetValue<std::string>(primitive_->GetAttr(ops::kReduction));
  attr_ptr_->reduction = kReductionModeMap[reduc_str];
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);
  return true;
}

int MultilabelMarginLossGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  const int64_t kInputGradIndex = 0;
  const int64_t kInputXIndex = 1;
  const int64_t kInputTargetIndex = 2;
  const int64_t kInputIsTargetIndex = 3;
  const int64_t kOutputGradIndex = 0;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> y_grad_shape = inputs[kInputGradIndex]->GetShapeVector();
  std::vector<int64_t> x_shape = inputs[kInputXIndex]->GetShapeVector();
  std::vector<int64_t> target_shape = inputs[kInputTargetIndex]->GetShapeVector();
  std::vector<int64_t> is_target_shape = inputs[kInputIsTargetIndex]->GetShapeVector();
  std::vector<int64_t> x_grad_shape = outputs[kOutputGradIndex]->GetShapeVector();
  input_shapes.emplace_back(y_grad_shape);
  input_shapes.emplace_back(x_shape);
  input_shapes.emplace_back(target_shape);
  input_shapes.emplace_back(is_target_shape);
  output_shapes.emplace_back(x_grad_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> MultilabelMarginLossGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MultilabelMarginLossGradPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MultilabelMarginLossGrad, MultilabelMarginLossGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
