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

#include "kernel/gpu/cuda/nn/multilabel_margin_loss_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "mindspore/ops/infer/multilabel_margin_loss.h"
#include "kernel/gpu/cuda_impl/cuda_ops/multilabel_margin_loss_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateMultilabelMarginLossKernelPtr(const std::string &kernel_name,
                                                                                   const uint32_t &device_id) {
  return std::make_unique<cukernel::MultilabelMarginLossHelperGpuKernel<T>>(kernel_name, device_id);
}
using MultilabelMarginLossPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, MultilabelMarginLossPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32),
   CreateMultilabelMarginLossKernelPtr<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateMultilabelMarginLossKernelPtr<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateMultilabelMarginLossKernelPtr<double>}};
}  // namespace

bool MultilabelMarginLossGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
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

bool MultilabelMarginLossGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
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

int MultilabelMarginLossGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> inp_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> tgt_shape = inputs[1]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[0]->GetShapeVector();
  std::vector<int64_t> is_tgt_shape = outputs[1]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);
  input_shapes.emplace_back(tgt_shape);
  output_shapes.emplace_back(out_shape);
  output_shapes.emplace_back(is_tgt_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> MultilabelMarginLossGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MultilabelMarginLossPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MultilabelMarginLoss, MultilabelMarginLossGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
