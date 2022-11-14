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

#include "plugin/device/gpu/kernel/math/histogram_fixed_width_gpu_kernel.h"
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateHistogramFixedWidthKernelPtr(const std::string &kernel_name,
                                                                                  const uint32_t &device_id) {
  return std::make_unique<cukernel::HistogramFixedWidthHelperGpuKernel<T>>(kernel_name, device_id);
}
using HistogramFixedWidthPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, HistogramFixedWidthPtrCreatorFunc>> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   CreateHistogramFixedWidthKernelPtr<int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   CreateHistogramFixedWidthKernelPtr<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   CreateHistogramFixedWidthKernelPtr<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   CreateHistogramFixedWidthKernelPtr<half>}};
}  // namespace

bool HistogramFixedWidthGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
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

bool HistogramFixedWidthGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::HistogramFixedWidth>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_ptr;
    return false;
  }
  attr_ptr_->nbins = kernel_ptr->get_nbins();
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);
  return true;
}

int HistogramFixedWidthGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> inp_x_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<int64_t> inp_range_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<int64_t> out_shape = outputs.at(kIndex0)->GetShapeVector();
  input_shapes.emplace_back(inp_x_shape);
  input_shapes.emplace_back(inp_range_shape);
  output_shapes.emplace_back(out_shape);

  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> HistogramFixedWidthGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, HistogramFixedWidthPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, HistogramFixedWidth, HistogramFixedWidthGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
