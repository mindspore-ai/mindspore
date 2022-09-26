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

#include "plugin/device/gpu/kernel/nn/dilation2d_backprop_filter_gpu_kernel.h"
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t InputIndex = 0;
constexpr size_t FilterIndex = 1;
constexpr size_t Out_backpropIndex = 2;
constexpr size_t OutputIndex = 0;

template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateDilation2DBackpropFilterKernelPtr(const std::string &kernel_name,
                                                                                       const uint32_t &device_id) {
  return std::make_unique<cukernel::Dilation2DBackpropFilterHelperGpuKernel<T>>(kernel_name, device_id);
}
using Dilation2DBackpropFilterPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, Dilation2DBackpropFilterPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateDilation2DBackpropFilterKernelPtr<half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateDilation2DBackpropFilterKernelPtr<float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateDilation2DBackpropFilterKernelPtr<double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateDilation2DBackpropFilterKernelPtr<int32_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateDilation2DBackpropFilterKernelPtr<int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   CreateDilation2DBackpropFilterKernelPtr<uint8_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   CreateDilation2DBackpropFilterKernelPtr<uint16_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   CreateDilation2DBackpropFilterKernelPtr<uint32_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   CreateDilation2DBackpropFilterKernelPtr<uint64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   CreateDilation2DBackpropFilterKernelPtr<int8_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   CreateDilation2DBackpropFilterKernelPtr<int16_t>}};
}  // namespace

bool Dilation2DBackpropFilterGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
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

bool Dilation2DBackpropFilterGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_dilation2d_backprop_filter_ptr = std::dynamic_pointer_cast<ops::Dilation2DBackpropFilter>(base_operator);
  kernel_name_ = kernel_dilation2d_backprop_filter_ptr->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  attr_ptr_->stride = kernel_dilation2d_backprop_filter_ptr->get_stride();
  attr_ptr_->dilation = kernel_dilation2d_backprop_filter_ptr->get_dilation();
  attr_ptr_->pad_mode = kernel_dilation2d_backprop_filter_ptr->get_pad_mode();
  attr_ptr_->format = kernel_dilation2d_backprop_filter_ptr->get_format();
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);

  Resize(base_operator, inputs, outputs);
  return true;
}

int Dilation2DBackpropFilterGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
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
  std::vector<int64_t> inp_shape = inputs[InputIndex]->GetShapeVector();
  std::vector<int64_t> fil_shape = inputs[FilterIndex]->GetShapeVector();
  std::vector<int64_t> out_backprop_shape = inputs[Out_backpropIndex]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[OutputIndex]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);
  input_shapes.emplace_back(fil_shape);
  input_shapes.emplace_back(out_backprop_shape);
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> Dilation2DBackpropFilterGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, Dilation2DBackpropFilterPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Dilation2DBackpropFilter, Dilation2DBackpropFilterGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
