/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/math/isclose_gpu_kernel.h"
#include <utility>
#include <map>
namespace mindspore {
namespace kernel {
constexpr size_t MAX_DIMS = 7;
namespace {
template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateIsCloseKernelPtr(const std::string &kernel_name,
                                                                      const uint32_t &device_id) {
  return std::make_unique<cukernel::IsCloseHelperGpuKernel<T>>(kernel_name, device_id);
}
using IsClosePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, IsClosePtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<uint8_t>}};
}  // namespace

bool IsCloseGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
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

bool IsCloseGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto [is_match, index] = MatchKernelAttr(GetKernelAttrFromTensors(inputs, outputs), GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));

  size_t first_rank = inputs[kIndex0]->GetShapeVector().size();
  size_t second_rank = inputs[kIndex1]->GetShapeVector().size();
  if (first_rank > MAX_DIMS || second_rank > MAX_DIMS) {
    MS_EXCEPTION(ValueError) << "IsClose support up to 7d, but got " << first_rank << "d and " << second_rank << "d.";
  }

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  if (inputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs, which is invalid.";
    return false;
  }
  std::vector<size_t> shape =
    std::vector<size_t>(inputs[0]->GetDeviceShapeVector().begin(), inputs[0]->GetDeviceShapeVector().end());

  for (size_t i = 0; i < outputs.size(); i++) {
    std::vector<int64_t> out_shape = outputs[i]->GetDeviceShapeVector();
    output_shapes.emplace_back(out_shape);
  }

  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }
  for (size_t i = 0; i < inputs.size(); i++) input_shapes.emplace_back(inputs[i]->GetDeviceShapeVector());
  InitSizeLists();
  if (!is_input_dynamic_shape_.has_value()) {
    bool is_input_dynamic_shape = false;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto input_shape = inputs[i]->GetDeviceShapeVector();
      if (std::any_of(input_shape.begin(), input_shape.end(), [](int64_t dim) { return dim < 0; })) {
        is_input_dynamic_shape = true;
        break;
      }
    }
    is_input_dynamic_shape_ = is_input_dynamic_shape;
  }

  return true;
}

int IsCloseGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  std::vector<std::vector<int64_t>> input_shapes;
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  attr_ptr_->rtol = inputs[kIndex2]->GetValueWithCheck<float>();
  attr_ptr_->atol = inputs[kIndex3]->GetValueWithCheck<float>();
  attr_ptr_->equal_nan = inputs[kIndex4]->GetValueWithCheck<bool>();
  helper_ptr_->SetKernelParam(attr_ptr_);

  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> inpx_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<int64_t> inpy_shape = inputs.at(kIndex1)->GetShapeVector();
  std::vector<int64_t> out_shape = outputs.at(kIndex0)->GetShapeVector();
  input_shapes.emplace_back(inpx_shape);
  input_shapes.emplace_back(inpy_shape);
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> IsCloseGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, IsClosePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, IsClose, IsCloseGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
