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

#include "plugin/device/gpu/kernel/math/isclose_gpu_kernel.h"
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
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
   CreateIsCloseKernelPtr<uint8_t>}};
}  // namespace

bool IsCloseGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool IsCloseGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  base_operator_ = base_operator;
  inputs_ = inputs;
  outputs_ = outputs;
  auto [is_match, index] = MatchKernelAttr(GetKernelAttrFromTensors(inputs, outputs), GetOpSupport());
  if (!is_match) {
    return false;
  }

  attr_ptr_->atol = GetValue<float>(base_operator->GetAttr("atol"));
  attr_ptr_->rtol = GetValue<float>(base_operator->GetAttr("rtol"));
  attr_ptr_->equal_nan = GetValue<bool>(base_operator->GetAttr("equal_nan"));
  helper_ptr_ = kernel_attr[index].second(kernel_name_, device_id_);
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);

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
    std::vector<size_t>(inputs[0]->GetDeviceShapeAdaptively().begin(), inputs[0]->GetDeviceShapeAdaptively().end());

  for (size_t i = 0; i < outputs.size(); i++) {
    std::vector<int64_t> out_shape = outputs[i]->GetDeviceShapeAdaptively();
    output_shapes.emplace_back(out_shape);
  }

  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }
  for (size_t i = 0; i < inputs.size(); i++) input_shapes.emplace_back(inputs[i]->GetDeviceShapeAdaptively());
  helper_ptr_->CalMemSize(input_shapes, output_shapes);
  InitSizeLists();
  is_need_retrieve_output_shape_ = true;
  if (!is_input_dynamic_shape_.has_value()) {
    bool is_input_dynamic_shape = false;
    for (size_t i = 0; i < inputs.size(); i++) {
      auto input_shape = inputs[i]->GetDeviceShapeAdaptively();
      if (std::any_of(input_shape.begin(), input_shape.end(), [](int64_t dim) { return dim < 0; })) {
        is_input_dynamic_shape = true;
        break;
      }
    }
    is_input_dynamic_shape_ = is_input_dynamic_shape;
  }

  return true;
}

int IsCloseGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  std::vector<std::vector<int64_t>> input_shapes;
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
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
  input_size_list_ = helper_ptr_->GetInputSizeList();
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
