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
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>

#include "kernel/common_utils.h"
#include "abstract/utils.h"

#include "plugin/device/gpu/kernel/nn/apply_adagrad_d_a_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_adagrad_d_a_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyAdagradDAInputsNum = 8;
constexpr size_t kApplyAdagradDAOutputsNum = 3;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccumIndex = 1;
constexpr size_t kSquaredAccumIndex = 2;
constexpr size_t kGradIndex = 3;
constexpr size_t kLRIndex = 4;
constexpr size_t kL1Index = 5;
constexpr size_t kL2Index = 6;
constexpr size_t kGlobalStepIndex = 7;
std::map<size_t, std::string> InputNames = {{kVarIndex, "var"},
                                            {kAccumIndex, "gradient_accumulator"},
                                            {kSquaredAccumIndex, "gradient_squared_accumulator"},
                                            {kGradIndex, "grad"},
                                            {kLRIndex, "lr"},
                                            {kL1Index, "l1"},
                                            {kL2Index, "l2"},
                                            {kGlobalStepIndex, "global_step"}};
std::map<size_t, std::string> OutputNames = {
  {kVarIndex, "var"}, {kAccumIndex, "gradient_accumulator"}, {kSquaredAccumIndex, "gradient_squared_accumulator"}};
}  // namespace

bool ApplyAdagradDAGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For'" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  CheckParam(inputs, outputs);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ApplyAdagradDA>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast ApplyAdagradDA ops failed!";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto x_type_id = kernel_attr.GetInputAttr(kVarIndex).dtype;
  unit_size_ = abstract::TypeIdSize(x_type_id);
  return true;
}

int ApplyAdagradDAGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  is_null_input_ = false;
  stream_ptr_ = nullptr;
  input_elements_ = 0;
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', resize failed, ret: " << ret;
    return ret;
  }
  std::vector<int64_t> var_shape = inputs.at(kVarIndex)->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();
  CheckShape(inputs, outputs);

  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }
  batch_size_ = 1;
  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  if (batch_size_ > 0) {
    input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
    input_elements_ = input_elements_ / batch_size_;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  std::vector<size_t> input_shape;
  input_shape = LongVecToSizeVec(var_shape);
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");

  if (batch_rank_ > 1) {
    if (var_shape.size() < lr_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape size of 'var' must be greater than 'lr_shape', but got the shape of 'var': "
                    << Vector2Str(var_shape) << " and 'lr_shape': " << Vector2Str(lr_shape);
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(lr_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the shape of 'lr', "
                       "but got the batch shape of 'var': "
                    << Vector2Str(var_batch_shape) << " and the shape of 'lr': " << Vector2Str(lr_shape);
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

std::vector<KernelAttr> ApplyAdagradDAGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAdagradDAFunc> &pair) { return pair.first; });
  return support_list;
}

void ApplyAdagradDAGpuKernelMod::CheckShape(const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) const {
  std::vector<std::vector<int64_t>> input_shapes(kApplyAdagradDAInputsNum);
  for (size_t i = 0; i < kApplyAdagradDAInputsNum; ++i) {
    input_shapes[i] = inputs[i]->GetShapeVector();
  }

  if (input_shapes[0].empty()) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the shape of var can not be empty.";
  }

  std::vector<std::vector<int64_t>> output_shapes(kApplyAdagradDAOutputsNum);
  for (size_t i = 0; i < kApplyAdagradDAOutputsNum; ++i) {
    output_shapes[i] = outputs[i]->GetShapeVector();
  }

  for (size_t i = 1; i < kLRIndex; ++i) {
    if (input_shapes[i] != input_shapes[0]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of '" << InputNames[i]
                        << "' and 'var' should be the same, but got shape of '" << InputNames[i]
                        << "':" << input_shapes[i] << " and shape of 'var': " << input_shapes[0];
    }
  }

  for (size_t i = 0; i < kApplyAdagradDAOutputsNum; ++i) {
    if (output_shapes[i] != input_shapes[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of output '" << OutputNames[i] << "' and input"
                        << InputNames[i] << " should be the same, but got shape of '" << OutputNames[i]
                        << "':" << output_shapes[i] << " and shape of input: " << InputNames[i] << ": "
                        << input_shapes[i];
    }
  }
}

void ApplyAdagradDAGpuKernelMod::CheckDType(const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) const {
  std::vector<TypeId> input_types(kApplyAdagradDAInputsNum);
  for (size_t i = 0; i < kApplyAdagradDAInputsNum; ++i) {
    input_types[i] = inputs[i]->GetDtype();
  }

  if (input_types[0] != kNumberTypeFloat16 && input_types[0] != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'var' should be float16 or float32, but got "
                      << input_types[0] << " .";
  }

  std::vector<TypeId> output_types(kApplyAdagradDAOutputsNum);
  for (size_t i = 0; i < kApplyAdagradDAOutputsNum; ++i) {
    output_types[i] = outputs[i]->GetDtype();
  }

  for (size_t i = 1; i < kLRIndex; ++i) {
    if (input_types[i] != input_types[0]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the type of '" << InputNames[i]
                        << "' and 'var' should be the same, but got type of '" << InputNames[i]
                        << "':" << input_types[i] << " and type of 'var': " << input_types[0];
    }
  }

  for (size_t i = kLRIndex; i < kL2Index; ++i) {
    if (input_types[i] != kNumberTypeFloat16 && input_types[i] != kNumberTypeFloat32) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the type of '" << InputNames[i]
                        << "' should be float16 or float32, but got " << input_types[0] << " .";
    }
  }

  if (input_types[kGlobalStepIndex] != kNumberTypeInt32 && input_types[kGlobalStepIndex] != kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'gobal_step' should be int32 or int64, but got "
                      << input_types[kGlobalStepIndex] << " .";
  }

  for (size_t i = 0; i < kApplyAdagradDAOutputsNum; ++i) {
    if (output_types[i] != input_types[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the type of output " << OutputNames[i] << " and input"
                        << InputNames[i] << " should be the same, but got type of " << OutputNames[i] << ":"
                        << output_types[i] << " and type of input: " << InputNames[i] << ": " << input_types[i];
    }
  }
}

void ApplyAdagradDAGpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdagradDAInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdagradDAOutputsNum, kernel_name_);
  CheckShape(inputs, outputs);
  CheckDType(inputs, outputs);
}

bool ApplyAdagradDAGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  stream_ptr_ = stream_ptr;
  if (is_null_input_) {
    return true;
  }
  return kernel_func_(this, inputs, outputs);
}

template <typename T, typename T1, typename T2, typename T3, typename T4>
bool ApplyAdagradDAGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto *var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto *accum = reinterpret_cast<T *>(inputs[kAccumIndex]->addr);
  auto *squared_accum = reinterpret_cast<T *>(inputs[kSquaredAccumIndex]->addr);
  const auto *grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  const auto *lr = reinterpret_cast<T1 *>(inputs[kLRIndex]->addr);
  const auto *l1 = reinterpret_cast<T2 *>(inputs[kL1Index]->addr);
  const auto *l2 = reinterpret_cast<T3 *>(inputs[kL2Index]->addr);
  const auto *global_step = reinterpret_cast<T4 *>(inputs[kGlobalStepIndex]->addr);

  auto *output_var = reinterpret_cast<T *>(outputs[kVarIndex]->addr);
  auto *output_accum = reinterpret_cast<T *>(outputs[kAccumIndex]->addr);
  auto *output_squared_accum = reinterpret_cast<T *>(outputs[kSquaredAccumIndex]->addr);

  auto status =
    ApplyAdagradDA(batch_size_, input_elements_, var, accum, squared_accum, grad, lr, l1, l2, global_step, output_var,
                   output_accum, output_squared_accum, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ApplyAdagradDAGpuKernelMod::ApplyAdagradDAFunc>>
  ApplyAdagradDAGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, float, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, float, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, half, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, half, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, float, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, float, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, half, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, half, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, float, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, float, half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, half, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, float, half, half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, float, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, float, half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, half, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<float, half, half, half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, float, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, float, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, half, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, half, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, float, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, float, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, half, float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, half, half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, float, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, float, half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, half, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, float, half, half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, float, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, float, half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, half, float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, half, half, int64_t>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdagradDA, ApplyAdagradDAGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
