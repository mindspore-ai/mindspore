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
#include "kernel/gpu/cuda/nn/apply_adagrad_d_a_gpu_kernel.h"

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "ops_utils/op_utils.h"

#include "kernel/gpu/cuda_impl/cuda_ops/apply_adagrad_d_a_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyAdagradDAInputsNum = 8;
constexpr size_t kApplyAdagradDAOutputsNum = 1;
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
std::map<size_t, std::string> OutputNames = {{kVarIndex, "var"}};
}  // namespace

bool ApplyAdagradDAGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  batch_rank_ = ops::get_batch_rank(primitive_);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For'" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
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

int ApplyAdagradDAGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  is_null_input_ = false;
  stream_ptr_ = nullptr;
  input_elements_ = 0;
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();

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

  return KRET_OK;
}

std::vector<KernelAttr> ApplyAdagradDAGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAdagradDAFunc> &pair) { return pair.first; });
  return support_list;
}

bool ApplyAdagradDAGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdagradDAOutputsNum, kernel_name_);
  stream_ptr_ = stream_ptr;
  if (is_null_input_) {
    return true;
  }
  return kernel_func_(this, inputs, outputs);
}

template <typename T, typename T1, typename T2, typename T3, typename T4>
bool ApplyAdagradDAGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdagradDAInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdagradDAOutputsNum, kernel_name_);
  auto *var = reinterpret_cast<T *>(inputs[kVarIndex]->device_ptr());
  auto *accum = reinterpret_cast<T *>(inputs[kAccumIndex]->device_ptr());
  auto *squared_accum = reinterpret_cast<T *>(inputs[kSquaredAccumIndex]->device_ptr());
  const auto *grad = reinterpret_cast<T *>(inputs[kGradIndex]->device_ptr());
  const auto *lr = reinterpret_cast<T1 *>(inputs[kLRIndex]->device_ptr());
  const auto *l1 = reinterpret_cast<T2 *>(inputs[kL1Index]->device_ptr());
  const auto *l2 = reinterpret_cast<T3 *>(inputs[kL2Index]->device_ptr());
  const auto *global_step = reinterpret_cast<T4 *>(inputs[kGlobalStepIndex]->device_ptr());

  auto *output_var = reinterpret_cast<T *>(outputs[kVarIndex]->device_ptr());

  auto status = ApplyAdagradDA(batch_size_, input_elements_, var, accum, squared_accum, grad, lr, l1, l2, global_step,
                               output_var, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
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
       .AddOutputAttr(kNumberTypeFloat16),
     &ApplyAdagradDAGpuKernelMod::LaunchKernel<half, half, half, half, int64_t>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdagradDA, ApplyAdagradDAGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
