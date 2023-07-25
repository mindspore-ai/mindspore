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

#include "plugin/device/gpu/kernel/nn/sparse_apply_adagrad_d_a_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_apply_adagrad_d_a_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyAdagradDAInputsNum = 9;
constexpr size_t kSparseApplyAdagradDAOutputsNum = 1;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccumIndex = 1;
constexpr size_t kSquaredAccumIndex = 2;
constexpr size_t kGradIndex = 3;
constexpr size_t kIndicesIndex = 4;
constexpr size_t kLRIndex = 5;
constexpr size_t kL1Index = 6;
constexpr size_t kL2Index = 7;
constexpr size_t kGlobalStepIndex = 8;
std::map<size_t, std::string> InputNames = {{kVarIndex, "var"},
                                            {kAccumIndex, "gradient_accum"},
                                            {kSquaredAccumIndex, "gradient_square_accum"},
                                            {kGradIndex, "grad"},
                                            {kIndicesIndex, "indices"},
                                            {kLRIndex, "lr"},
                                            {kL1Index, "l1"},
                                            {kL2Index, "l2"},
                                            {kGlobalStepIndex, "global_step"}};
std::map<size_t, std::string> OutputNames = {{kVarIndex, "var"}};
}  // namespace

bool SparseApplyAdagradDAGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For'" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  CheckParam(inputs, outputs);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseApplyAdagradDA>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast SparseApplyAdagradDA ops failed!";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_var_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  unit_indices_size_ = abstract::TypeIdSize(inputs[kIndicesIndex]->GetDtype());
  return true;
}

int SparseApplyAdagradDAGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  is_null_input_ = false;
  stream_ptr_ = nullptr;
  input_elements_ = 0;

  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> var_shape = std::vector<int64_t>(inputs.at(kVarIndex)->GetDeviceShapeAdaptively().begin(),
                                                        inputs.at(kVarIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> grad_shape = std::vector<int64_t>(inputs.at(kGradIndex)->GetDeviceShapeAdaptively().begin(),
                                                         inputs.at(kGradIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> indices = std::vector<int64_t>(inputs.at(kIndicesIndex)->GetDeviceShapeAdaptively().begin(),
                                                      inputs.at(kIndicesIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> lr_shape = std::vector<int64_t>(inputs.at(kLRIndex)->GetDeviceShapeAdaptively().begin(),
                                                       inputs.at(kLRIndex)->GetDeviceShapeAdaptively().end());
  int64_t indices_nums_ = std::accumulate(indices.begin(), indices.end(), int64_t(1), std::multiplies<int64_t>());

  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', but got the shape of 'lr': " << lr_shape
                  << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }
  batch_size_ = 1;
  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  if (batch_size_ > 0) {
    input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
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
                    << var_shape << " and 'lr_shape': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(lr_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the shape of 'lr', "
                       "but got the batch shape of 'var': "
                    << var_batch_shape << " and the shape of 'lr': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
  }

  // indices check
  if (indices.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' must be a 1-D vector, but got " << indices.size()
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }

  auto indices_size = indices[0];
  if (grad_shape[0] != SizeToLong(indices_size)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the first dimension value of 'grad' must be equal to "
                     "the first dimension value of 'indices', but got the first dimension value of 'grad': "
                  << grad_shape[0] << ", and the first dimension value of 'indices': " << indices_size;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = input_size_list_[0] / unit_var_size_;
  indices_size_ = 1;
  for (size_t i = 0; i < indices.size(); i++) {
    indices_size_ *= indices[i];
  }

  workspace_size_list_.emplace_back(indices_nums_ * unit_indices_size_);
  workspace_size_list_.emplace_back(indices_nums_ * sizeof(int32_t));
  workspace_size_list_.emplace_back((indices_nums_ + 1) * sizeof(int32_t));
  workspace_size_list_.emplace_back((indices_nums_ + 1) * sizeof(int32_t));
  workspace_size_list_.emplace_back(sizeof(int32_t));

  return KRET_OK;
}

void SparseApplyAdagradDAGpuKernelMod::CheckShape(const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs) const {
  std::vector<std::vector<int64_t>> input_shapes(kSparseApplyAdagradDAInputsNum);
  for (size_t i = 0; i < kSparseApplyAdagradDAInputsNum; ++i) {
    input_shapes[i] = inputs[i]->GetShapeVector();
  }

  if (input_shapes[0].empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of var can not be empty.";
  }

  std::vector<std::vector<int64_t>> output_shapes(kSparseApplyAdagradDAOutputsNum);
  for (size_t i = 0; i < kSparseApplyAdagradDAOutputsNum; ++i) {
    output_shapes[i] = outputs[i]->GetShapeVector();
  }

  for (size_t i = 1; i < kIndicesIndex; ++i) {
    if (input_shapes[i] != input_shapes[0]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of '" << InputNames[i]
                        << "' and 'var' should be the same, but got shape of '" << InputNames[i]
                        << "':" << input_shapes[i] << " and shape of 'var': " << input_shapes[0];
    }
  }

  for (size_t i = kLRIndex; i < kSparseApplyAdagradDAInputsNum; ++i) {
    if (input_shapes[i].size() != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of '" << InputNames[i]
                        << "' must be 0, but got value: " << input_shapes[i];
    }
  }
  if (input_shapes[kIndicesIndex][0] != input_shapes[kGradIndex][0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of '" << InputNames[kIndicesIndex][0]
                      << "' should be the same as shape of first dimension of '" << InputNames[kGradIndex][0]
                      << "but got value: " << input_shapes[kIndicesIndex];
  }

  for (size_t i = 0; i < kSparseApplyAdagradDAOutputsNum; ++i) {
    if (output_shapes[i] != input_shapes[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of output '" << OutputNames[i] << "' and input"
                        << InputNames[i] << " should be the same, but got shape of '" << OutputNames[i]
                        << "':" << output_shapes[i] << " and shape of input: " << InputNames[i] << ": "
                        << input_shapes[i];
    }
  }
}

void SparseApplyAdagradDAGpuKernelMod::CheckDType(const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs) const {
  std::vector<TypeId> input_types(kSparseApplyAdagradDAInputsNum);
  for (size_t i = 0; i < kSparseApplyAdagradDAInputsNum; ++i) {
    input_types[i] = inputs[i]->GetDtype();
  }

  std::vector<TypeId> output_types(kSparseApplyAdagradDAOutputsNum);
  for (size_t i = 0; i < kSparseApplyAdagradDAOutputsNum; ++i) {
    output_types[i] = outputs[i]->GetDtype();
  }

  for (size_t i = 1; i < kGlobalStepIndex; ++i) {
    if (i == kIndicesIndex) continue;
    if (input_types[i] != input_types[0]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the type of '" << InputNames[i]
                        << "' and 'var' should be the same, but got type of '" << InputNames[i]
                        << "':" << input_types[i] << " and type of 'var': " << input_types[0];
    }
  }

  if (input_types[kGlobalStepIndex] != kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'global_step' should be int64, but got "
                      << input_types[kGlobalStepIndex] << " .";
  }
  if (input_types[kIndicesIndex] != kNumberTypeInt32 && input_types[kIndicesIndex] != kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'indices' should be int32 or int64, but got "
                      << input_types[kIndicesIndex] << " .";
  }

  for (size_t i = 0; i < kSparseApplyAdagradDAOutputsNum; ++i) {
    if (output_types[i] != input_types[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the type of output " << OutputNames[i] << " and input"
                        << InputNames[i] << " should be the same, but got type of " << OutputNames[i] << ":"
                        << output_types[i] << " and type of input: " << InputNames[i] << ": " << input_types[i];
    }
  }
}

void SparseApplyAdagradDAGpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyAdagradDAInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyAdagradDAOutputsNum, kernel_name_);
  CheckDType(inputs, outputs);
}

bool SparseApplyAdagradDAGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  stream_ptr_ = stream_ptr;
  if (is_null_input_) {
    return true;
  }
  return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
}

template <typename T, typename S, typename S1>
bool SparseApplyAdagradDAGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &workspace,
                                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto *var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto *accum = reinterpret_cast<T *>(inputs[kAccumIndex]->addr);
  auto *squared_accum = reinterpret_cast<T *>(inputs[kSquaredAccumIndex]->addr);
  const auto *grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  const auto *indices = reinterpret_cast<S *>(inputs[kIndicesIndex]->addr);
  const auto *lr = reinterpret_cast<T *>(inputs[kLRIndex]->addr);
  const auto *l1 = reinterpret_cast<T *>(inputs[kL1Index]->addr);
  const auto *l2 = reinterpret_cast<T *>(inputs[kL2Index]->addr);
  const auto *global_step = reinterpret_cast<S1 *>(inputs[kGlobalStepIndex]->addr);

  auto *output_var = reinterpret_cast<T *>(outputs[kVarIndex]->addr);

  auto *indices_sort = reinterpret_cast<S *>(workspace[kIndex0]->addr);
  auto *rows_index = reinterpret_cast<int32_t *>(workspace[kIndex1]->addr);
  auto *thready_pos = reinterpret_cast<int32_t *>(workspace[kIndex2]->addr);
  auto *thready_pos_shrink = reinterpret_cast<int32_t *>(workspace[kIndex3]->addr);
  auto *shrink_num = reinterpret_cast<int32_t *>(workspace[kIndex4]->addr);

  auto status =
    CalSparseApplyAdagradDA(batch_size_, indices_size_, input_elements_, var, accum, squared_accum, grad, indices, lr,
                            l1, l2, global_step, output_var, indices_sort, rows_index, thready_pos, thready_pos_shrink,
                            shrink_num, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseApplyAdagradDAGpuKernelMod::SparseApplyAdagradDAFunc>>
  SparseApplyAdagradDAGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int8_t, int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int16_t, int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int32_t, int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int64_t, int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int8_t, int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int16_t, int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int32_t, int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int64_t, int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int8_t, int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int16_t, int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int32_t, int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int64_t, int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int8_t, int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int16_t, int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int32_t, int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<int64_t, int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<double, int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<float, int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<half, int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<double, int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<float, int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<half, int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<double, int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<float, int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<half, int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<double, int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<float, int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseApplyAdagradDAGpuKernelMod::LaunchKernel<half, int32_t, int32_t>},
};

std::vector<KernelAttr> SparseApplyAdagradDAGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseApplyAdagradDAFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseApplyAdagradDA, SparseApplyAdagradDAGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
