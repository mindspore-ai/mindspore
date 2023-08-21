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

#include "plugin/device/gpu/kernel/nn/sparse_apply_momentum_gpu_kernel.h"
#include <algorithm>
#include <iostream>
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/sparse_apply_momentum.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyMomentumInputsNum = 6;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kLrIndex = 2;
constexpr size_t kGradIndex = 3;
constexpr size_t kIndicesIndex = 4;
constexpr size_t kMomentumIndex = 5;
}  // namespace

bool SparseApplyMomentumGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimSparseApplyMomentum->name()) {
    MS_LOG(ERROR) << "For 'SparseApplyMomentum', the kernel name must be 'SparseApplyMomentum', but got "
                  << kernel_name_;
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseApplyMomentum>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "SparseApplyMomentum ops failed!";
    return false;
  }
  use_nesterov_ = kernel_ptr->get_use_nesterov();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_var_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  unit_indices_size_ = abstract::TypeIdSize(inputs[kIndicesIndex]->GetDtype());
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int SparseApplyMomentumGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  if (input_size_list_.size() != kSparseApplyMomentumInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 6 but got " << input_size_list_.size();
    return KRET_RESIZE_FAILED;
  }

  std::vector<int64_t> var_shape = std::vector<int64_t>(inputs.at(kVarIndex)->GetDeviceShapeAdaptively().begin(),
                                                        inputs.at(kVarIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> accum_shape = std::vector<int64_t>(inputs.at(kAccIndex)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kAccIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> lr_shape = std::vector<int64_t>(inputs.at(kLrIndex)->GetDeviceShapeAdaptively().begin(),
                                                       inputs.at(kLrIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> grad_shape = std::vector<int64_t>(inputs.at(kGradIndex)->GetDeviceShapeAdaptively().begin(),
                                                         inputs.at(kGradIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> indices_shape =
    std::vector<int64_t>(inputs.at(kIndicesIndex)->GetDeviceShapeAdaptively().begin(),
                         inputs.at(kIndicesIndex)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> momentum_shape =
    std::vector<int64_t>(inputs.at(kMomentumIndex)->GetDeviceShapeAdaptively().begin(),
                         inputs.at(kMomentumIndex)->GetDeviceShapeAdaptively().end());
  int64_t indices_nums_ =
    std::accumulate(indices_shape.begin(), indices_shape.end(), int64_t(1), std::multiplies<int64_t>());

  global_indices_shape_ = indices_shape[0];
  if (!lr_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', lr is not a scalar.";
    return KRET_RESIZE_FAILED;
  }
  if (!momentum_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', momentum is not a scalar.";
    return KRET_RESIZE_FAILED;
  }
  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'accum' must be the same as the shape of 'var', "
                     "but got the shape of 'accum': "
                  << accum_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'grad' must be the same as the shape of 'var', "
                     "but got the shape of 'grad': "
                  << grad_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (grad_shape[0] != indices_shape[0]) {
    MS_LOG(ERROR)
      << "For '" << kernel_name_
      << "', the first element of shape of 'indices' must be the same as the first element of shape of 'grad', "
         "but got the shape of 'indices': "
      << indices_shape << " and the shape of 'grad': " << grad_shape;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = input_size_list_[0] / unit_var_size_;
  indices_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); i++) {
    indices_size_ *= indices_shape[i];
  }

  workspace_size_list_.emplace_back(indices_nums_ * unit_indices_size_);
  workspace_size_list_.emplace_back(indices_nums_ * sizeof(int32_t));
  workspace_size_list_.emplace_back((indices_nums_ + 1) * sizeof(int32_t));
  workspace_size_list_.emplace_back((indices_nums_ + 1) * sizeof(int32_t));
  workspace_size_list_.emplace_back(sizeof(int32_t));

  return KRET_OK;
}

template <typename T, typename S>
bool SparseApplyMomentumGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  auto var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto accum = reinterpret_cast<T *>(inputs[kAccIndex]->addr);
  auto lr = reinterpret_cast<T *>(inputs[kLrIndex]->addr);
  auto grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  auto indices = reinterpret_cast<S *>(inputs[kIndicesIndex]->addr);
  auto momentum = reinterpret_cast<T *>(inputs[kMomentumIndex]->addr);

  auto indices_sort = reinterpret_cast<S *>(workspace[kIndex0]->addr);
  auto rows_index = reinterpret_cast<int32_t *>(workspace[kIndex1]->addr);
  auto thready_pos = reinterpret_cast<int32_t *>(workspace[kIndex2]->addr);
  auto thready_pos_shrink = reinterpret_cast<int32_t *>(workspace[kIndex3]->addr);
  auto shrink_num = reinterpret_cast<int32_t *>(workspace[kIndex4]->addr);

  auto var_out = reinterpret_cast<T *>(outputs[kVarIndex]->addr);

  std::vector<S> indices_host(global_indices_shape_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_host.data(), indices, sizeof(S) * global_indices_shape_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseApplyMomentum', cudaMemcpy value variable failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseApplyMomentum', cudaStreamSyncFailed");
  }
  for (int i = 0; i < global_indices_shape_; i++) {
    if (indices_host[i] >= global_indices_shape_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' is out of range.";
      return false;
    }
  }

  auto status = CalSparseApplyMomentum(input_elements_, indices_size_, var, accum, lr, grad, indices, momentum,
                                       use_nesterov_, indices_sort, rows_index, thready_pos, thready_pos_shrink,
                                       shrink_num, var_out, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseApplyMomentumGpuKernelMod::SparseApplyMomentumFunc>>
  SparseApplyMomentumGpuKernelMod::func_list_ = {{KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddOutputAttr(kNumberTypeInt8),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int8_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddOutputAttr(kNumberTypeInt16),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int16_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddOutputAttr(kNumberTypeInt32),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int32_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddOutputAttr(kNumberTypeInt64),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int64_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddOutputAttr(kNumberTypeUInt8),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddOutputAttr(kNumberTypeUInt16),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddOutputAttr(kNumberTypeUInt32),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddOutputAttr(kNumberTypeUInt64),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt8)
                                                    .AddOutputAttr(kNumberTypeInt8),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int8_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt16)
                                                    .AddOutputAttr(kNumberTypeInt16),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int16_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddOutputAttr(kNumberTypeInt32),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int32_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddOutputAttr(kNumberTypeInt64),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<int64_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeUInt8)
                                                    .AddOutputAttr(kNumberTypeUInt8),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeUInt16)
                                                    .AddOutputAttr(kNumberTypeUInt16),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeUInt32)
                                                    .AddOutputAttr(kNumberTypeUInt32),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeUInt64)
                                                    .AddOutputAttr(kNumberTypeUInt64),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddOutputAttr(kNumberTypeFloat16),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<half, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddOutputAttr(kNumberTypeFloat32),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<float, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeInt32)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddOutputAttr(kNumberTypeFloat64),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<double, int32_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddOutputAttr(kNumberTypeFloat16),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<half, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddOutputAttr(kNumberTypeFloat32),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<float, int64_t>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddOutputAttr(kNumberTypeFloat64),
                                                  &SparseApplyMomentumGpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> SparseApplyMomentumGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseApplyMomentumFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseApplyMomentum, SparseApplyMomentumGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
