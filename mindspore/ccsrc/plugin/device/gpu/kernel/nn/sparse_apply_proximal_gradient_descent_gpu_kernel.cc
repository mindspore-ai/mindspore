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

#include "plugin/device/gpu/kernel/nn/sparse_apply_proximal_gradient_descent_gpu_kernel.h"
#include <algorithm>
#include "kernel/common_utils.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_apply_proximal_gradient_descent_impl.cuh"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyProximalGradientDescentInputsNum = 6;
constexpr size_t kSparseApplyProximalGradientDescentOutputsNum = 1;
constexpr size_t kVarIndex = 0;
constexpr size_t kAlphaIndex = 1;
constexpr size_t kL1Index = 2;
constexpr size_t kL2Index = 3;
constexpr size_t kGradIndex = 4;
constexpr size_t kIndicesIndex = 5;
constexpr size_t kOutputIndex = 0;
}  // namespace

bool SparseApplyProximalGradientDescentGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                          const std::vector<KernelTensorPtr> &inputs,
                                                          const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }

  unit_var_size_ = GetTypeByte(TypeIdToType(inputs[kIndex0]->GetDtype()));
  unit_indices_size_ = GetTypeByte(TypeIdToType(inputs[kIndicesIndex]->GetDtype()));

  return true;
}

int SparseApplyProximalGradientDescentGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                           const std::vector<KernelTensorPtr> &inputs,
                                                           const std::vector<KernelTensorPtr> &outputs,
                                                           const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != kSparseApplyProximalGradientDescentInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 6.";
    return KRET_RESIZE_FAILED;
  }

  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> alpha_shape = inputs[kAlphaIndex]->GetShapeVector();
  std::vector<int64_t> l1_shape = inputs[kL1Index]->GetShapeVector();
  std::vector<int64_t> l2_shape = inputs[kL2Index]->GetShapeVector();
  std::vector<int64_t> indices_shape = inputs[kIndicesIndex]->GetShapeVector();

  int64_t indices_nums_ =
    std::accumulate(indices_shape.begin(), indices_shape.end(), int64_t(1), std::multiplies<int64_t>());

  global_indices_shape_ = indices_shape[0];
  input_elements_ = 0;
  batch_size_ = 1;
  if (!alpha_shape.empty()) {
    batch_size_ = std::accumulate(alpha_shape.begin(), alpha_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  if (batch_size_ > 0) {
    input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
    input_elements_ = input_elements_ / batch_size_;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }

  if (!alpha_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'alpha' must be a scalar,and dimension of 'alpha' must be 0,but got the dimension of 'alpha': "
                  << alpha_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!l1_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l1' must be a scalar,and dimension of 'l1' must be 0,but got the dimension of 'l1': "
                  << l1_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!l2_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l2' must be a scalar,and dimension of 'l2' must be 0,but got the dimension of 'l2': "
                  << l2_shape;
    return KRET_RESIZE_FAILED;
  }

  indices_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); i++) {
    indices_size_ *= indices_shape[i];
  }

  workspace_size_list_.emplace_back(
    indices_nums_ * unit_indices_size_);  // cudaMalloc((void **)&indices_sort, sizeof(S) * indices_size);
  workspace_size_list_.emplace_back(
    indices_nums_ * sizeof(int32_t));  // cudaMalloc((void **)&rows_index, sizeof(int32_t) * indices_size);
  workspace_size_list_.emplace_back(
    (indices_nums_ + 1) * sizeof(int32_t));  // cudaMalloc((void **)&thready_pos, sizeof(int32_t) * (indices_size + 1));
  workspace_size_list_.emplace_back(
    (indices_nums_ + 1) *
    sizeof(int32_t));  // cudaMalloc((void **)&thready_pos_shrink, sizeof(int32_t) * (indices_size + 1));
  workspace_size_list_.emplace_back(sizeof(int32_t));  // cudaMalloc((void **)&shrink_num, sizeof(int32_t));

  return KRET_OK;
}

template <typename T, typename S>
bool SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                                  const std::vector<AddressPtr> &workspace,
                                                                  const std::vector<AddressPtr> &outputs) {
  auto var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto alpha = reinterpret_cast<T *>(inputs[kAlphaIndex]->addr);
  auto l1 = reinterpret_cast<T *>(inputs[kL1Index]->addr);
  auto l2 = reinterpret_cast<T *>(inputs[kL2Index]->addr);
  auto grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  auto indices = reinterpret_cast<S *>(inputs[kIndicesIndex]->addr);
  auto var_out = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);

  auto *indices_sort = reinterpret_cast<S *>(workspace[kIndex0]->addr);
  auto *rows_index = reinterpret_cast<int32_t *>(workspace[kIndex1]->addr);
  auto *thready_pos = reinterpret_cast<int32_t *>(workspace[kIndex2]->addr);
  auto *thready_pos_shrink = reinterpret_cast<int32_t *>(workspace[kIndex3]->addr);
  auto *shrink_num = reinterpret_cast<int32_t *>(workspace[kIndex4]->addr);

  std::vector<S> indices_host(global_indices_shape_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(indices_host.data(), indices, sizeof(S) * global_indices_shape_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'SparseApplyProximalGradientDescent', cudaMemcpy value variable failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseApplyProximalGradientDescent', cudaStreamSyncFailed");
  }
  for (int i = 0; i < global_indices_shape_; i++) {
    if (indices_host[i] >= global_indices_shape_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' is out of range.";
      return false;
    }
  }

  std::vector<T> l1host(1);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(l1host.data(), l1, sizeof(T) * 1, cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "For 'SparseApplyProximalGradientDescent', cudaMemcpy value variable failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseApplyProximalGradientDescent', cudaStreamSyncFailed");
  }

  if (l1host[0] > 0) {
    auto status = CalSparseApplyProximalGradientDescent(
      input_elements_, indices_size_, var, alpha, l1, l2, grad, indices, var_out, indices_sort, rows_index, thready_pos,
      thready_pos_shrink, shrink_num, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    auto status = CalSparseApplyProximalGradientDescent_2(
      input_elements_, indices_size_, var, alpha, l1, l2, grad, indices, var_out, indices_sort, rows_index, thready_pos,
      thready_pos_shrink, shrink_num, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, SparseApplyProximalGradientDescentGpuKernelMod::KernelFunc>>
  SparseApplyProximalGradientDescentGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &SparseApplyProximalGradientDescentGpuKernelMod::LaunchKernel<uint64_t, int64_t>}};

std::vector<KernelAttr> SparseApplyProximalGradientDescentGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseApplyProximalGradientDescent,
                      SparseApplyProximalGradientDescentGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
