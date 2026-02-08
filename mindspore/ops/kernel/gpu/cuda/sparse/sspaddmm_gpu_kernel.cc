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

#include "kernel/gpu/cuda/sparse/sspaddmm_gpu_kernel.h"
#include <unordered_map>

namespace mindspore {
namespace kernel {
constexpr int64_t kNumTwo = 2;
constexpr int INPUT_NUM = 9;
constexpr int OUTPUT_NUM = 3;

bool SspaddmmGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != INPUT_NUM || outputs.size() != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output must be " << INPUT_NUM << " and " << OUTPUT_NUM
                  << ", but got " << inputs.size() << " and " << outputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_indices_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  unit_values_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  return true;
}

int SspaddmmGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> x1_indices_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                               inputs.at(kIndex0)->GetDeviceShapeVector().end());
  std::vector<int64_t> x2_indices_shape = std::vector<int64_t>(inputs.at(kIndex3)->GetDeviceShapeVector().begin(),
                                                               inputs.at(kIndex3)->GetDeviceShapeVector().end());
  std::vector<int64_t> x3_dense_shape = std::vector<int64_t>(inputs.at(kIndex6)->GetDeviceShapeVector().begin(),
                                                             inputs.at(kIndex6)->GetDeviceShapeVector().end());
  std::vector<int64_t> y_indices_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                              outputs.at(kIndex0)->GetDeviceShapeVector().end());
  x1_values_num_ = x1_indices_shape[1];
  x2_values_num_ = x2_indices_shape[1];
  y_values_num_ = y_indices_shape[1];
  x3_dense_col_ = x3_dense_shape[1];
  if (y_values_num_ == 0) {
    is_null_input_ = true;
  }

  workspace_size_list_.emplace_back(x2_values_num_ * sizeof(int64_t));        // index
  output_size_list_.emplace_back(y_values_num_ * sizeof(int64_t) * kNumTwo);  // y_indices
  output_size_list_.emplace_back(y_values_num_ * unit_values_size_);          // y_values
  output_size_list_.emplace_back(kNumTwo * sizeof(int64_t));                  // y_shape

  return KRET_OK;
}

template <typename T, typename S>
bool SspaddmmGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs) {
  S *x1_indices = GetDeviceAddress<S>(inputs, 0);
  T *x1_values = GetDeviceAddress<T>(inputs, 1);
  S *x1_shape = GetDeviceAddress<S>(inputs, 2);
  S *x2_indices = GetDeviceAddress<S>(inputs, 3);
  T *x2_values = GetDeviceAddress<T>(inputs, 4);
  T *x3_dense = GetDeviceAddress<T>(inputs, 6);
  T *alpha = GetDeviceAddress<T>(inputs, 7);
  T *beta = GetDeviceAddress<T>(inputs, 8);

  int64_t *index = GetDeviceAddress<int64_t>(workspace, 0);

  int64_t *y_indices = GetDeviceAddress<int64_t>(outputs, 0);
  T *y_values = GetDeviceAddress<T>(outputs, 1);
  int64_t *y_shape = GetDeviceAddress<int64_t>(outputs, 2);

  const int64_t kSize = x2_values_num_;
  const int64_t kSizeX2 = kNumTwo * kSize;
  std::vector<S> x2(kSizeX2);
  S x1_devicetohost_shape[kNumTwo];
  int64_t x1_host_shape[kNumTwo];
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(y_values, 0, y_values_num_ * unit_values_size_, stream),
                                    "For SspaddmmGpuKernelMod, failed to cudaMemset for y_values.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(index, 0, x2_values_num_ * unit_values_size_, stream),
                                    "For SspaddmmGpuKernelMod, failed to cudaMemset for index.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(&x1_devicetohost_shape, x1_shape, sizeof(S) * kNumTwo, cudaMemcpyDeviceToHost, stream),
    "For SspaddmmGpuKernelMod cudaMemcpyAsync x1_shape Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(x2.data(), x2_indices, sizeof(S) * x2_values_num_ * kNumTwo, cudaMemcpyDeviceToHost, stream),
    "For SspaddmmGpuKernelMod cudaMemcpyAsync x2_values Fail");
  if (cudaStreamQuery(stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream),
                                       "For 'SspaddmmGpuKernelMod', cuda Stream Sync Failed.");
  }

  // cal y_shape
  x1_host_shape[0] = static_cast<int64_t>(x1_devicetohost_shape[0]);
  x1_host_shape[1] = static_cast<int64_t>(x1_devicetohost_shape[1]);
  // cal index for y_values and y_indices
  std::vector<int64_t> idx(kSize);
  int64_t count = 0;
  idx[0] = count;
  for (int64_t i = 1; i < x2_values_num_; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (x2[i] == x2[j]) {
        idx[i] = idx[j];
        break;
      } else if (i == j + 1) {
        idx[i] = ++count;
        break;
      }
    }
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(index, idx.data(), sizeof(int64_t) * x2_values_num_, cudaMemcpyHostToDevice, stream),
    "For SspaddmmGpuKernelMod cudaMemcpyAsync index failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(y_shape, x1_host_shape, kNumTwo * sizeof(int64_t), cudaMemcpyHostToDevice, stream),
    "For SspaddmmGpuKernelMod cudaMemcpyAsync x1_shape failed.");

  // x1 + x2 @ x3_dense
  auto status = CalSparseAddSparse(x1_indices, x1_values, x1_values_num_, y_indices, y_values, y_values_num_, beta,
                                   device_id_, stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  // the result of x2 @ x3_dense will write to output directly
  status = CalSparseMulDense(x2_indices, x2_values, x2_values_num_, x3_dense, y_indices, y_values, y_values_num_,
                             x3_dense_col_, x1_values_num_, alpha, index, device_id_, stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define GPU_SSPADDMM_KERNEL_REGISTER_ONE(ms_value_type, ms_index_type) \
  {                                                                    \
    KernelAttr()                                                       \
      .AddInputAttr(ms_index_type)                                     \
      .AddInputAttr(ms_value_type)                                     \
      .AddInputAttr(ms_index_type)                                     \
      .AddInputAttr(ms_index_type)                                     \
      .AddInputAttr(ms_value_type)                                     \
      .AddInputAttr(ms_index_type)

#define GPU_SSPADDMM_KERNEL_REGISTER_TWO(ms_value_type, value_type, index_type) \
  .AddInputAttr(ms_value_type)                                                  \
    .AddInputAttr(ms_value_type)                                                \
    .AddInputAttr(ms_value_type)                                                \
    .AddOutputAttr(kNumberTypeInt64)                                            \
    .AddOutputAttr(ms_value_type)                                               \
    .AddOutputAttr(kNumberTypeInt64),                                           \
    &SspaddmmGpuKernelMod::LaunchKernel<value_type, index_type>                 \
  }

std::vector<std::pair<KernelAttr, SspaddmmGpuKernelMod::SspaddmmFunc>> SspaddmmGpuKernelMod::func_list_ = {
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt8, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt8, int8_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt16, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt16, int16_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt32, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt32, int32_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt64, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt64, int64_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt8, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt8, uint8_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt16, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt16, uint16_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt32, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt32, uint32_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt64, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt64, uint64_t, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeFloat16, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeFloat16, half, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeFloat32, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeFloat32, float, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeFloat64, kNumberTypeInt32)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeFloat64, double, int),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt8, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt8, int8_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt16, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt16, int16_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt32, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt32, int32_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeInt64, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeInt64, int64_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt8, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt8, uint8_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeFloat32, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeFloat32, float, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeFloat64, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeFloat64, double, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt16, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt16, uint16_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt32, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt32, uint32_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeUInt64, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeUInt64, uint64_t, int64_t),
  GPU_SSPADDMM_KERNEL_REGISTER_ONE(kNumberTypeFloat16, kNumberTypeInt64)
    GPU_SSPADDMM_KERNEL_REGISTER_TWO(kNumberTypeFloat16, half, int64_t)};

std::vector<KernelAttr> SspaddmmGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SspaddmmFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Sspaddmm, SspaddmmGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
