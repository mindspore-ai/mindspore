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

#include "kernel/gpu/cuda/sparse/sparse_addmm_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kNumTwo = 2;

bool SparseAddmmGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  constexpr int INPUT_NUM = 7;
  constexpr int OUTPUT_NUM = 1;
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
  unit_indices_size_ = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  unit_values_size_ = abstract::TypeIdSize(inputs[kIndex1]->dtype_id());
  return true;
}

int SparseAddmmGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();

  std::vector<int64_t> indices_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                             inputs.at(kIndex0)->GetDeviceShapeVector().end());
  std::vector<int64_t> mat2_shape_ = std::vector<int64_t>(inputs.at(kIndex3)->GetDeviceShapeVector().begin(),
                                                          inputs.at(kIndex3)->GetDeviceShapeVector().end());
  std::vector<int64_t> mat3_shape_ = std::vector<int64_t>(inputs.at(kIndex4)->GetDeviceShapeVector().begin(),
                                                          inputs.at(kIndex4)->GetDeviceShapeVector().end());
  std::vector<int64_t> output_shape_ = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                            outputs.at(kIndex0)->GetDeviceShapeVector().end());

  int64_t out_elements_ =
    std::accumulate(output_shape_.begin(), output_shape_.end(), int64_t(1), std::multiplies<int64_t>());

  input_values_num_ = indices_shape_[0];
  output_values_num_ = out_elements_;
  output_row_ = mat3_shape_[0];
  output_col_ = mat3_shape_[1];
  mat2_row_ = mat2_shape_[0];
  mat2_col_ = mat2_shape_[1];
  mat3_row_ = mat3_shape_[0];
  mat3_col_ = mat3_shape_[1];
  if (input_values_num_ == 0) {
    is_null_input_ = true;
  }

  output_size_list_.emplace_back(out_elements_ * unit_values_size_);  // output_dense
  return KRET_OK;
}

template <typename T, typename S>
bool SparseAddmmGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs) {
  S *input_indices = GetDeviceAddress<S>(inputs, 0);
  T *input_values = GetDeviceAddress<T>(inputs, 1);
  S *input_shape = GetDeviceAddress<S>(inputs, 2);
  T *mat2 = GetDeviceAddress<T>(inputs, 3);
  T *mat3 = GetDeviceAddress<T>(inputs, 4);
  T *alpha = GetDeviceAddress<T>(inputs, 5);
  T *beta = GetDeviceAddress<T>(inputs, 6);
  T *output = GetDeviceAddress<T>(outputs, 0);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  const int64_t kSize = input_values_num_;
  const int64_t kSizeX2 = 2 * kSize;

  std::vector<S> input(kSizeX2);
  S shape[2];

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(input.data(), input_indices, sizeof(S) * input_values_num_ * kNumTwo, cudaMemcpyDeviceToHost,
                    stream),
    "For SparseAddmmGpuKernelMod cudaMemcpyAsync x1_indices Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(&shape, input_shape, sizeof(S) * kNumTwo, cudaMemcpyDeviceToHost, stream),
    "For SparseAddmmGpuKernelMod cudaMemcpyAsync x1_shape Fail");
  if (cudaStreamQuery(stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "For 'SparseAddmm', cuda Stream Sync Failed.");
  }
  mat1_row_ = shape[0];
  mat1_col_ = shape[1];
  if (mat1_row_ < 1 || mat1_col_ < 1) {
    MS_EXCEPTION(RuntimeError) << "For '" << kernel_name_ << "', the value of shape should be positive. "
                               << "But got shape:" << mat1_row_ << mat1_col_;
  }
  for (int64_t i = 1; i < input_values_num_; ++i) {
    S col = input[2 * i + 1];
    S row = input[2 * i];
    if (col < 0 || row < 0) {
      MS_EXCEPTION(RuntimeError) << "For '" << kernel_name_ << "', the value of indice should be positive. "
                                 << "But got indice: " << row << " " << col;
    }
    if (col > mat1_col_ || row > mat1_row_) {
      MS_EXCEPTION(RuntimeError) << "For '" << kernel_name_ << "', the value of indice should be in range of x1 shape. "
                                 << "But got indice: " << row << " " << col << " and shape:" << mat1_row_ << " "
                                 << mat1_col_;
    }
  }
  if (mat3_row_ != mat1_row_ || mat3_col_ != mat2_col_ || mat1_col_ != mat2_row_) {
    MS_EXCEPTION(RuntimeError) << "For '" << kernel_name_
                               << "', the shape of input x1, x2 and x3 must meet the constraints: "
                               << "x1 rows = x3 rows, x1 cols = x2 rows, x2 cols = x3 cols. "
                               << "but got x1 rows (" << mat1_row_ << ") x3 rows (" << mat3_row_ << ") x1 cols ("
                               << mat1_col_ << ") x2 rows (" << mat2_row_ << ") x2 cols (" << mat2_col_ << ") x3 cols ("
                               << mat3_col_ << ")";
  }

  auto status = SparseAddmm(input_indices, input_values, input_values_num_, mat2, mat3, mat2_col_, alpha, beta, output,
                            output_row_, output_col_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseAddmmGpuKernelMod::SparseAddmmFunc>> SparseAddmmGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &SparseAddmmGpuKernelMod::LaunchKernel<int8_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &SparseAddmmGpuKernelMod::LaunchKernel<int16_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &SparseAddmmGpuKernelMod::LaunchKernel<int32_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &SparseAddmmGpuKernelMod::LaunchKernel<int64_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint8_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint16_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint32_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint64_t, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &SparseAddmmGpuKernelMod::LaunchKernel<float, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &SparseAddmmGpuKernelMod::LaunchKernel<double, int>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &SparseAddmmGpuKernelMod::LaunchKernel<int8_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &SparseAddmmGpuKernelMod::LaunchKernel<int16_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &SparseAddmmGpuKernelMod::LaunchKernel<int32_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &SparseAddmmGpuKernelMod::LaunchKernel<int64_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint8_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint16_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint32_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &SparseAddmmGpuKernelMod::LaunchKernel<uint64_t, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &SparseAddmmGpuKernelMod::LaunchKernel<float, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &SparseAddmmGpuKernelMod::LaunchKernel<double, int64_t>}};
std::vector<KernelAttr> SparseAddmmGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseAddmmFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseAddmm, SparseAddmmGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
