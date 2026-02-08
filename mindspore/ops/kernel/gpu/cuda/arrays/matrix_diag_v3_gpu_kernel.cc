/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/matrix_diag_v3_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/matrix_diag_v3_impl.cuh"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ops/infer/matrix_diag_v3.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kMatrixDiagV3InputsNum = 5;
constexpr int kMatrixDiagV3OutputsNum = 1;
}  // namespace

bool MatrixDiagV3GpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatrixDiagV3InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatrixDiagV3OutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto align = GetValue<std::string>(primitive_->GetAttr("align"));
  left_align_super_diag_ = (align == "LEFT_LEFT" || align == "LEFT_RIGHT");
  left_align_sub_diag_ = (align == "LEFT_LEFT" || align == "RIGHT_LEFT");
  return true;
}

int MatrixDiagV3GpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = inputs.at(kIndex0)->GetShapeVector();
  x_size_ = std::accumulate(x_shape.begin(), x_shape.end(), int64_t(1), std::multiplies{});
  if (x_shape.size() < kDim1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', resize failed, some undefined behaviors happened.";
    return KRET_RESIZE_FAILED;
  }
  max_diag_len_ = x_shape.back();
  auto k_shape = inputs.at(kIndex1)->GetShapeVector();
  k_size_ = std::accumulate(k_shape.begin(), k_shape.end(), int64_t(1), std::multiplies{});
  y_shape_ = outputs.at(kIndex0)->GetShapeVector();
  y_size_ = std::accumulate(y_shape_.begin(), y_shape_.end(), int64_t(1), std::multiplies{});
  if (y_shape_.size() < kDim2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', resize failed, some undefined behaviors happened.";
    return KRET_RESIZE_FAILED;
  }
  num_cols_ = y_shape_.at(y_shape_.size() - kIndex1);
  num_rows_ = y_shape_.at(y_shape_.size() - kIndex2);
  return KRET_OK;
}

template <typename DataType>
bool MatrixDiagV3GpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs) {
  if (x_size_ == 0 || y_size_ == 0) {
    return true;
  }
  auto x_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  auto k_ptr = GetDeviceAddress<kIntType>(inputs, kIndex1);
  auto padding_value_ptr = GetDeviceAddress<DataType>(inputs, kIndex4);
  auto y_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  auto any = [](auto &&... args) -> bool { return ((args == nullptr) || ...); };
  if (any(cuda_stream_, x_ptr, k_ptr, padding_value_ptr, y_ptr)) {
    return false;
  }
  // Get 'k' and store as [lower_diag_index, upper_diag_index].
  kIntType k_stand;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&k_stand, k_ptr, sizeof(kIntType), cudaMemcpyDeviceToHost, cuda_stream_),
    "For '" << kernel_name_ << "', cudaMemcpyAsync input 'k' to host failed.");
  if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
  }
  int64_t upper_diag_index, lower_diag_index = IntToLong(k_stand);
  if (k_size_ == 1) {
    upper_diag_index = lower_diag_index;
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&k_stand, k_ptr + 1, sizeof(kIntType), cudaMemcpyDeviceToHost, cuda_stream_),
      "For '" << kernel_name_ << "', cudaMemcpyAsync input 'k' to host failed.");
    if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
    }
    upper_diag_index = IntToLong(k_stand);
  }

  auto status =
    MatrixDiagV3(x_ptr, padding_value_ptr, y_ptr, y_size_, num_rows_, num_cols_, lower_diag_index, upper_diag_index,
                 max_diag_len_, left_align_super_diag_, left_align_sub_diag_, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, MatrixDiagV3GpuKernelMod::MatrixDiagV3LaunchFunc>>
  MatrixDiagV3GpuKernelMod::func_list_ = {{KernelAttr()
                                             .AddInputAttr(kNumberTypeInt8)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt8)
                                             .AddOutputAttr(kNumberTypeInt8),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<int8_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeInt16)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt16)
                                             .AddOutputAttr(kNumberTypeInt16),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<int16_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddOutputAttr(kNumberTypeInt32),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<int32_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeInt64)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt64)
                                             .AddOutputAttr(kNumberTypeInt64),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<int64_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeUInt8)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeUInt8)
                                             .AddOutputAttr(kNumberTypeUInt8),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<uint8_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeUInt16)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeUInt16)
                                             .AddOutputAttr(kNumberTypeUInt16),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<uint16_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeUInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeUInt32)
                                             .AddOutputAttr(kNumberTypeUInt32),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<uint32_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeUInt64)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeUInt64)
                                             .AddOutputAttr(kNumberTypeUInt64),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<uint64_t>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeFloat16)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeFloat16)
                                             .AddOutputAttr(kNumberTypeFloat16),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<half>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeFloat32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeFloat32)
                                             .AddOutputAttr(kNumberTypeFloat32),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<float>},
                                          {KernelAttr()
                                             .AddInputAttr(kNumberTypeFloat64)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeInt32)
                                             .AddInputAttr(kNumberTypeFloat64)
                                             .AddOutputAttr(kNumberTypeFloat64),
                                           &MatrixDiagV3GpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MatrixDiagV3GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MatrixDiagV3GpuKernelMod::MatrixDiagV3LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixDiagV3, MatrixDiagV3GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
