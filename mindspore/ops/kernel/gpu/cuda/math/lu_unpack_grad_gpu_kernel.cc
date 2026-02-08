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
#include "kernel/gpu/cuda/math/lu_unpack_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr const size_t kLuUnpackGradInputNum = 3;
constexpr const size_t kLuUnpackGradOutputNum = 2;
constexpr const int64_t Min_Dim = 2;
}  // namespace

bool LuUnpackGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kLuUnpackGradInputNum || outputs.size() != kLuUnpackGradOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kLuUnpackGradInputNum << " and "
                  << kLuUnpackGradOutputNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[pair.second].second;
  unit_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->dtype_id());
  input_L_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeVector().end());
  input_U_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeVector().begin(),
                                       inputs.at(kIndex1)->GetDeviceShapeVector().end());
  input_LU_shape = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeVector().begin(),
                                        inputs.at(kIndex2)->GetDeviceShapeVector().end());
  return true;
}

int LuUnpackGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  ResetResource();
  input_L_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeVector().end());
  input_U_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeVector().begin(),
                                       inputs.at(kIndex1)->GetDeviceShapeVector().end());
  input_LU_shape = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeVector().begin(),
                                        inputs.at(kIndex2)->GetDeviceShapeVector().end());
  int64_t input_L_elements_ =
    std::accumulate(input_L_shape.begin(), input_L_shape.end(), int64_t(1), std::multiplies<int64_t>());
  int64_t input_U_elements_ =
    std::accumulate(input_U_shape.begin(), input_U_shape.end(), int64_t(1), std::multiplies<int64_t>());
  int64_t input_LU_elements_ =
    std::accumulate(input_LU_shape.begin(), input_LU_shape.end(), int64_t(1), std::multiplies<int64_t>());
  input_elements_ = input_L_elements_ + input_U_elements_ + input_LU_elements_;
  int64_t input_L_size = input_L_elements_ * unit_size_;
  int64_t input_U_size = input_U_elements_ * unit_size_;
  int64_t input_LU_size = input_LU_elements_ * unit_size_;
  output_size_list_.emplace_back(input_LU_size);
  output_size_list_.emplace_back(input_LU_size);
  workspace_size_list_.emplace_back(input_L_size);
  workspace_size_list_.emplace_back(input_U_size);

  return KRET_OK;
}

void LuUnpackGradGpuKernelMod::ResetResource() noexcept {
  input_elements_ = 0;
  is_null_input_ = false;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool LuUnpackGradGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<kernel::KernelTensor *> &outputs) {
  T *l_grad_input = GetDeviceAddress<T>(inputs, kIndex0);
  T *u_grad_input = GetDeviceAddress<T>(inputs, kIndex1);
  T *l_grad_output = GetDeviceAddress<T>(outputs, kIndex0);
  T *u_grad_output = GetDeviceAddress<T>(outputs, kIndex1);

  int64_t lu_data_dims = input_LU_shape.size();
  int64_t lu_data_elem_num =
    std::accumulate(input_LU_shape.begin(), input_LU_shape.end(), int64_t(1), std::multiplies<int64_t>());
  int64_t lu_data_height = input_LU_shape[lu_data_dims - 2];
  int64_t lu_data_width = input_LU_shape[lu_data_dims - 1];
  int64_t lu_data_stride = lu_data_height * lu_data_width;
  int64_t matrix_num = lu_data_elem_num / lu_data_stride;
  int64_t output_stride = lu_data_stride;

  int64_t input_L_dims = input_L_shape.size();
  int64_t input_U_dims = input_U_shape.size();
  int64_t matrix_L_width = input_L_shape[input_L_dims - 2];
  int64_t matrix_L_height = input_L_shape[input_L_dims - 1];
  int64_t matrix_U_width = input_U_shape[input_U_dims - 2];
  int64_t matrix_U_height = input_U_shape[input_U_dims - 1];
  cudaError_t status = cudaErrorNotReady;
  if (lu_data_width > lu_data_height) {
    status = CalTrilExpendWidth(matrix_num * output_stride, l_grad_input, matrix_L_height, matrix_L_width,
                                l_grad_output, lu_data_height, lu_data_width, device_id_, stream_ptr_);
    CHECK_CUDA_STATUS(status, kernel_name_);
    status = CalTriuUpper(matrix_num * output_stride, u_grad_input, matrix_U_height, matrix_U_width, u_grad_output,
                          lu_data_height, lu_data_width, device_id_, stream_ptr_);
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    if (lu_data_height > lu_data_width) {
      status = CalTrilLower(matrix_num * output_stride, l_grad_input, matrix_L_height, matrix_L_width, l_grad_output,
                            lu_data_height, lu_data_width, device_id_, stream_ptr_);
      CHECK_CUDA_STATUS(status, kernel_name_);
      status = CalTriuExpendHeight(matrix_num * output_stride, u_grad_input, matrix_U_height, matrix_U_width,
                                   u_grad_output, lu_data_height, lu_data_width, device_id_, stream_ptr_);
      CHECK_CUDA_STATUS(status, kernel_name_);
    } else {
      status = CalTrilLower(matrix_num * output_stride, l_grad_input, matrix_L_height, matrix_L_width, l_grad_output,
                            lu_data_height, lu_data_width, device_id_, stream_ptr_);
      CHECK_CUDA_STATUS(status, kernel_name_);
      status = CalTriuUpper(matrix_num * output_stride, u_grad_input, matrix_U_height, matrix_U_width, u_grad_output,
                            lu_data_height, lu_data_width, device_id_, stream_ptr_);
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, LuUnpackGradGpuKernelMod::LuUnpackGradFunc>> LuUnpackGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackGradGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackGradGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackGradGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGradGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackGradGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackGradGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackGradGpuKernelMod::LaunchKernel<uint8_t>}};

std::vector<KernelAttr> LuUnpackGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuUnpackGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LuUnpackGrad, LuUnpackGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
