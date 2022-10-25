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
#include "plugin/device/cpu/kernel/lu_unpack_grad_cpu_kernel.h"
#include <utility>
#include <functional>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kInputNum = 3;
const size_t kOutputNum = 2;
}  // namespace

bool LuUnpackGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuUnpackGradFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LuUnpackGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_L_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_U_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  LU_data_shape_ = inputs[kIndex2]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename T>
bool LuUnpackGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  auto L_grad_output_data = reinterpret_cast<T *>(outputs[0]->addr);
  auto U_grad_output_data = reinterpret_cast<T *>(outputs[1]->addr);
  auto LU_data_dims = LU_data_shape_.size();
  int64_t LU_data_elem_num = std::accumulate(LU_data_shape_.begin(), LU_data_shape_.end(), 1, std::multiplies<int>());
  int64_t LU_data_height = LU_data_shape_[LU_data_dims - 2];
  int64_t LU_data_width = LU_data_shape_[LU_data_dims - 1];
  auto LU_data_stride = LU_data_height * LU_data_width;
  auto matrix_num = LU_data_elem_num / LU_data_stride;
  auto LU_dim_min = std::min(LU_data_height, LU_data_width);
  for (int64_t i = 0; i < LU_data_elem_num; i++) {
    *(L_grad_output_data + i) = static_cast<T>(0);
    *(U_grad_output_data + i) = static_cast<T>(0);
  }

  auto input_L_dims = input_L_shape_.size();
  auto input_U_dims = input_U_shape_.size();
  int64_t matrix_L_width = input_L_shape_[input_L_dims - 2];
  int64_t matrix_L_height = input_L_shape_[input_L_dims - 1];
  auto matrix_L_size = matrix_L_width * matrix_L_height;
  auto matrix_U_width = input_U_shape_[input_U_dims - 2];
  auto matrix_U_height = input_U_shape_[input_U_dims - 1];
  auto matrix_U_size = matrix_U_width * matrix_U_height;
  auto output_stride = LU_data_stride;

  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  for (int64_t a = 0; a < matrix_num; a++) {
    MatrixMap input_L(reinterpret_cast<T *>(inputs[0]->addr) + a * matrix_L_size, matrix_L_width, matrix_L_height);
    MatrixMap input_U(reinterpret_cast<T *>(inputs[1]->addr) + a * matrix_U_size, matrix_U_width, matrix_U_height);
    if (LU_data_width > LU_data_height) {
      MatrixMap output_L(reinterpret_cast<T *>(outputs[0]->addr) + a * output_stride, LU_data_height, LU_data_width);
      T *MiddlePtr = new T[matrix_L_size];
      MatrixMap MiddleData(MiddlePtr, matrix_L_width, matrix_L_height);
      MiddleData = input_L.template triangularView<Eigen::StrictlyLower>();
      for (auto i = 0; i < LU_data_height; i++) {
        for (auto j = 0; j < LU_dim_min; j++) {
          output_L(i, j) = MiddleData(i, j);
        }
      }
      delete[] MiddlePtr;
    } else {
      MatrixMap output_L(reinterpret_cast<T *>(outputs[0]->addr) + a * output_stride, LU_data_height, LU_data_width);
      output_L = input_L.template triangularView<Eigen::StrictlyLower>();
    }

    // triu
    if (LU_data_height > LU_data_width) {
      MatrixMap output_U(reinterpret_cast<T *>(outputs[1]->addr) + a * output_stride, LU_data_height, LU_data_width);
      T *MiddlePtr = new T[matrix_U_size];
      MatrixMap MiddleData(MiddlePtr, matrix_U_width, matrix_U_height);
      MiddleData = input_U.template triangularView<Eigen::Upper>();
      for (auto i = 0; i < LU_dim_min; i++) {
        for (auto j = i; j < LU_data_width; j++) {
          output_U(i, j) = MiddleData(i, j);
        }
      }
      delete[] MiddlePtr;
    } else {
      MatrixMap output_U(reinterpret_cast<T *>(outputs[1]->addr) + a * output_stride, LU_data_height, LU_data_width);
      output_U = input_U.template triangularView<Eigen::Upper>();
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, LuUnpackGradCpuKernelMod::LuUnpackGradFunc>> LuUnpackGradCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackGradCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackGradCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackGradCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackGradCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackGradCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackGradCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackGradCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackGradCpuKernelMod::LaunchKernel<uint8_t>}};

std::vector<KernelAttr> LuUnpackGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuUnpackGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LuUnpackGrad, LuUnpackGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
