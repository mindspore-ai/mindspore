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
#include "plugin/device/cpu/kernel/lu_unpack_cpu_kernel.h"
#include <utility>
#include <functional>
#include <algorithm>
#include <numeric>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDimNum = 2;
constexpr size_t kFirstDim = 1;
constexpr size_t kSecondDim = 2;
constexpr uint32_t kOutputNum = 3;
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kFirstInputIndex = 0;
constexpr uint32_t kFirstOutputIndex = 0;
constexpr uint32_t kSecondOutputIndex = 1;
constexpr uint32_t kThirdOutputIndex = 2;
}  // namespace

void LuUnpackCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  node_wpt_ = kernel_node;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputNum, kernel_name_);
  auto LU_data_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto LU_pivots_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (LU_data_shape.size() < kDimNum) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', LU_data's dimensions must be greater than or equal to 2.";
  }
  if (LU_pivots_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', LU_pivots's dimensions must be greater than or equal to 1.";
  }
  if (LU_pivots_shape[LU_pivots_shape.size() - 1] !=
      std::min(LU_data_shape[LU_data_shape.size() - kFirstDim], LU_data_shape[LU_data_shape.size() - kSecondDim])) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "',"
                             << " the last dimension of LU_pivots must be the same as the minimum value of the last"
                             << " two dimensions of the LU_data.";
  }
  for (size_t i = 0; i < LU_pivots_shape.size() - 1; i++) {
    if (LU_data_shape[i] != LU_pivots_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "',"
                               << " batch dimension of LU_pivots should match batch dimension of LU_data.";
    }
  }
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuUnpackFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "LuUnpack does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}
template <typename T_data, typename T_pivots>
void LuUnpackCpuKernelMod::LuUnpack(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &outputs, int64_t Lu_data_dim1,
                                    int64_t Lu_pivots_dim, T_pivots *const Lu_pivots_working_ptr, int64_t matrix_index,
                                    int64_t matrix_size, int64_t matrix_width, int64_t matrix_height,
                                    int64_t pivots_stride, int64_t L_stride, int64_t U_stride, T_data *const P_eye) {
  using MatrixMap = Eigen::Map<Eigen::Matrix<T_data, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input(reinterpret_cast<T_data *>(inputs[kFirstInputIndex]->addr) + matrix_index * matrix_size, matrix_width,
                  matrix_height);
  //  Triu
  if (matrix_width > matrix_height) {
    if (matrix_size <= 0) {
      MS_EXCEPTION(ValueError) << "Memory alloc size matrix_size should be greater than 0.";
    }
    T_data *MiddlePtr = new T_data[matrix_size];
    MatrixMap MiddleData(MiddlePtr, matrix_width, matrix_height);
    MiddleData = input.template triangularView<Eigen::Upper>();
    MatrixMap(reinterpret_cast<T_data *>(outputs[kThirdOutputIndex]->addr) + matrix_index * U_stride, matrix_height,
              matrix_height) = MiddleData.block(0, 0, matrix_height, matrix_height);
    delete[] MiddlePtr;
  } else {
    MatrixMap(reinterpret_cast<T_data *>(outputs[kThirdOutputIndex]->addr) + matrix_index * U_stride, matrix_width,
              matrix_height) = input.template triangularView<Eigen::Upper>();
  }
  //  Tril
  if (matrix_height > matrix_width) {
    T_data *MiddlePtr = new T_data[matrix_size];
    MatrixMap MiddleData(MiddlePtr, matrix_width, matrix_height);
    MiddleData = input.template triangularView<Eigen::UnitLower>();
    MatrixMap(reinterpret_cast<T_data *>(outputs[kSecondOutputIndex]->addr) + matrix_index * L_stride, matrix_width,
              matrix_width) = MiddleData.block(0, 0, matrix_width, matrix_width);
    delete[] MiddlePtr;
  } else {
    MatrixMap(reinterpret_cast<T_data *>(outputs[kSecondOutputIndex]->addr) + matrix_index * L_stride, matrix_width,
              matrix_height) = input.template triangularView<Eigen::UnitLower>();
  }
  //  Swap
  std::vector<T_pivots> final_order;
  size_t Lu_data_dim1_unsigned = static_cast<size_t>(Lu_data_dim1);
  final_order.resize(Lu_data_dim1_unsigned);
  for (size_t i = 0; i < Lu_data_dim1_unsigned; i++) {
    final_order[i] = T_pivots(i);
  }
  for (T_pivots id = 0; id < Lu_pivots_dim; id++) {
    size_t perm_id = 0;
    size_t perm_pivots_id = 0;
    for (size_t i = 0; i < Lu_data_dim1_unsigned; i++) {
      if (id == final_order[i]) {
        perm_id = i;
      }
      if (!((*(Lu_pivots_working_ptr + id) <= Lu_data_dim1) && (*(Lu_pivots_working_ptr + id) >= 1))) {
        MS_EXCEPTION(ValueError) << "The value of the elements in LU_pivots must be greater than 1 "
                                 << "and less than the size of the penultimate dimension of LU_data.";
      }
      if ((*(Lu_pivots_working_ptr + id) - 1) == final_order[i]) {
        perm_pivots_id = i;
      }
    }
    std::swap(final_order[perm_id], final_order[perm_pivots_id]);
  }
  //  Index_select
  auto output_y0 = reinterpret_cast<T_data *>(outputs[kFirstOutputIndex]->addr);
  size_t indices_num = final_order.size();
  size_t inner_size = Lu_data_dim1_unsigned;
  size_t slice_size = static_cast<size_t>(inner_size * sizeof(T_data));
  for (size_t j = 0; j < indices_num; ++j) {
    size_t params_idx = static_cast<size_t>(final_order[j] * inner_size);
    size_t out_idx = j * inner_size;
    (void)std::memcpy(output_y0 + matrix_index * pivots_stride + out_idx, P_eye + params_idx, slice_size);
  }
}

template <typename T_data, typename T_pivots>
bool LuUnpackCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto input_0_Shape = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto input_1_Shape = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);

  size_t LU_data_dims = input_0_Shape.size();
  std::vector<int64_t> LU_data_dims_vector = input_0_Shape;
  int64_t Lu_data_dim1 = LU_data_dims_vector[LU_data_dims - 2];
  int64_t Lu_data_dim2 = LU_data_dims_vector[LU_data_dims - 1];

  size_t LU_pivots_dims = input_1_Shape.size();
  std::vector<int64_t> LU_pivots_dims_vector = input_1_Shape;
  int64_t Lu_pivots_dim = LU_pivots_dims_vector[LU_pivots_dims - 1];
  int64_t pivots_stride = Lu_data_dim1 * Lu_data_dim1;
  int64_t L_stride = 0;
  int64_t U_stride = 0;
  if (Lu_data_dim1 > Lu_data_dim2) {
    L_stride = Lu_data_dim1 * Lu_data_dim2;
    U_stride = Lu_data_dim2 * Lu_data_dim2;
  } else {
    L_stride = Lu_data_dim1 * Lu_data_dim1;
    U_stride = Lu_data_dim1 * Lu_data_dim2;
  }
  int64_t matrix_width = Lu_data_dim1;
  int64_t matrix_height = Lu_data_dim2;
  int64_t matrix_size = matrix_width * matrix_height;
  auto input_x1 = reinterpret_cast<T_pivots *>(inputs[1]->addr);

  int32_t block_size = Lu_data_dim1 * Lu_data_dim1;
  T_data *P_eye = new T_data[block_size]{};
  T_data num = static_cast<T_data>(1);
  for (int32_t i = 0; i < Lu_data_dim1; i++) {
    *(P_eye + (Lu_data_dim1 + 1) * i) = num;
  }

  int64_t Lu_data_stride = Lu_data_dim1 * Lu_data_dim2;
  int64_t Lu_pivots_stride = Lu_pivots_dim;
  int64_t batch_num =
    std::accumulate(input_0_Shape.begin(), input_0_Shape.end(), 1, std::multiplies<int>()) / Lu_data_stride;
  for (int64_t matrix_index = 0; matrix_index < batch_num; matrix_index++) {
    T_pivots *Lu_pivots_working_ptr = input_x1 + matrix_index * Lu_pivots_stride;
    LuUnpack(inputs, outputs, Lu_data_dim1, Lu_pivots_dim, Lu_pivots_working_ptr, matrix_index, matrix_size,
             matrix_width, matrix_height, pivots_stride, L_stride, U_stride, P_eye);
  }

  delete[] P_eye;
  return true;
}

std::vector<std::pair<KernelAttr, LuUnpackCpuKernelMod::LuUnpackFunc>> LuUnpackCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackCpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackCpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackCpuKernelMod::LaunchKernel<double, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackCpuKernelMod::LaunchKernel<double, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LuUnpackCpuKernelMod::LaunchKernel<double, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackCpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackCpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackCpuKernelMod::LaunchKernel<float, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackCpuKernelMod::LaunchKernel<float, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LuUnpackCpuKernelMod::LaunchKernel<float, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackCpuKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackCpuKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackCpuKernelMod::LaunchKernel<float16, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackCpuKernelMod::LaunchKernel<float16, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LuUnpackCpuKernelMod::LaunchKernel<float16, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackCpuKernelMod::LaunchKernel<int64_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackCpuKernelMod::LaunchKernel<int64_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &LuUnpackCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackCpuKernelMod::LaunchKernel<int32_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackCpuKernelMod::LaunchKernel<int32_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LuUnpackCpuKernelMod::LaunchKernel<int32_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackCpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackCpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackCpuKernelMod::LaunchKernel<int16_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackCpuKernelMod::LaunchKernel<int16_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &LuUnpackCpuKernelMod::LaunchKernel<int16_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &LuUnpackCpuKernelMod::LaunchKernel<int8_t, uint8_t>}};

std::vector<KernelAttr> LuUnpackCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LuUnpackFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LuUnpack, LuUnpackCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
