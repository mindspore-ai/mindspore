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

#include "plugin/device/cpu/kernel/scatter_add_with_axis_cpu_kernel.h"

#include <atomic>
#include <complex>

#include "kernel/common_utils.h"
#include "mindspore/core/ops/scatter_add_with_axis.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
#define DO_COMPUTE_CASE(DTYPE, TYPE, ITYPE, inputs, outputs) \
  case (DTYPE): {                                            \
    if ((ITYPE) == kNumberTypeInt32) {                       \
      LaunchKernel<TYPE, int32_t>(inputs, outputs);          \
      break;                                                 \
    } else {                                                 \
      LaunchKernel<TYPE, int64_t>(inputs, outputs);          \
      break;                                                 \
    }                                                        \
  }
}  // namespace

namespace mindspore {
namespace kernel {
namespace {
#define ADD_KERNEL(t1, t2, t3, t4) \
  KernelAttr()                     \
    .AddInputAttr(kNumberType##t1) \
    .AddInputAttr(kNumberType##t2) \
    .AddInputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)
const int32_t kInputNum = 3;
const int32_t kOutputNum = 1;
const int32_t KSplitSize = 64 * 1024;
}  // namespace

bool ScatterAddWithAxisCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto op_prim = std::dynamic_pointer_cast<ops::ScatterAddWithAxis>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  axis_ = op_prim->get_axis();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int ScatterAddWithAxisCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  x_type_ = inputs[kIndex0]->GetDtype();
  indices_type_ = inputs[kIndex1]->GetDtype();
  x_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  indices_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  updates_shape_ = inputs[kIndex2]->GetDeviceShapeAdaptively();

  // Get and check 3 input dim info
  int64_t value_dim_num_x1 = static_cast<int64_t>(x_shape_.size());
  int64_t value_dim_num_x2 = static_cast<int64_t>(indices_shape_.size());
  int64_t value_dim_num_x3 = static_cast<int64_t>(updates_shape_.size());
  if (value_dim_num_x1 != value_dim_num_x2 || value_dim_num_x2 != value_dim_num_x3) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dim values of three inputs must be same, but got "
                  << "data: " << value_dim_num_x1 << ", indices: " << value_dim_num_x2
                  << ", update: " << value_dim_num_x3;
    return KRET_RESIZE_FAILED;
  }
  if (axis_ < value_dim_num_x1 * -1 || axis_ >= value_dim_num_x1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of axis is out of range!";
    return KRET_RESIZE_FAILED;
  }

  int64_t sub_data_fix = 1;
  int64_t sub_index_fix = 1;
  data_dim_vec_.clear();
  index_dim_vec_.clear();
  for (int64_t i = value_dim_num_x2 - 1; i >= 0; --i) {
    size_t j = static_cast<size_t>(i);
    if (x_shape_[j] < indices_shape_[j] || indices_shape_[j] != updates_shape_[j] || updates_shape_[j] <= 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the " << j << " dimension verification failed: "
                    << "input0[" << x_shape_[j] << "], input1[" << indices_shape_[j] << "], input2["
                    << updates_shape_[j] << "]";
      return KRET_RESIZE_FAILED;
    }
    if (i > 0) {
      sub_data_fix *= x_shape_[j];
      data_dim_vec_.push_back(sub_data_fix);
      sub_index_fix *= indices_shape_[j];
      index_dim_vec_.push_back(sub_index_fix);
    }
  }
  return KRET_OK;
}

bool ScatterAddWithAxisCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  // check param
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  switch (x_type_) {
    DO_COMPUTE_CASE(kNumberTypeFloat16, float16, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeFloat32, float, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeFloat64, double, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeBool, bool, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeInt8, int8_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeInt16, int16_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeInt32, int32_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeInt64, int64_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeUInt8, uint8_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeUInt16, uint16_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeUInt32, uint32_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeUInt64, uint64_t, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeComplex64, std::complex<float>, indices_type_, inputs, outputs);
    DO_COMPUTE_CASE(kNumberTypeComplex128, std::complex<double>, indices_type_, inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the data type of input x ["
                              << TypeIdToType(x_type_)->ToString()
                              << "] is unsupported. It should be "
                                 "float16|float|double|bool|int8|int16|int32|int64|uint8|unint32|"
                                 "unit64|complex16|complex32.";
  }
  return true;
}

template <typename T, typename TI>
void ScatterAddWithAxisCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &outputs) {
  T *input_x1 = static_cast<T *>(inputs[0]->addr);
  TI *input_x2 = static_cast<TI *>(inputs[1]->addr);
  T *input_x3 = static_cast<T *>(inputs[2]->addr);
  T *output_y = static_cast<T *>(outputs[0]->addr);
  int64_t value_dim_num_x1 = static_cast<int64_t>(x_shape_.size());
  axis_ = axis_ < 0 ? axis_ + value_dim_num_x1 : axis_;
  int64_t axis_dim_value = static_cast<int64_t>(x_shape_[axis_]);
  int64_t total_value_num = static_cast<int64_t>(inputs[0]->size / sizeof(T));
  int64_t update_value_num = static_cast<int64_t>(inputs[2]->size / sizeof(T));

  // using input to initial output
  auto ret = memcpy_s(output_y, outputs[0]->size, input_x1, inputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret;
  }
  // update data by adding "updates" according to indices
  for (int64_t i = 0; i < update_value_num; ++i) {
    int64_t remain_index = i;
    int64_t index_value = 0;
    int64_t counter = 0;
    if (input_x2[i] < axis_dim_value * -1 || input_x2[i] >= axis_dim_value) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices value " << input_x2[i] << " is out of bound "
                               << axis_dim_value << "!";
    }
    int64_t input_x2_value = input_x2[i] < 0 ? input_x2[i] + axis_dim_value : input_x2[i];
    for (int64_t j = static_cast<int64_t>(index_dim_vec_.size()) - 1; j >= 0; --j) {
      int64_t index_tmp = counter == axis_ ? input_x2_value : remain_index / index_dim_vec_[LongToSize(j)];
      index_value += (index_tmp * data_dim_vec_[LongToSize(j)]);
      remain_index %= index_dim_vec_[LongToSize(j)];
      ++counter;
    }
    index_value += (counter == axis_ ? input_x2_value : remain_index);
    if (index_value >= total_value_num) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', update index " << index_value << "greater than "
                               << total_value_num << "which is overflow!";
    }
    output_y[index_value] += input_x3[i];
  }
}

std::vector<KernelAttr> ScatterAddWithAxisCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    ADD_KERNEL(UInt8, Int32, UInt8, UInt8),       ADD_KERNEL(UInt8, Int64, UInt8, UInt8),
    ADD_KERNEL(UInt16, Int32, UInt16, UInt16),    ADD_KERNEL(UInt16, Int64, UInt16, UInt16),
    ADD_KERNEL(UInt32, Int32, UInt32, UInt32),    ADD_KERNEL(UInt32, Int64, UInt32, UInt32),
    ADD_KERNEL(UInt64, Int32, UInt64, UInt64),    ADD_KERNEL(UInt64, Int64, UInt64, UInt64),
    ADD_KERNEL(Int8, Int32, Int8, Int8),          ADD_KERNEL(Int8, Int64, Int8, Int8),
    ADD_KERNEL(Int16, Int32, Int16, Int16),       ADD_KERNEL(Int16, Int64, Int16, Int16),
    ADD_KERNEL(Int32, Int32, Int32, Int32),       ADD_KERNEL(Int32, Int64, Int32, Int32),
    ADD_KERNEL(Int64, Int32, Int64, Int64),       ADD_KERNEL(Int64, Int64, Int64, Int64),
    ADD_KERNEL(Float16, Int32, Float16, Float16), ADD_KERNEL(Float16, Int64, Float16, Float16),
    ADD_KERNEL(Float32, Int32, Float32, Float32), ADD_KERNEL(Float32, Int64, Float32, Float32),
    ADD_KERNEL(Float64, Int32, Float64, Float64), ADD_KERNEL(Float64, Int64, Float64, Float64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterAddWithAxis, ScatterAddWithAxisCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
