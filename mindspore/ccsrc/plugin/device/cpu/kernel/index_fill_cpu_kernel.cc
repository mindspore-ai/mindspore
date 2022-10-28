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

#include "plugin/device/cpu/kernel/index_fill_cpu_kernel.h"

#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace {
#define INDEXFILL_COMPUTE_CASE(DTYPE, TYPE, inputs, outputs) \
  case (DTYPE): {                                            \
    LaunchKernel<TYPE>(inputs, outputs);                     \
    break;                                                   \
  }
}  // namespace

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
const uint32_t kInputIndex0 = 0;
const uint32_t kInputIndex1 = 1;
const uint32_t kInputIndex2 = 2;
const uint32_t kInputIndex3 = 3;
}  // namespace

bool IndexFillCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  x_type_ = inputs[kInputIndex0]->GetDtype();
  dim_type_ = inputs[kInputIndex1]->GetDtype();
  indices_type_ = inputs[kInputIndex2]->GetDtype();
  if (dim_type_ != kNumberTypeInt32) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the dtype of 'dim' must be int32 or int64, but got "
                            << dim_type_;
  }
  if (indices_type_ != kNumberTypeInt32) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the dtype of 'index' must be int32, but got "
                            << indices_type_;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int IndexFillCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs[kInputIndex0]->GetDeviceShapeAdaptively();
  dim_shape_ = inputs[kInputIndex1]->GetDeviceShapeAdaptively();
  indices_shape_ = inputs[kInputIndex2]->GetDeviceShapeAdaptively();
  value_shape_ = inputs[kInputIndex3]->GetDeviceShapeAdaptively();
  return ret;
}

bool IndexFillCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  switch (x_type_) {
    INDEXFILL_COMPUTE_CASE(kNumberTypeUInt8, uint8_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeUInt16, uint16_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeUInt32, uint32_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeUInt64, uint64_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeInt8, int8_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeInt16, int16_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeInt32, int32_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeInt64, int64_t, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeFloat16, float16, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeFloat32, float, inputs, outputs)
    INDEXFILL_COMPUTE_CASE(kNumberTypeFloat64, double, inputs, outputs)
    default:
      MS_EXCEPTION(TypeError) << "Unsupported input data type: " << TypeIdToType(x_type_)->ToString();
  }
  return true;
}

template <typename T>
void IndexFillCpuKernelMod::DoFill(int32_t data_num, const T *input_x, const int32_t *input_dim,
                                   const std::map<int32_t, bool> &index_dict, const T *input_value, T *output_y,
                                   const int32_t x_dim_nums) {
  int32_t dim_flag = 0;
  if (x_dim_nums != 0) {
    dim_flag = *input_dim % x_dim_nums + 1;
  }

  int32_t remain_dims = 1;
  if (dim_flag == x_dim_nums) {
    if (dim_flag != 0) {
      remain_dims = LongToInt(x_shape_[IntToSize(*input_dim)]);
    }
    for (int32_t i = 0; i < data_num; i++) {
      int32_t index_flag = 0;
      if (remain_dims != 0) {
        index_flag = i % remain_dims;
      }
      auto f = index_dict.find(index_flag);
      if (f != index_dict.end()) {
        output_y[i] = *input_value;
      } else {
        output_y[i] = input_x[i];
      }
    }
  } else {
    for (int32_t i = *input_dim + 1; i < x_dim_nums; i++) {
      remain_dims *= LongToInt(x_shape_[IntToSize(i)]);
    }
    for (int32_t i = 0; i < data_num; i++) {
      int32_t index_flag = 0;
      if (remain_dims != 0) {
        index_flag = (i / remain_dims) % LongToInt(x_shape_[IntToSize(*input_dim)]);
      }
      auto f = index_dict.find(index_flag);
      if (f != index_dict.end()) {
        output_y[i] = *input_value;
      } else {
        output_y[i] = input_x[i];
      }
    }
  }
}

template <typename T>
void IndexFillCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  T *input_0 = static_cast<T *>(inputs[0]->addr);
  int32_t *input_1 = static_cast<int32_t *>(inputs[1]->addr);
  int32_t *input_2 = static_cast<int32_t *>(inputs[2]->addr);
  T *input_3 = static_cast<T *>(inputs[3]->addr);
  T *output_0 = static_cast<T *>(outputs[0]->addr);
  int32_t x_dim_nums = static_cast<int32_t>(x_shape_.size());
  int32_t data_num = static_cast<int32_t>(inputs[0]->size / sizeof(T));
  uint32_t index_num = inputs[2]->size / sizeof(int32_t);
  int32_t cur_dim = *input_1;
  if (cur_dim < 0) {
    *input_1 = *input_1 + x_dim_nums;
  }

  std::map<int32_t, bool> index_dict;
  if (x_dim_nums == 0) {
    for (uint32_t i = 0; i < index_num; i++) {
      if (input_2[i] < -1 || input_2[i] > 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the input of 'index' out of range (expected to be in range of [-1, 0], "
                          << "but got " << input_2[i] << ")";
      } else {
        (void)index_dict.insert(std::pair<int32_t, bool>(0, true));
      }
    }
  } else if (cur_dim < -x_dim_nums || cur_dim >= x_dim_nums) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of 'dim' out of range (expected to be in range of ["
                      << (0 - x_dim_nums) << ", " << (x_dim_nums - 1) << "], but got " << cur_dim << ")";
  } else {
    for (uint32_t i = 0; i < index_num; i++) {
      int32_t cur_index = LongToInt(x_shape_[IntToSize(*input_1)]);
      if (input_2[i] < -cur_index || input_2[i] >= cur_index) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the input of 'index' out of range (expected to be in range of [" << (0 - cur_index)
                          << ", " << (cur_index - 1) << "], but got " << input_2[i] << ")";
      } else {
        if (input_2[i] < 0) {
          input_2[i] = input_2[i] + cur_index;
        }
        (void)index_dict.insert(std::pair<int32_t, bool>(input_2[i], true));
      }
    }
  }

  DoFill<T>(data_num, input_0, input_1, index_dict, input_3, output_0, x_dim_nums);
}

std::vector<KernelAttr> IndexFillCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddOutputAttr(kNumberTypeUInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddOutputAttr(kNumberTypeUInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeUInt32)
                                                   .AddOutputAttr(kNumberTypeUInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeUInt64)
                                                   .AddOutputAttr(kNumberTypeUInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddOutputAttr(kNumberTypeInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddOutputAttr(kNumberTypeInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IndexFill, IndexFillCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
