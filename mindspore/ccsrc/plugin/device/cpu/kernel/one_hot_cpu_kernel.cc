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

#include "plugin/device/cpu/kernel/one_hot_cpu_kernel.h"
#include <string>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOneHotInputsNum = 5;
constexpr size_t kOneHotOutputsNum = 1;
#define INPUT_COMPUTE_CASE(DTYPE, TYPE, ODTYPE, INPUTS, OUTPUTS)             \
  case (DTYPE): {                                                            \
    switch (ODTYPE) {                                                        \
      INPUT_COMPUTE_CASE_INT(DTYPE, TYPE, ODTYPE, INPUTS, OUTPUTS)           \
      INPUT_COMPUTE_CASE_FLOAT(DTYPE, TYPE, ODTYPE, INPUTS, OUTPUTS)         \
      default:                                                               \
        MS_LOG(EXCEPTION) << " For OneHot the dtype of output not support."; \
    }                                                                        \
    break;                                                                   \
  }

#define INPUT_COMPUTE_CASE_INT(DTYPE, TYPE, ODTYPE, INPUTS, OUTPUTS)      \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeInt8, int8_t, INPUTS, OUTPUTS)     \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeInt16, int16_t, INPUTS, OUTPUTS)   \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeInt32, int32_t, INPUTS, OUTPUTS)   \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeInt64, int64_t, INPUTS, OUTPUTS)   \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeUInt8, uint8_t, INPUTS, OUTPUTS)   \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeUInt16, uint16_t, INPUTS, OUTPUTS) \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeUInt32, uint32_t, INPUTS, OUTPUTS) \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeUInt64, uint64_t, INPUTS, OUTPUTS)

#define INPUT_COMPUTE_CASE_FLOAT(DTYPE, TYPE, ODTYPE, INPUTS, OUTPUTS)                    \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeComplex64, std::complex<float>, INPUTS, OUTPUTS)   \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeComplex128, std::complex<double>, INPUTS, OUTPUTS) \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeFloat64, double, INPUTS, OUTPUTS)                  \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeFloat32, float_t, INPUTS, OUTPUTS)                 \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeFloat16, float16, INPUTS, OUTPUTS)                 \
  OUTPUT_COMPUTE_CASE(TYPE, kNumberTypeBool, bool, INPUTS, OUTPUTS)                       \
  OUTPUT_COMPUTE_CASE(TYPE, kObjectTypeString, std::string, INPUTS, OUTPUTS)

#define OUTPUT_COMPUTE_CASE(TYPE, ODTYPE, OTYPE, INPUTS, OUTPUTS) \
  case (ODTYPE): {                                                \
    LaunchKernel<TYPE, OTYPE>(INPUTS, OUTPUTS);                   \
    break;                                                        \
  }
}  // namespace

bool OneHotCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kOneHotInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOneHotOutputsNum, kernel_name_);

  input_dtype_ = inputs[kIndex0]->dtype_id();
  output_dtype_ = outputs[kIndex0]->dtype_id();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int OneHotCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto output_shape = outputs[kIndex0]->GetShapeVector();
  int64_t axis = inputs[axis_index_]->GetValueWithCheck<int64_t>();
  if (axis != -1 && LongToSize(axis) >= output_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'axis' must be -1, or an int which is less than the dimension of output, but got "
                      << axis << ", got the dimension of output " << output_shape.size();
  }
  if (axis == -1) {
    axis_ = output_shape.size() - 1;
  } else {
    axis_ = LongToSize(axis);
  }
  depth_ = LongToSize(output_shape[axis_]);
  stride_ = 1;
  for (size_t i = axis_ + 1; i < output_shape.size(); ++i) {
    stride_ *= LongToSize(output_shape[i]);
  }
  return KRET_OK;
}

bool OneHotCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                const std::vector<kernel::KernelTensor *> &,
                                const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kOneHotInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOneHotOutputsNum, kernel_name_);
  switch (input_dtype_) {
    INPUT_COMPUTE_CASE(kNumberTypeUInt8, uint8_t, output_dtype_, inputs, outputs);
    INPUT_COMPUTE_CASE(kNumberTypeInt32, int32_t, output_dtype_, inputs, outputs);
    INPUT_COMPUTE_CASE(kNumberTypeInt64, int64_t, output_dtype_, inputs, outputs);
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dtype of input 'x' " << TypeIdToType(input_dtype_)->ToString()
                    << " not support.";
      return false;
  }
  return true;
}

template <typename ID, typename OD>
void OneHotCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  const auto *indices = reinterpret_cast<ID *>(inputs[0]->device_ptr());
  auto on_value = GetDeviceAddress<OD>(inputs, kIndex2)[0];
  auto off_value = GetDeviceAddress<OD>(inputs, kIndex3)[0];
  auto *output = reinterpret_cast<OD *>(outputs[0]->device_ptr());
  size_t elem_num = inputs[0]->size() / sizeof(ID);
  auto task = [this, &indices, &on_value, &off_value, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t stride_num = i / stride_;
      size_t output_index = stride_num * depth_ * stride_ + i % stride_;
      if (indices[i] < 0) {
        for (size_t j = 0; j < depth_; j++) {
          output[output_index] = off_value;
          output_index += stride_;
        }
        continue;
      }

      size_t index = IntToSize(indices[i]);
      for (size_t j = 0; j < depth_; j++) {
        if (index == j) {
          output[output_index] = on_value;
        } else {
          output[output_index] = off_value;
        }
        output_index += stride_;
      }
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
}

std::vector<KernelAttr> OneHotCpuKernelMod::support_list_ = {KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kObjectTypeString),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kObjectTypeString),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                               .AddOutputAttr(kObjectTypeString)};
std::vector<KernelAttr> OneHotCpuKernelMod::GetOpSupport() { return support_list_; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, OneHot, OneHotCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
