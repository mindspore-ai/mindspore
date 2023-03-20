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
#include "mindspore/core/ops/one_hot.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOneHotInputsNum = 4;
constexpr size_t kOneHotOutputsNum = 1;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
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

inline void check_input_num(size_t input_num, const std::string &kernel_name) {
  if (input_num != kOneHotInputsNum) {
    MS_LOG_EXCEPTION << "For " << kernel_name << ", input num must be " << kOneHotInputsNum << ", but got "
                     << input_num;
  }
}

bool OneHotCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  auto input_size = inputs.size();
  check_input_num(input_size, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);

  input_dtype_ = inputs[kIndex0]->GetDtype();
  output_dtype_ = outputs[kIndex0]->GetDtype();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int OneHotCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto output_shape = outputs[kIndex0]->GetShapeVector();
  auto one_hot_ptr = std::dynamic_pointer_cast<ops::OneHot>(base_operator);
  MS_EXCEPTION_IF_NULL(one_hot_ptr);
  int64_t axis = one_hot_ptr->get_axis();
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

bool OneHotCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  check_input_num(inputs.size(), kernel_name_);
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
void OneHotCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto *indices = reinterpret_cast<ID *>(inputs[0]->addr);
  auto on_value = GetDeviceAddress<OD>(inputs, kIndex2)[0];
  auto off_value = GetDeviceAddress<OD>(inputs, kIndex3)[0];
  auto *output = reinterpret_cast<OD *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(ID);
  auto task = [this, &indices, &on_value, &off_value, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t stride_num = i / stride_;
      size_t output_index = stride_num * depth_ * stride_ + i % stride_;
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
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddOutputAttr(kObjectTypeString),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddOutputAttr(kObjectTypeString),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddOutputAttr(kObjectTypeString),
                                                             // depth is a input with int64 type:
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddOutputAttr(kObjectTypeString),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddOutputAttr(kObjectTypeString),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddInputAttr(kNumberTypeUInt8)
                                                               .AddOutputAttr(kNumberTypeUInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddInputAttr(kNumberTypeUInt16)
                                                               .AddOutputAttr(kNumberTypeUInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddInputAttr(kNumberTypeUInt32)
                                                               .AddOutputAttr(kNumberTypeUInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddInputAttr(kNumberTypeUInt64)
                                                               .AddOutputAttr(kNumberTypeUInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddInputAttr(kNumberTypeInt8)
                                                               .AddOutputAttr(kNumberTypeInt8),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddInputAttr(kNumberTypeInt16)
                                                               .AddOutputAttr(kNumberTypeInt16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddInputAttr(kNumberTypeInt32)
                                                               .AddOutputAttr(kNumberTypeInt32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddOutputAttr(kNumberTypeInt64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddInputAttr(kNumberTypeFloat16)
                                                               .AddOutputAttr(kNumberTypeFloat16),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddInputAttr(kNumberTypeFloat32)
                                                               .AddOutputAttr(kNumberTypeFloat32),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddInputAttr(kNumberTypeFloat64)
                                                               .AddOutputAttr(kNumberTypeFloat64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddInputAttr(kNumberTypeBool)
                                                               .AddOutputAttr(kNumberTypeBool),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddInputAttr(kNumberTypeComplex64)
                                                               .AddOutputAttr(kNumberTypeComplex64),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddInputAttr(kNumberTypeComplex128)
                                                               .AddOutputAttr(kNumberTypeComplex128),
                                                             KernelAttr()
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kNumberTypeInt64)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddInputAttr(kObjectTypeString)
                                                               .AddOutputAttr(kObjectTypeString)};
std::vector<KernelAttr> OneHotCpuKernelMod::GetOpSupport() { return support_list_; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, OneHot, OneHotCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
