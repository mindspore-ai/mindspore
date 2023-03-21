/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sequence/scalar_cast_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <complex>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputsNum = 1;
constexpr int kOutputsNum = 1;
}  // namespace
bool ScalarCastCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ScalarCastCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  auto ele_shape = inputs[0]->GetShapeVector();
  if (!ele_shape.empty() && !(ele_shape.size() == 1 && ele_shape[0] == 1)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << " the input shape should be 0 or 1, but got " << ele_shape;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool ScalarCastCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  const auto ele_addr = GetDeviceAddress<T>(inputs, 0);
  S *output_addr = GetDeviceAddress<S>(outputs, 0);
  *output_addr = static_cast<S>(ele_addr[0]);
  return true;
}

#define ADD_TENSOR_KERNEL(x_dtype, y_dtype, x_type, y_type)                       \
  {                                                                               \
    KernelAttr().AddInputAttr(x_dtype).AddOutputAttr(kObjectTypeNumber, y_dtype), \
      &ScalarCastCpuKernelMod::LaunchKernel<x_type, y_type>                       \
  }

#define ADD_KERNEL(x_dtype, y_dtype, x_type, y_type)                                                 \
  {                                                                                                  \
    KernelAttr().AddInputAttr(kObjectTypeNumber, x_dtype).AddOutputAttr(kObjectTypeNumber, y_dtype), \
      &ScalarCastCpuKernelMod::LaunchKernel<x_type, y_type>                                          \
  }

static const std::vector<std::pair<KernelAttr, ScalarCastCpuKernelMod::KernelRunFunc>> func_list = {
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeUInt8, float, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeUInt16, float, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeUInt32, float, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeUInt64, float, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeInt8, float, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeInt16, float, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeInt32, float, int),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeFloat16, float, float16),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeFloat32, float, float),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeFloat64, float, double),
  ADD_TENSOR_KERNEL(kNumberTypeFloat32, kNumberTypeBool, float, bool),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeUInt8, double, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeUInt16, double, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeUInt32, double, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeUInt64, double, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeInt8, double, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeInt16, double, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeInt32, double, int),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeFloat16, double, float16),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeFloat32, double, float),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeFloat64, double, double),
  ADD_TENSOR_KERNEL(kNumberTypeFloat64, kNumberTypeBool, double, bool),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeUInt8, float16, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeUInt16, float16, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeUInt32, float16, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeUInt64, float16, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeInt8, float16, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeInt16, float16, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeInt32, float16, int),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeInt64, float16, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeFloat16, float16, float16),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeFloat32, float16, float),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeFloat64, float16, double),
  ADD_TENSOR_KERNEL(kNumberTypeFloat16, kNumberTypeBool, float16, bool),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeUInt8, uint8_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeUInt16, uint8_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeUInt32, uint8_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeUInt64, uint8_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeInt8, uint8_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeInt16, uint8_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeFloat16, uint8_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeFloat32, uint8_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeFloat64, uint8_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeUInt8, kNumberTypeBool, uint8_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeUInt8, uint16_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeUInt16, uint16_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeUInt32, uint16_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeUInt64, uint16_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeInt8, uint16_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeInt16, uint16_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeFloat16, uint16_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeFloat32, uint16_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeFloat64, uint16_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeUInt16, kNumberTypeBool, uint16_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeUInt8, uint32_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeUInt16, uint32_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeUInt32, uint32_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeUInt64, uint32_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeInt8, uint32_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeInt16, uint32_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeFloat16, uint32_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeFloat32, uint32_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeFloat64, uint32_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeUInt32, kNumberTypeBool, uint32_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeUInt8, uint64_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeUInt16, uint64_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeUInt32, uint64_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeUInt64, uint64_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeInt8, uint64_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeInt16, uint64_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeFloat16, uint64_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeFloat32, uint64_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeFloat64, uint64_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeUInt64, kNumberTypeBool, uint64_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeUInt8, int8_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeUInt16, int8_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeUInt32, int8_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeUInt64, int8_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeInt8, int8_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeInt16, int8_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeInt32, int8_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeFloat16, int8_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeFloat32, int8_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeFloat64, int8_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeInt8, kNumberTypeBool, int8_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeUInt8, int16_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeUInt16, int16_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeUInt32, int16_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeUInt64, int16_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeInt8, int16_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeInt16, int16_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeInt32, int16_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeFloat16, int16_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeFloat32, int16_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeFloat64, int16_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeInt16, kNumberTypeBool, int16_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeUInt8, int32_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeUInt16, int32_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeUInt32, int32_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeUInt64, int32_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeInt8, int32_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeInt16, int32_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeInt32, int32_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeFloat16, int32_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeFloat32, int32_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeFloat64, int32_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeInt32, kNumberTypeBool, int32_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeUInt8, int64_t, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeUInt16, int64_t, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeUInt32, int64_t, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeUInt64, int64_t, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeInt8, int64_t, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeInt16, int64_t, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeInt32, int64_t, int),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeFloat16, int64_t, float16),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
  ADD_TENSOR_KERNEL(kNumberTypeInt64, kNumberTypeBool, int64_t, bool),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeUInt8, bool, uint8_t),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeUInt16, bool, uint16_t),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeUInt32, bool, uint32_t),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeUInt64, bool, uint64_t),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeInt8, bool, int8_t),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeInt16, bool, int16_t),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeInt32, bool, int),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeInt64, bool, int64_t),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeFloat16, bool, float16),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeFloat32, bool, float),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeFloat64, bool, double),
  ADD_TENSOR_KERNEL(kNumberTypeBool, kNumberTypeBool, bool, bool),
  ADD_KERNEL(kNumberTypeFloat32, kNumberTypeFloat32, float, float),
  ADD_KERNEL(kNumberTypeFloat32, kNumberTypeFloat64, float, double),
  ADD_KERNEL(kNumberTypeFloat32, kNumberTypeInt32, float, int),
  ADD_KERNEL(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t),
  ADD_KERNEL(kNumberTypeFloat32, kNumberTypeBool, float, bool),
  ADD_KERNEL(kNumberTypeFloat64, kNumberTypeFloat32, double, float),
  ADD_KERNEL(kNumberTypeFloat64, kNumberTypeFloat64, double, double),
  ADD_KERNEL(kNumberTypeFloat64, kNumberTypeInt32, double, int),
  ADD_KERNEL(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t),
  ADD_KERNEL(kNumberTypeFloat64, kNumberTypeBool, double, bool),
  ADD_KERNEL(kNumberTypeInt32, kNumberTypeFloat32, int, float),
  ADD_KERNEL(kNumberTypeInt32, kNumberTypeFloat64, int, double),
  ADD_KERNEL(kNumberTypeInt32, kNumberTypeInt32, int, int),
  ADD_KERNEL(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
  ADD_KERNEL(kNumberTypeInt32, kNumberTypeBool, int, bool),
  ADD_KERNEL(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
  ADD_KERNEL(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
  ADD_KERNEL(kNumberTypeInt64, kNumberTypeInt32, int64_t, int),
  ADD_KERNEL(kNumberTypeInt64, kNumberTypeBool, int64_t, bool),
  ADD_KERNEL(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)};

const std::vector<std::pair<KernelAttr, ScalarCastCpuKernelMod::KernelRunFunc>> &ScalarCastCpuKernelMod::GetFuncList()
  const {
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScalarCast, ScalarCastCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
