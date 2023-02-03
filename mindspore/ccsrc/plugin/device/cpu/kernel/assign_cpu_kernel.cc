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

#include "plugin/device/cpu/kernel/assign_cpu_kernel.h"

#include <complex>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAssignInputsNum = 2;
constexpr size_t kAssignOutputsNum = 1;
const std::map<TypeId, size_t> input_x_dtype_size_map = {{kNumberTypeBool, sizeof(bool)},
                                                         {kNumberTypeInt8, sizeof(int8_t)},
                                                         {kNumberTypeInt16, sizeof(int16_t)},
                                                         {kNumberTypeInt32, sizeof(int32_t)},
                                                         {kNumberTypeInt64, sizeof(int64_t)},
                                                         {kNumberTypeUInt8, sizeof(uint8_t)},
                                                         {kNumberTypeUInt16, sizeof(uint16_t)},
                                                         {kNumberTypeUInt32, sizeof(uint32_t)},
                                                         {kNumberTypeUInt64, sizeof(uint64_t)},
                                                         {kNumberTypeFloat16, sizeof(float16)},
                                                         {kNumberTypeFloat32, sizeof(float)},
                                                         {kNumberTypeFloat64, sizeof(double)},
                                                         {kNumberTypeComplex64, sizeof(std::complex<float>)},
                                                         {kNumberTypeComplex128, sizeof(std::complex<double>)}};
}  // namespace

bool AssignCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  does not support " << kernel_attr;
    return false;
  }

  input_x_dtype_ = inputs[0]->GetDtype();
  return true;
}

int AssignCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAssignInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAssignOutputsNum, kernel_name_);
  std::vector<int64_t> input_x_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> input_y_shape = inputs[kIndex1]->GetShapeVector();
  if (input_x_shape.size() != input_y_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'x' and 'y' must have the same dimension, but got the dimension of 'x': "
                      << input_x_shape.size() << " and the dimension of 'y': " << input_y_shape.size();
  }
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    if (input_x_shape[i] != input_y_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'x' and 'y' must have the same shape, but got the shape of 'x': "
                        << Vector2Str(input_x_shape) << " and the shape of 'y': " << Vector2Str(input_y_shape);
    }
    batch_size_ *= LongToSize(input_x_shape[i]);
  }
  auto type_len = input_x_dtype_size_map.find(input_x_dtype_);
  if (type_len == input_x_dtype_size_map.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' must be bool, int, uint, float, complex, but got "
                      << input_x_dtype_;
  }
  input_x_dtype_size_ = type_len->second;

  return KRET_OK;
}

bool AssignCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  bool ret = true;
  switch (input_x_dtype_) {
    case (kNumberTypeBool): {
      ret = LaunchKernel<bool>(inputs, outputs);
      break;
    }
    case (kNumberTypeInt8): {
      ret = LaunchKernel<int8_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeInt16): {
      ret = LaunchKernel<int16_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeInt32): {
      ret = LaunchKernel<int32_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeInt64): {
      ret = LaunchKernel<int64_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeUInt8): {
      ret = LaunchKernel<uint8_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeUInt16): {
      ret = LaunchKernel<uint16_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeUInt32): {
      ret = LaunchKernel<uint32_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeUInt64): {
      ret = LaunchKernel<uint64_t>(inputs, outputs);
      break;
    }
    case (kNumberTypeFloat16): {
      ret = LaunchKernel<float16>(inputs, outputs);
      break;
    }
    case (kNumberTypeFloat32): {
      ret = LaunchKernel<float>(inputs, outputs);
      break;
    }
    case (kNumberTypeFloat64): {
      ret = LaunchKernel<double>(inputs, outputs);
      break;
    }
    case (kNumberTypeComplex64): {
      ret = LaunchKernel<std::complex<float>>(inputs, outputs);
      break;
    }
    case (kNumberTypeComplex128): {
      ret = LaunchKernel<std::complex<double>>(inputs, outputs);
      break;
    }
    default:
      ret = false;
      MS_LOG(EXCEPTION) << "For 'Assign', unsupported input data type: " << TypeIdToString(input_x_dtype_);
  }
  return ret;
}

template <typename T>
bool AssignCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto max_size = inputs[0]->size;
  size_t total_size = input_x_dtype_size_ * batch_size_;
  if (total_size > max_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', memcpy size must be less than or equal to max size, but got memcpy size: " << total_size
                      << ", and max size: " << max_size;
  }

  auto input0_addr = reinterpret_cast<int8_t *>(inputs[0]->addr);
  auto input1_addr = reinterpret_cast<int8_t *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<int8_t *>(outputs[0]->addr);
  auto task = [&](size_t start, size_t end) {
    int8_t *input0 = input0_addr + start;
    int8_t *input1 = input1_addr + start;
    int8_t *output = output_addr + start;
    size_t length = end - start;
    size_t max_length = total_size - start;
    int ret = memcpy_s(input0, max_length, input1, length);
    if (ret != 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy_s error. Error no " << ret;
    }
    ret = memcpy_s(output, max_length, input1, length);
    if (ret != 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy_s error. Error no " << ret;
    }
  };
  ParallelLaunchAutoSearch(task, total_size, this, &parallel_search_info_);
  return true;
}

std::vector<KernelAttr> AssignCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeComplex64)
      .AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeComplex128)
      .AddOutputAttr(kNumberTypeComplex128),
  };

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Assign, AssignCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
