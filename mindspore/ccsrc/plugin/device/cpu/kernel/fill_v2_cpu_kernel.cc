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

#include "plugin/device/cpu/kernel/fill_v2_cpu_kernel.h"

#include <cmath>
#include <string>
#include <thread>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kFillV2InputsNum = 2;
constexpr size_t kFillV2OutputsNum = 1;
}  // namespace

bool FillV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  input1_dtype_ = inputs[0]->GetDtype();
  output_dtype_ = outputs[0]->GetDtype();
  return true;
}

int FillV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  output_shape_ = outputs[0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

bool FillV2CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  // Check the number of input and output
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFillV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFillV2OutputsNum, kernel_name_);

  // Get the shape of the output based on the first input
  std::vector<int64_t> dims;
  switch (input1_dtype_) {
    case (kNumberTypeInt32):
      CalculateDims<int32_t>(inputs[0], &dims);
      break;
    case (kNumberTypeInt64):
      CalculateDims<int64_t>(inputs[0], &dims);
      break;
    default:
      MS_LOG(EXCEPTION) << "the datatype of the input1 not support, support datatype: int32, int64.";
  }

  // Check output shape
  auto output = outputs[0];
  std::vector<int64_t> output_new_shape_;
  auto num = output_shape_.size();
  for (size_t i = 0; i < num; i++) {
    auto element = output_shape_[i];
    output_new_shape_.emplace_back(element);
  }
  if (output_new_shape_ != dims) {
    MS_LOG(EXCEPTION) << "the shape of output is error, the data of the input1 not match the shape of the output.";
  }

  // Fill according to the different data types of the output
  auto value = inputs[1];
  switch (output_dtype_) {
    case (kNumberTypeBool):
      LaunchKernel<bool>(&output, value);
      break;
    case (kNumberTypeInt8):
      LaunchKernel<int8_t>(&output, value);
      break;
    case (kNumberTypeInt16):
      LaunchKernel<int16_t>(&output, value);
      break;
    case (kNumberTypeInt32):
      LaunchKernel<int32_t>(&output, value);
      break;
    case (kNumberTypeInt64):
      LaunchKernel<int64_t>(&output, value);
      break;
    case (kNumberTypeUInt8):
      LaunchKernel<uint8_t>(&output, value);
      break;
    case (kNumberTypeUInt16):
      LaunchKernel<uint16_t>(&output, value);
      break;
    case (kNumberTypeUInt32):
      LaunchKernel<uint32_t>(&output, value);
      break;
    case (kNumberTypeUInt64):
      LaunchKernel<uint64_t>(&output, value);
      break;
    case (kNumberTypeFloat16):
      LaunchKernel<float16>(&output, value);
      break;
    case (kNumberTypeFloat32):
      LaunchKernel<float>(&output, value);
      break;
    case (kNumberTypeFloat64):
      LaunchKernel<double>(&output, value);
      break;
    default:
      MS_LOG(EXCEPTION) << "the datatype of the input2 not support, support datatype: "
                           "bool, int8, int16, int32, int64, uint8, uint16, uint32, "
                           "uint64, float16, float32, float64.";
  }
  return true;
}

template <typename T>
void FillV2CpuKernelMod::CalculateDims(const AddressPtr &input, std::vector<int64_t> *dims) const {
  MS_EXCEPTION_IF_NULL(input);
  auto *input_data = reinterpret_cast<T *>(input->addr);
  size_t data_num = input->size / sizeof(T);
  for (size_t i = 0; i < data_num; i++) {
    auto dim = static_cast<int64_t>(input_data[i]);
    if (dim < 0) {
      MS_LOG(EXCEPTION) << "the data of the input1 must all be greater than 0, there is a negative value in input1.";
    }
    if (dim == 0) {
      MS_LOG(EXCEPTION) << "the data of the input1 must all be greater than 0, there is a zero value in input1.";
    }
    (*dims).emplace_back(dim);
  }
}

template <typename T>
void FillV2CpuKernelMod::LaunchKernel(AddressPtr *output, const AddressPtr &value) {
  auto *output_data = reinterpret_cast<T *>((*output)->addr);
  auto *value_data = reinterpret_cast<T *>((value->addr));
  size_t lens = static_cast<size_t>((*output)->size / sizeof(T));
  auto task = [output_data, value_data](const size_t start, const size_t end) {
    for (size_t i = start; i < end; i++) {
      output_data[i] = *value_data;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

std::vector<KernelAttr> FillV2CpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FillV2, FillV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
