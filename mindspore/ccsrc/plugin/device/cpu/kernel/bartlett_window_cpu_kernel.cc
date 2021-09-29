/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/bartlett_window_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBartlettWindowInputsNum = 1;
constexpr size_t kBartlettWindowOutputsNum = 1;
}  // namespace

void BartlettWindowCpuKernelMod<T, S>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  output_dtype = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  periodic_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, PERIODIC);
  if ((input_dtype != kNumberTypeInt32) && (input_dtype != kNumberTypeInt64)) {
    MS_LOG(EXCEPTION) << "Input tensor types must be int32 or int64";
  }
  input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (input_shape.size() > 0) {
    MS_EXCEPTION(ValueError) << "The dim of window_length must be 0.";
  }
}

// template <typename T, typename S>
// void BartlettWindowCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
//                                            const std::vector<AddressPtr> &outputs) {
//   auto input = reinterpret_cast<T *>(inputs[0]->addr);
//   auto output = reinterpret_cast<S *>(outputs[0]->addr);
//   auto input_data = *input;
//   const size_t window_length = static_cast<size_t>(*input);
//   const S output_one = static_cast<S>(1.);
//   if (input_data < 0) {
//     MS_EXCEPTION(ValueError) << "Input window_length must ≥ 0!";
//   }
//   if (input_data == 1) {
//     *output = output_one;
//   } else {
//     if (periodic_) {
//       input_data += 1;
//     }
//     const size_t first_half_size = static_cast<size_t>((input_data - 1) / 2);
//     const double x = static_cast<double>(input_data);
//     for (size_t i = 0; i <= first_half_size; i++) {
//       auto value = static_cast<S>((2. * i) / (x - 1.));
//       *(output + i) = value;
//     }
//     for (size_t i = first_half_size + 1; i < window_length; i++) {
//       auto value = static_cast<S>(2. - (2. * i) / (x - 1.));
//       *(output + i) = value;
//     }
//   }
// }

template <typename T, typename S>
bool BartlettWindowCpuKernelMod<T, S>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBartlettWindowInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBartlettWindowOutputsNum, kernel_name_);
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<S *>(outputs[0]->addr);
  auto input_data = *input;
  const size_t window_length = static_cast<size_t>(*input);
  const S output_one = static_cast<S>(1.);
  if (input_data < 0) {
    MS_EXCEPTION(ValueError) << "Input window_length must ≥ 0!";
  }
  if (input_data == 1) {
    *output = output_one;
  } else {
    if (periodic_) {
      input_data += 1;
    }
    const size_t first_half_size = static_cast<size_t>((input_data - 1) / 2);
    const double x = static_cast<double>(input_data);
    for (size_t i = 0; i <= first_half_size; i++) {
      auto value = static_cast<S>((2. * i) / (x - 1.));
      *(output + i) = value;
    }
    for (size_t i = first_half_size + 1; i < window_length; i++) {
      auto value = static_cast<S>(2. - (2. * i) / (x - 1.));
      *(output + i) = value;
    }
  }
  // if (output_dtype == kNumberTypeFloat16) {
  //   if (input_dtype == kNumberTypeInt32) {
  //     LaunchKernel<int32_t, float16>(inputs, outputs);
  //   } else if (input_dtype == kNumberTypeInt64) {
  //     LaunchKernel<int64_t, float16>(inputs, outputs);
  //   }
  // } else if (output_dtype == kNumberTypeFloat32) {
  //   if (input_dtype == kNumberTypeInt32) {
  //     LaunchKernel<int32_t, float>(inputs, outputs);
  //   } else if (input_dtype == kNumberTypeInt64) {
  //     LaunchKernel<int64_t, float>(inputs, outputs);
  //   }
  // } else if (output_dtype == kNumberTypeFloat64) {
  //   if (input_dtype == kNumberTypeInt32) {
  //     LaunchKernel<int32_t, double>(inputs, outputs);
  //   } else if (input_dtype == kNumberTypeInt64) {
  //     LaunchKernel<int64_t, double>(inputs, outputs);
  //   }
  // }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
