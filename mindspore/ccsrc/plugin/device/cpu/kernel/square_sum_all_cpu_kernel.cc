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

#include "plugin/device/cpu/kernel/square_sum_all_cpu_kernel.h"
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSquareSumAllInputsNum = 2;
constexpr size_t kSquareSumAllOutputsNum = 2;
constexpr float kPowerSquareExp = 2.0;

template <typename T>
void SquareSum(const T *in0, const T *in1, float *out0, float *out1, size_t start, size_t end) {
  for (size_t index = start; index < end; index++) {
    // as the size of both two input tensors are known to be identical, we can compute sum of two tensors in one for
    // loop.
    size_t split = end / kSquareSumAllInputsNum;
    if (index < split) {
      auto ret = pow(static_cast<float>(in0[index]), kPowerSquareExp);
      out0[0] = (index == 0) ? ret : out0[0] + ret;
    } else {
      auto ret = pow(static_cast<float>(in1[index - split]), kPowerSquareExp);
      out1[0] = (index == split) ? ret : out1[0] + ret;
    }
  }
}
}  // namespace

void SquareSumAllCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

template <typename T>
void SquareSumAllCpuKernelMod::InitWorkspaceSize() {
  (void)workspace_size_list_.emplace_back(sizeof(T));
  (void)workspace_size_list_.emplace_back(sizeof(T));
}

void SquareSumAllCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  DeprecatedNativeCpuKernelMod::InitInputOutputSize(kernel_node);
  if (dtype_ == kNumberTypeFloat16) {
    InitWorkspaceSize<float16>();
  } else if (dtype_ == kNumberTypeFloat32) {
    InitWorkspaceSize<float>();
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]: " << TypeIdToType(dtype_)->ToString();
  }
}

bool SquareSumAllCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  bool ret = true;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSquareSumAllInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSquareSumAllOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    ret = LaunchKernel<float16>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, workspace, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]: " << TypeIdToType(dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool SquareSumAllCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &workspace,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  const T *input_0_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const T *input_1_addr = reinterpret_cast<T *>(inputs[1]->addr);
  T *output_0_addr = reinterpret_cast<T *>(outputs[0]->addr);
  T *output_1_addr = reinterpret_cast<T *>(outputs[1]->addr);
  float *workspace_0_addr = reinterpret_cast<float *>(workspace[0]->addr);
  float *workspace_1_addr = reinterpret_cast<float *>(workspace[1]->addr);
  auto task = std::bind(SquareSum<T>, input_0_addr, input_1_addr, workspace_0_addr, workspace_1_addr,
                        std::placeholders::_1, std::placeholders::_2);
  ParallelLaunchAutoSearch(task, input_size_ * kSquareSumAllInputsNum, this, &parallel_search_info_);
  output_0_addr[0] = static_cast<T>(workspace_0_addr[0]);
  output_1_addr[0] = static_cast<T>(workspace_1_addr[0]);
  return true;
}

std::vector<KernelAttr> SquareSumAllCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),

    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SquareSumAll, SquareSumAllCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
