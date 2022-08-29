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

#include "plugin/device/cpu/kernel/zeta_cpu_kernel.h"
#include <functional>
#include <vector>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace {
const std::size_t kZetaInputNum{2u};
const std::size_t kZetaOutputNum{1u};
}  // namespace

namespace mindspore {
namespace kernel {
void ZetaCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input0_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input1_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

template <typename T>
inline T ScalarZeta(T a, T b) {
  return Eigen::numext::zeta(a, b);
}

template <typename T>
bool ZetaCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  T *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  T *input1 = reinterpret_cast<T *>(inputs[1]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  std::size_t total = inputs[0]->size / sizeof(T);
  auto task = [input0, input1, output](std::size_t begin, std::size_t end) {
    std::transform(input0 + begin, input0 + end, input1 + begin, output + begin, ScalarZeta<T>);
  };
  ParallelLaunchAutoSearch(task, total, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool ZetaCpuKernelMod::CheckZeta(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> &outputs, std::size_t inputs_num,
                                 std::size_t outputs_num) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  if (!IsSameShape(input0_shape_, input1_shape_)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of 'x' should be same with the shape of 'q', "
                      << "but got the shape of 'x': " << Vector2Str(input0_shape_)
                      << " and 'q': " << Vector2Str(input1_shape_);
  }
  if (!IsSameShape(input0_shape_, output_shape_)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape of output should be same with the shape of the 'x', "
                      << "but got the shape of the output: " << Vector2Str(output_shape_)
                      << " and 'x': " << Vector2Str(input0_shape_);
  }
  return true;
}

bool ZetaCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> &workspace,
                              const std::vector<kernel::AddressPtr> &outputs) {
  switch (dtype_) {
    case (kNumberTypeFloat32):
      if (CheckZeta<float>(inputs, outputs, kZetaInputNum, kZetaOutputNum) == false) {
        return false;
      }
      return LaunchKernel<float>(inputs, outputs);
      break;
    case (kNumberTypeFloat64):
      if (CheckZeta<double>(inputs, outputs, kZetaInputNum, kZetaOutputNum) == false) {
        return false;
      }
      return LaunchKernel<double>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "the datatype of the input not support, support datatype: "
                           "float32, float64.";
  }
  return true;
}

std::vector<KernelAttr> ZetaCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  };
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Zeta, ZetaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
