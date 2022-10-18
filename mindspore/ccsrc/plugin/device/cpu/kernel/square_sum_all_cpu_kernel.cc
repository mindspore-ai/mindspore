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
const char kBatchRank[] = "batch_rank";

template <typename T>
void SquareSum(const T *in0, const T *in1, float *out0, float *out1, int64_t batch_size, size_t start, size_t end) {
  for (size_t index = start; index < end; index++) {
    // as the size of both two input tensors are known to be identical, we can compute sum of two tensors in one for
    // loop.
    size_t split = end / kSquareSumAllInputsNum;
    if (index < split) {
      auto ret = pow(static_cast<float>(in0[index]), kPowerSquareExp);
      size_t batch_index = index / batch_size;
      out0[batch_index] = out0[batch_index] + ret;
    } else {
      auto ret = pow(static_cast<float>(in1[index - split]), kPowerSquareExp);
      size_t batch_index = (index - split) / batch_size;
      out1[batch_index] = out1[batch_index] + ret;
    }
  }
}
}  // namespace

bool SquareSumAllCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs.at(kIndex0)->GetDtype();
  dtype_size_ = abstract::TypeIdSize(dtype_);
  PrimitivePtr prim = base_operator->GetPrim();
  if (prim->HasAttr(kBatchRank)) {
    int64_t batch_rank = GetValue<int64_t>(prim->GetAttr(kBatchRank));
    batch_rank_ = LongToSize(batch_rank);
  }
  return true;
}

int SquareSumAllCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  num_batch_ =
    std::accumulate(input_shape.begin(), input_shape.begin() + batch_rank_, size_t(1), std::multiplies<size_t>());
  x_size_ = std::accumulate(input_shape.begin() + batch_rank_, input_shape.end(), size_t(1), std::multiplies<size_t>());
  workspace_size_list_.emplace_back(num_batch_ * sizeof(float));
  workspace_size_list_.emplace_back(num_batch_ * sizeof(float));
  return KRET_OK;
}

bool SquareSumAllCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  bool ret = true;
  if (input_size_ == 0) {
    return ret;
  }
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
  for (size_t i = 0; i < num_batch_; ++i) {
    workspace_0_addr[i] = static_cast<float>(0.0);
    workspace_1_addr[i] = static_cast<float>(0.0);
  }
  auto task = std::bind(SquareSum<T>, input_0_addr, input_1_addr, workspace_0_addr, workspace_1_addr, x_size_,
                        std::placeholders::_1, std::placeholders::_2);
  ParallelLaunchAutoSearch(task, input_size_ * kSquareSumAllInputsNum, this, &parallel_search_info_);
  for (size_t i = 0; i < num_batch_; ++i) {
    output_0_addr[i] = static_cast<T>(workspace_0_addr[i]);
    output_1_addr[i] = static_cast<T>(workspace_1_addr[i]);
  }
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
