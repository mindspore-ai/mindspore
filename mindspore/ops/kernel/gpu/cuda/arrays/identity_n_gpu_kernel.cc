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

#include "kernel/gpu/cuda/arrays/identity_n_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = IdentityNGpuKernelMod::KernelRunFunc;
}  // namespace
#define IDENTITY_N_GPU_REGISTER(T_DT, T) \
  KernelAttr().AddAllSameAttr(true).AddInputAttr(T_DT).AddOutputAttr(T_DT), &IdentityNGpuKernelMod::LaunchKernel<T>

bool IdentityNGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int IdentityNGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  return KRET_OK;
}

template <typename T>
bool IdentityNGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  for (uint64_t i = 0; i < inputs.size(); i++) {
    T *input_addr = GetDeviceAddress<T>(inputs, i);
    T *output_addr = GetDeviceAddress<T>(outputs, i);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_addr, input_addr, inputs[i]->size(), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "cudaMemcpyAsync value variable failed.");
  }
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &IdentityNGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {IDENTITY_N_GPU_REGISTER(kNumberTypeFloat32, float)},   {IDENTITY_N_GPU_REGISTER(kNumberTypeFloat16, half)},
    {IDENTITY_N_GPU_REGISTER(kNumberTypeFloat64, double)},  {IDENTITY_N_GPU_REGISTER(kNumberTypeInt8, int8_t)},
    {IDENTITY_N_GPU_REGISTER(kNumberTypeInt16, int16_t)},   {IDENTITY_N_GPU_REGISTER(kNumberTypeInt32, int)},
    {IDENTITY_N_GPU_REGISTER(kNumberTypeInt64, int64_t)},   {IDENTITY_N_GPU_REGISTER(kNumberTypeUInt8, uint8_t)},
    {IDENTITY_N_GPU_REGISTER(kNumberTypeUInt16, uint16_t)}, {IDENTITY_N_GPU_REGISTER(kNumberTypeUInt32, uint32_t)},
    {IDENTITY_N_GPU_REGISTER(kNumberTypeUInt64, uint64_t)}, {IDENTITY_N_GPU_REGISTER(kNumberTypeBool, bool)}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, IdentityN, IdentityNGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
