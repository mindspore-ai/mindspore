/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/math/round_gpu_kernel.h"
#include <memory>
#include <type_traits>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool RoundGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RoundGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ele_num_ = SizeOf(inputs.at(kIndex0)->GetShapeVector());
  is_null_input_ = (ele_num_ == 0);
  return KRET_OK;
}

template <ElwiseOpType Op, typename Inp_t, typename Out_t>
bool RoundGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  const auto input_ptr = reinterpret_cast<Inp_t *>(inputs[kIndex0]->device_ptr());
  auto output_ptr = reinterpret_cast<Out_t *>(outputs[kIndex0]->device_ptr());

  auto decimals = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  if (decimals != 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " only support decimals equal 0, but got decimals equal "
                      << decimals;
    return false;
  }

  auto ret =
    UnaryOpsCudaFunc<Op, Inp_t, Out_t>(ele_num_, input_ptr, output_ptr, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(ret, kernel_name_);
  return true;
}

#define ADD_ROUND_TYPE(Op, NUMBER_TYPE, TYPE)                                                       \
  KernelAttr().AddInputAttr(NUMBER_TYPE).AddInputAttr(kNumberTypeInt64).AddOutputAttr(NUMBER_TYPE), \
    &RoundGpuKernelMod::LaunchKernel<Op, TYPE, TYPE>

std::vector<std::pair<KernelAttr, RoundGpuKernelMod::RoundFunc>> RoundGpuKernelMod::func_list_ = {
  {ADD_ROUND_TYPE(ElwiseOpType::kRound, kNumberTypeFloat16, half)},
  {ADD_ROUND_TYPE(ElwiseOpType::kRound, kNumberTypeFloat32, float)},
  {ADD_ROUND_TYPE(ElwiseOpType::kRound, kNumberTypeFloat64, double)},
  {ADD_ROUND_TYPE(ElwiseOpType::kRound, kNumberTypeInt32, int32_t)},
  {ADD_ROUND_TYPE(ElwiseOpType::kRound, kNumberTypeInt64, int64_t)}};

std::vector<KernelAttr> RoundGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RoundFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Round, RoundGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
