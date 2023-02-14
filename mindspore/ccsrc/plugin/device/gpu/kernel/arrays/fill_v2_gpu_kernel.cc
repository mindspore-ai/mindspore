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

#include "plugin/device/gpu/kernel/arrays/fill_v2_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fill_v2_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kFillV2InputsNum = 2;
constexpr int kFillV2OutputsNum = 1;
}  // namespace

bool FillV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int FillV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  output_size_ = SizeToLong(SizeOf(output_shape_));
  return KRET_OK;
}

template <typename DataType>
bool FillV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (output_size_ == 0) {
    return true;
  }

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFillV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFillV2OutputsNum, kernel_name_);

  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);

  DataType *input_ptr = GetDeviceAddress<DataType>(inputs, kIndex1);
  MS_ERROR_IF_NULL_W_RET_VAL(input_ptr, false);

  DataType *output_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(output_ptr, false);

  FillV2(output_size_, input_ptr, output_ptr, device_id_, cuda_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaGetLastError(), "FillV2 kernel failed.");

  return true;
}

#define FILL_V2_GPU_REG(MS_T, MS_S, T) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddOutputAttr(MS_S), &FillV2GpuKernelMod::LaunchKernel<T>

template <typename T>
using Complex = mindspore::utils::Complex<T>;

std::vector<std::pair<KernelAttr, FillV2GpuKernelMod::FillV2LaunchFunc>> FillV2GpuKernelMod::func_list_ = {
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeBool, bool)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeInt8, int8_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeInt16, int16_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int64_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeUInt8, uint8_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeUInt16, uint16_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeUInt32, uint32_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeUInt64, uint64_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeFloat16, half)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeFloat32, float)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeFloat64, double)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeComplex64, Complex<float>)},
  {FILL_V2_GPU_REG(kNumberTypeInt32, kNumberTypeComplex128, Complex<double>)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeBool, bool)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeInt8, int8_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeInt16, int16_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeUInt8, uint8_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeUInt16, uint16_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeUInt32, uint32_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeUInt64, uint64_t)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeFloat16, half)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeFloat32, float)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeFloat64, double)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeComplex64, Complex<float>)},
  {FILL_V2_GPU_REG(kNumberTypeInt64, kNumberTypeComplex128, Complex<double>)}};

std::vector<KernelAttr> FillV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, FillV2GpuKernelMod::FillV2LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FillV2, FillV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
