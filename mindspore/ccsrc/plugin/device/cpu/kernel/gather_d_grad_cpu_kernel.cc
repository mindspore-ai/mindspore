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

#include "plugin/device/cpu/kernel/gather_d_grad_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
size_t get_element_num(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<size_t>());
}

template <typename I, typename T>
void GatherDGradCopyTask(size_t cur, std::vector<size_t> *pos, T *input, I *index, const int &dim, T *output,
                         const std::vector<size_t> &output_shape, const std::vector<size_t> &out_cargo_size,
                         const std::vector<size_t> &input_cargo_size) {
  for (size_t i = 0; i < output_shape[cur]; ++i) {
    (*pos)[cur] = i;
    if (cur == output_shape.size() - 1) {
      size_t input_offset = 0;
      size_t out_offset = 0;
      // out offset
      for (size_t j = 0; j < output_shape.size(); ++j) {
        out_offset += (*pos)[j] * out_cargo_size[j];
      }
      // input offset
      size_t cur_index = (*pos)[dim];
      (*pos)[dim] = index[out_offset];
      for (size_t j = 0; j < output_shape.size(); ++j) {
        input_offset += (*pos)[j] * input_cargo_size[j];
      }
      // do copy
      input[input_offset] += output[out_offset];
      (*pos)[dim] = cur_index;
    } else {
      // CopyTask
      GatherDGradCopyTask(cur + 1, pos, input, index, dim, output, output_shape, out_cargo_size, input_cargo_size);
    }
  }
}
}  // namespace

bool GatherDGradV2CpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (auto ret = MatchKernelFunc(kernel_name_, inputs, outputs); !ret) {
    return ret;
  }
  return true;
}

int GatherDGradV2CpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  index_shape_ = Convert2SizeT(inputs[kIndex2]->GetShapeVector());
  grad_shape_ = Convert2SizeT(inputs[kIndex3]->GetShapeVector());
  output_shape_ = Convert2SizeT(outputs[0]->GetShapeVector());
  return KRET_OK;
}

template <typename I, typename T>
bool GatherDGradV2CpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  auto *index = reinterpret_cast<I *>(inputs[kIndex2]->device_ptr());
  auto *grad = reinterpret_cast<T *>(inputs[kIndex3]->device_ptr());
  auto out = reinterpret_cast<T *>(outputs[0]->device_ptr());

  dim_value_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  if (dim_value_ < 0) {
    dim_value_ = dim_value_ + SizeToLong(output_shape_.size());
  }

  // check index
  auto index_size = get_element_num(index_shape_);
  int max_index = SizeToInt(output_shape_[dim_value_]);
  for (size_t i = 0; i < index_size; ++i) {
    if (index[i] >= max_index || index[i] < -max_index) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'index' must be in [" << -max_index << ", "
                        << max_index << "), but got: " << index[i];
    }
    if (index[i] < 0) {
      index[i] = max_index + index[i];
    }
  }
  // memset_s does not support data that more than 2GB.
  auto output_size = LongToSize(get_element_num(output_shape_)) * sizeof(T);
  auto output_addr = reinterpret_cast<char *>(outputs[0]->device_ptr());
  while (output_size > 0) {
    auto copy_size = std::min(output_size, static_cast<size_t>(INT32_MAX));
    auto ret = memset_s(output_addr, output_size, 0, copy_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
    }
    output_size -= copy_size;
    output_addr += copy_size;
  }

  // out_cargo_size
  std::vector<size_t> out_cargo_size = std::vector<size_t>(output_shape_.size(), 1);
  for (int i = static_cast<int>(out_cargo_size.size()) - 2; i >= 0; --i) {
    out_cargo_size[i] = output_shape_[i + 1] * out_cargo_size[i + 1];
  }
  // grad_cargo_size
  std::vector<size_t> grad_cargo_size = std::vector<size_t>(grad_shape_.size(), 1);
  for (int i = static_cast<int>(grad_cargo_size.size()) - 2; i >= 0; --i) {
    auto idx = IntToSize(i);
    grad_cargo_size[idx] = grad_shape_[idx + 1] * grad_cargo_size[idx + 1];
  }

  // copy task
  std::vector<size_t> pos(index_shape_.size(), 0);
  GatherDGradCopyTask<I, T>(0, &pos, out, index, dim_value_, grad, index_shape_, grad_cargo_size, out_cargo_size);
  return true;
}

#define REG_INDEX(DT1, DT2, T1, T2)                      \
  {                                                      \
    KernelAttr()                                         \
      .AddInputAttr(DT1)                                 \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
      .AddInputAttr(DT2)                                 \
      .AddInputAttr(DT1)                                 \
      .AddOutputAttr(DT1),                               \
      &GatherDGradV2CpuKernelMod::LaunchKernel<T2, T1>   \
  }

#define GATHER_D_GRAD_V2_CPU_REGISTER(DT, T) \
  REG_INDEX(DT, kNumberTypeInt64, T, int64_t), REG_INDEX(DT, kNumberTypeInt32, T, int32_t)

const std::vector<std::pair<KernelAttr, GatherDGradV2CpuKernelMod::KernelRunFunc>>
  &GatherDGradV2CpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, GatherDGradV2CpuKernelMod::KernelRunFunc>> func_list = {
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeComplex64, std::complex<float>),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeComplex128, std::complex<double>),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeFloat64, double),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeFloat32, float),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeFloat16, float16),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeUInt8, uint8_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeInt8, int8_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeUInt16, uint16_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeInt16, int16_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeUInt32, uint32_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeInt32, int32_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeUInt64, uint64_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeInt64, int64_t),
    GATHER_D_GRAD_V2_CPU_REGISTER(kNumberTypeBool, bool)};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GatherDGradV2, GatherDGradV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
