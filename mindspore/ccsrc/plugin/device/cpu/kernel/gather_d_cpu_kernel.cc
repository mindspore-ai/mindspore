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
#include "plugin/device/cpu/kernel/gather_d_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGatherDInputsNum = 3;
constexpr size_t kGatherDOutputsNum = 1;

int64_t get_element_num(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return SizeToLong(size);
}

template <typename T, typename I>
void CopyTask(size_t cur, std::vector<size_t> *pos, T *input, const I *index, const int &dim, T *output,
              const std::vector<size_t> &output_shape, const std::vector<size_t> &out_cargo_size,
              const std::vector<size_t> &input_cargo_size, bool reverse) {
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
      if (reverse) {
        input[input_offset] = output[out_offset];
      } else {
        output[out_offset] = input[input_offset];
      }
      (*pos)[dim] = cur_index;
    } else {
      // CopyTask
      CopyTask(cur + 1, pos, input, index, dim, output, output_shape, out_cargo_size, input_cargo_size, reverse);
    }
  }
}
}  // namespace

bool GatherDCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  const size_t kThree = 3;
  if (inputs.size() != kThree) {
    MS_LOG(ERROR) << "GatherD input size must be equal to 3!";
    return false;
  }
  if (auto ret = MatchKernelFunc(kernel_name_, inputs, outputs); !ret) {
    return ret;
  }
  return true;
}

int GatherDCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shape_ = Convert2SizeT(inputs[kIndex0]->GetShapeVector());
  index_shape_ = Convert2SizeT(inputs[kIndex2]->GetShapeVector());
  output_shape_ = index_shape_;
  return KRET_OK;
}

template <typename T, typename I>
bool GatherDCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &outputs) {
  auto input_size = LongToSize(get_element_num(input_shape_)) * sizeof(T);
  auto index_size = LongToSize(get_element_num(index_shape_)) * sizeof(I);
  auto output_size = LongToSize(get_element_num(output_shape_)) * sizeof(T);
  if (inputs[0]->size() != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'x' must be " << input_size << ", but got "
                      << inputs[0]->size() << ".";
  }
  if (inputs[2]->size() != index_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'index' must be " << index_size
                      << ", but got " << inputs[2]->size() << ".";
  }
  if (outputs[0]->size() != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of output must be " << output_size
                      << ", but got " << outputs[0]->size() << ".";
  }
  auto *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *dim = reinterpret_cast<int64_t *>(inputs[1]->device_ptr());
  auto *index = reinterpret_cast<I *>(inputs[2]->device_ptr());
  auto output = reinterpret_cast<T *>(outputs[0]->device_ptr());

  int32_t input_rank = SizeToInt(input_shape_.size());
  int32_t copy_dim = LongToInt(*dim);
  copy_dim = copy_dim < 0 ? copy_dim + input_rank : copy_dim;

  // check index
  int max_index = SizeToInt(input_shape_[IntToSize(copy_dim)]);
  index_size = LongToSize(get_element_num(index_shape_));
  for (size_t i = 0; i < index_size; ++i) {
    if (index[i] >= max_index || index[i] < -max_index) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'index' must be in [" << -max_index << ", "
                        << max_index << "), but got: " << index[i];
    }
    if (index[i] < 0) {
      index[i] = max_index + index[i];
    }
  }

  // out_cargo_size
  std::vector<size_t> out_cargo_size = std::vector<size_t>(output_shape_.size(), 1);
  for (int i = SizeToInt(out_cargo_size.size()) - 2; i >= 0; --i) {
    out_cargo_size[i] = output_shape_[i + 1] * out_cargo_size[i + 1];
  }
  // input_cargo_size
  std::vector<size_t> input_cargo_size = std::vector<size_t>(input_shape_.size(), 1);
  for (int i = SizeToInt(input_cargo_size.size()) - 2; i >= 0; --i) {
    input_cargo_size[i] = input_shape_[i + 1] * input_cargo_size[i + 1];
  }
  // copy task
  std::vector<size_t> pos(index_shape_.size(), 0);
  CopyTask<T, I>(0, &pos, input, index, copy_dim, output, output_shape_, out_cargo_size, input_cargo_size, false);
  return true;
}

#define REG_INDEX(DT1, DT2, T1, T2)                      \
  {                                                      \
    KernelAttr()                                         \
      .AddInputAttr(DT1)                                 \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
      .AddInputAttr(DT2)                                 \
      .AddOutputAttr(DT1),                               \
      &GatherDCpuKernelMod::LaunchKernel<T1, T2>         \
  }

#define GATHER_D_CPU_REGISTER(DT, T) \
  REG_INDEX(DT, kNumberTypeInt64, T, int64_t), REG_INDEX(DT, kNumberTypeInt32, T, int32_t)

const std::vector<std::pair<KernelAttr, GatherDCpuKernelMod::KernelRunFunc>> &GatherDCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, GatherDCpuKernelMod::KernelRunFunc>> func_list = {
    GATHER_D_CPU_REGISTER(kNumberTypeComplex64, std::complex<float>),
    GATHER_D_CPU_REGISTER(kNumberTypeComplex128, std::complex<double>),
    GATHER_D_CPU_REGISTER(kNumberTypeFloat64, double),
    GATHER_D_CPU_REGISTER(kNumberTypeFloat32, float),
    GATHER_D_CPU_REGISTER(kNumberTypeFloat16, float16),
    GATHER_D_CPU_REGISTER(kNumberTypeUInt8, uint8_t),
    GATHER_D_CPU_REGISTER(kNumberTypeInt8, int8_t),
    GATHER_D_CPU_REGISTER(kNumberTypeUInt16, uint16_t),
    GATHER_D_CPU_REGISTER(kNumberTypeInt16, int16_t),
    GATHER_D_CPU_REGISTER(kNumberTypeUInt32, uint32_t),
    GATHER_D_CPU_REGISTER(kNumberTypeInt32, int32_t),
    GATHER_D_CPU_REGISTER(kNumberTypeUInt64, uint64_t),
    GATHER_D_CPU_REGISTER(kNumberTypeInt64, int64_t),
    GATHER_D_CPU_REGISTER(kNumberTypeBool, bool)};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GatherD, GatherDCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
