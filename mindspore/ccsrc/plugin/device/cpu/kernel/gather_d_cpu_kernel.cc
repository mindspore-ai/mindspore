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

bool GatherDCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  const size_t kThree = 3;
  if (inputs.size() != kThree) {
    MS_LOG(ERROR) << "GatherD input size must be equal to 3!";
    return false;
  }
  if (auto ret = MatchKernelFunc(base_operator, inputs, outputs); !ret) {
    return ret;
  }
  return true;
}

int GatherDCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  const size_t kIndexIdx = 2;
  input_shape_ = Convert2SizeT(inputs[0]->GetShapeVector());
  index_shape_ = Convert2SizeT(inputs[kIndexIdx]->GetShapeVector());
  if (input_shape_.size() != index_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', shape size of 'x' must be equal to 'index', but got shape size of 'x': "
                      << input_shape_.size() << ", and shape size of 'index': " << index_shape_.size();
  }
  output_shape_ = index_shape_;

  return KRET_OK;
}

template <typename T, typename I>
bool GatherDCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGatherDInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGatherDOutputsNum, kernel_name_);
  auto input_size = LongToSize(get_element_num(input_shape_)) * sizeof(T);
  auto index_size = LongToSize(get_element_num(index_shape_)) * sizeof(I);
  size_t dim_size = sizeof(int);
  auto output_size = LongToSize(get_element_num(output_shape_)) * sizeof(T);
  if (inputs[0]->size != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'x' must be " << input_size << ", but got "
                      << inputs[0]->size << ".";
  }
  if (inputs[1]->size != dim_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'dim' must be " << dim_size << ", but got "
                      << inputs[1]->size << ".";
  }
  if (inputs[2]->size != index_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'index' must be " << index_size
                      << ", but got " << inputs[2]->size << ".";
  }
  if (outputs[0]->size != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of output must be " << output_size
                      << ", but got " << outputs[0]->size << ".";
  }
  auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *dim = reinterpret_cast<int32_t *>(inputs[1]->addr);
  auto *index = reinterpret_cast<I *>(inputs[2]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  int32_t input_rank = SizeToInt(input_shape_.size());
  if (dim[0] >= input_rank || dim[0] < -input_rank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dim' must be in [" << -input_rank << ", "
                      << input_rank << "), but got: " << dim[0];
  }
  if (dim[0] < 0) {
    dim[0] = static_cast<int>(dim[0] + input_rank);
  }
  // check index
  int max_index = SizeToInt(input_shape_[IntToSize(dim[0])]);
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
  int copy_dim = *dim;
  CopyTask<T, I>(0, &pos, input, index, copy_dim, output, output_shape_, out_cargo_size, input_cargo_size, false);
  return true;
}

const std::vector<std::pair<KernelAttr, GatherDCpuKernelMod::KernelRunFunc>> &GatherDCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, GatherDCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &GatherDCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &GatherDCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &GatherDCpuKernelMod::LaunchKernel<float16, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &GatherDCpuKernelMod::LaunchKernel<float16, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &GatherDCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &GatherDCpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &GatherDCpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &GatherDCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeBool),
     &GatherDCpuKernelMod::LaunchKernel<bool, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool),
     &GatherDCpuKernelMod::LaunchKernel<bool, int64_t>}};

  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GatherD, GatherDCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
