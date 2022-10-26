/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <utility>

#include "mindspore/core/ops/space_to_batch_nd.h"
#include "plugin/device/cpu/kernel/space_to_batch_nd_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t PADDING_SHAPE_1 = 2;
constexpr size_t kSpaceToBatchNDInputsNum = 1;
constexpr size_t kSpaceToBatchNDOutputsNum = 1;
constexpr char kKernelName[] = "SpaceToBatchND";
using KernelRunFunc = SpaceToBatchNDCpuKernelMod::KernelRunFunc;
}  // namespace
void SpaceToBatchNDCpuKernelMod::CheckParam() {
  for (size_t i = 0; i < block_rank_; i++) {
    if (block_size_[i] < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the elements of 'block_size' should be both larger than 1, but got " << i
                        << "'th block size " << block_size_[i] << ")\n";
    }
  }

  // check paddings_
  if (paddings_.size() != block_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the size of 'paddings' should be equal to the length of 'block_size':  " << block_rank_
                      << ", but got " << paddings_.size();
  }

  for (size_t idx_i = 0; idx_i < block_rank_; ++idx_i) {
    if (paddings_[idx_i].size() != PADDING_SHAPE_1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the size of each vector of 'paddings' should be equal to the length of 'block_size': "
                        << PADDING_SHAPE_1 << ", but got " << idx_i << "'th element: " << paddings_[idx_i].size();
    }
    for (size_t idx_j = 0; idx_j < PADDING_SHAPE_1; ++idx_j) {
      if (paddings_[idx_i][idx_j] < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the element of 'paddings' cannot be less than 0, "
                          << "but got paddings[" << idx_i << "][ " << idx_j << "]: " << paddings_[idx_i][idx_j];
      }
    }
    auto tmp_shape = input_shape_[idx_i + off_set_] + paddings_[idx_i][0] + paddings_[idx_i][1];
    if ((tmp_shape % block_size_[idx_i]) != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', padded shape should be divisible by block_size, but got padded shape: " << tmp_shape
                        << ", block_size: " << block_size_[idx_i];
    }
    if ((tmp_shape / block_size_[idx_i]) == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', padded shape cannot be less than block_size"
                        << ", but got padded shape: " << tmp_shape << ", block_size: " << block_size_[idx_i];
    }
  }
}

template <typename T>
bool SpaceToBatchNDCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  // check all shapes, blocks and paddings are valid
  CheckParam();

  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  int ret = memset_s(output, outputs[0]->size, 0, sizeof(T) * static_cast<size_t>(output_size_));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s error. Error no: " << ret;
  }

  for (int64_t pos = 0; pos < input_size_; pos += 1) {
    std::vector<int64_t> input_index(input_shape_.size(), 0);
    int64_t cur_pos = pos;
    for (int rev_i = SizeToInt(input_shape_.size()) - 1; rev_i >= 0; rev_i -= 1) {
      input_index[IntToSize(rev_i)] = cur_pos % input_shape_[IntToSize(rev_i)];
      cur_pos = cur_pos / input_shape_[IntToSize(rev_i)];
    }

    std::vector<int64_t> output_index(input_index);
    int64_t idx_on = 0;
    for (size_t i = off_set_; i < input_shape_.size(); i += 1) {
      output_index[i] = (input_index[i] + paddings_[i - off_set_][0]) / block_size_[i - off_set_];
      idx_on =
        idx_on * block_size_[i - off_set_] + (input_index[i] + paddings_[i - off_set_][0]) % block_size_[i - off_set_];
    }

    output_index[0] = idx_on * input_shape_[0] + input_index[0];

    int64_t out_pos = 0;

    for (size_t i = 0; i < output_shape_.size(); i += 1) {
      out_pos = out_pos * output_shape_[i] + output_index[i];
    }

    output[out_pos] = input[pos];
  }
  return true;
}

bool SpaceToBatchNDCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SpaceToBatchND>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast SpaceToBatchND ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();

  if (inputs.size() != kSpaceToBatchNDInputsNum || outputs.size() != kSpaceToBatchNDOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kSpaceToBatchNDInputsNum << " and "
                  << kSpaceToBatchNDOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  block_size_ = kernel_ptr->get_block_shape();
  paddings_ = kernel_ptr->get_paddings();
  block_rank_ = block_size_.size();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int SpaceToBatchNDCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost) == static_cast<int>(KRET_RESIZE_FAILED)) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return static_cast<int>(KRET_RESIZE_FAILED);
  }
  // get input_shape
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();

  if (input_shape_.size() < block_size_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input size should be no less than the block size, but get input size: "
                      << input_shape_.size() << " block size: " << block_size_.size();
  }

  input_size_ = 1;
  output_size_ = 1;
  for (size_t i = 0; i < input_shape_.size(); ++i) {
    input_size_ *= input_shape_[i];
  }
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ *= output_shape_[i];
  }

  off_set_ = input_shape_.size() - block_size_.size();

  return static_cast<int>(KRET_OK);
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SpaceToBatchNDCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpaceToBatchNDCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SpaceToBatchND, SpaceToBatchNDCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
