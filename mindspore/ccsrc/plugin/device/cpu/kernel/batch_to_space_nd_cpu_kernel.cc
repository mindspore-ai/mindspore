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
#include <string>
#include <utility>

#include "mindspore/core/ops/batch_to_space_nd.h"
#include "plugin/device/cpu/kernel/batch_to_space_nd_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
constexpr size_t CROP_SHAPE_1 = 2;
constexpr size_t kBatchToSpaceNDInputsNum = 1;
constexpr size_t kBatchToSpaceNDOutputsNum = 1;
constexpr char kKernelName[] = "BatchToSpaceND";

bool BatchToSpaceNDCpuKernelMod::CheckParam() {
  if (crops_.size() != block_rank_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the size of 'crops' should be equal to the length of 'block_shape':  " << block_rank_
                  << ", but got " << crops_.size();
    return false;
  }

  int64_t block_shape_prod = 1;
  for (size_t idx_i = 0; idx_i < block_rank_; ++idx_i) {
    if (block_shape_[idx_i] < 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the elements of 'block_shape' should be both larger than 1, but got " << idx_i
                    << "'th block size " << block_shape_[idx_i] << ")\n";
      return false;
    }
    block_shape_prod = block_shape_prod * block_shape_[idx_i];
    if (crops_[idx_i].size() != CROP_SHAPE_1) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the size of each vector of 'crops' should be equal to the length of 'block_shape': "
                    << CROP_SHAPE_1 << ", but got " << idx_i << "'th element: " << crops_[idx_i].size();
      return false;
    }
    for (size_t idx_j = 0; idx_j < CROP_SHAPE_1; ++idx_j) {
      if (crops_[idx_i][idx_j] < 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the element of 'crops' cannot be less than 0, "
                      << "but got crops[" << idx_i << "][ " << idx_j << "]: " << crops_[idx_i][idx_j];
        return false;
      }
    }
  }

  if (input_shape_[0] % block_shape_prod != 0) {
    MS_LOG(ERROR)
      << "For '" << kernel_name_
      << "', the first dim of 'input_x' must be divisible by 'block_shape_prod'. But got first dim of 'input_x': "
      << input_shape_[0] << ", 'block_shape_prod' with value: " << block_shape_prod << ".";
    return false;
  }
  return true;
}

template <typename T>
bool BatchToSpaceNDCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  // check all shapes, blocks and crops are valid
  if (!CheckParam()) {
    return false;
  }

  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  int ret = memset_s(output, outputs[0]->size, 0, sizeof(T) * LongToSize(output_size_));
  if (ret != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', memset_s error. Error no: " << ret;
  }

  for (int64_t pos = 0; pos < output_size_; pos += 1) {
    std::vector<int64_t> output_index(output_shape_.size(), 0);
    int64_t cur_pos = pos;
    for (int rev_i = SizeToInt(output_shape_.size()) - 1; rev_i >= 0; rev_i -= 1) {
      auto idx = IntToSize(rev_i);
      output_index[idx] = cur_pos % output_shape_[idx];
      cur_pos = cur_pos / output_shape_[idx];
    }

    std::vector<int64_t> input_index(output_index);
    int64_t idx_on = 0;
    for (size_t i = off_set_; i < output_shape_.size(); i += 1) {
      input_index[i] = (output_index[i] + crops_[i - off_set_][0]) / block_shape_[i - off_set_];
      idx_on =
        idx_on * block_shape_[i - off_set_] + (output_index[i] + crops_[i - off_set_][0]) % block_shape_[i - off_set_];
    }

    input_index[0] = idx_on * output_shape_[0] + output_index[0];

    int64_t in_pos = 0;

    for (size_t i = 0; i < input_shape_.size(); i += 1) {
      in_pos = in_pos * input_shape_[i] + input_index[i];
    }

    output[pos] = input[in_pos];
  }
  return true;
}

bool BatchToSpaceNDCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BatchToSpaceND>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast BatchToSpaceND ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();

  if (inputs.size() != kBatchToSpaceNDInputsNum || outputs.size() != kBatchToSpaceNDOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kBatchToSpaceNDInputsNum << " and "
                  << kBatchToSpaceNDOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  block_shape_ = kernel_ptr->get_block_shape();
  crops_ = kernel_ptr->get_crops();
  block_rank_ = block_shape_.size();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int BatchToSpaceNDCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto res = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); res != KRET_OK) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return res;
  }
  // get input_shape
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();

  input_size_ = 1;
  output_size_ = 1;
  for (size_t i = 0; i < input_shape_.size(); ++i) {
    input_size_ = input_shape_[i] * input_size_;
  }
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_size_ = output_shape_[i] * output_size_;
  }

  if (static_cast<int>(block_shape_.size()) - static_cast<int>(input_shape_.size()) >= 0) {
    MS_LOG(ERROR) << kernel_name_ << " resize failed because input shape should be greater than block shape, "
                  << "but input shape is " << input_shape_ << " and block shape is " << block_shape_;
    return KRET_RESIZE_FAILED;
  }
  off_set_ = input_shape_.size() - block_shape_.size();

  return KRET_OK;
}

std::vector<KernelAttr> BatchToSpaceNDCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8)},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16)},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32)},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32)},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64)},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)},
  };
  return support_list;
}

using FuncList = std::vector<std::pair<KernelAttr, BatchToSpaceNDCpuKernelMod::KernelRunFunc>>;
const FuncList &BatchToSpaceNDCpuKernelMod::GetFuncList() const {
  static const FuncList func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &BatchToSpaceNDCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BatchToSpaceND, BatchToSpaceNDCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
