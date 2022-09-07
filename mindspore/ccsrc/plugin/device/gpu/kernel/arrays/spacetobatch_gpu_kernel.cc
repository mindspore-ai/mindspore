/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/spacetobatch_gpu_kernel.h"
#include <map>
#include <utility>
namespace mindspore {
namespace kernel {
using KernelRunFunc = SpaceToBatchGpuKernelMod::KernelRunFunc;
const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;
bool SpaceToBatchGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  auto attr_pointer = std::dynamic_pointer_cast<ops::SpaceToBatch>(base_operator);
  block_size_ = static_cast<size_t>(GetValue<int64_t>(base_operator->GetAttr("block_size")));
  if (block_size_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'block_size' cannot be less than 1, but got "
                      << block_size_;
  }
  paddings_ = attr_pointer->get_paddings();
  if (paddings_.size() != PADDING_SHAPE_0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'paddings' cannot be equal to " << PADDING_SHAPE_0
                      << ", but got " << paddings_.size();
  }
  if (paddings_[0].size() != PADDING_SHAPE_1 || paddings_[1].size() != PADDING_SHAPE_1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of 'paddings' cannot be equal to " << PADDING_SHAPE_0
                      << ", but got " << paddings_.size();
  }
  return true;
}

int SpaceToBatchGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  size_t input_num = inputs.size();
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_num;
  }
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  // check input_shape
  input_shape_ = Convert2SizeTClipNeg(inputs[0]->GetShapeVector());
  if (input_shape_.size() != SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to " << SHAPE_SIZE
                      << ", but got " << input_shape_.size();
  }
  // check paddings_ shape
  for (size_t idx_i = 0; idx_i < PADDING_SHAPE_0; ++idx_i) {
    for (size_t idx_j = 0; idx_j < PADDING_SHAPE_1; ++idx_j) {
      if (paddings_[idx_i][idx_j] < 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the element of 'paddings' cannot be less than 0, "
                          << "but got paddings[" << idx_i << "][ " << idx_j << "]: " << paddings_[idx_i][idx_j];
      }
    }
    auto tmp_shape = input_shape_[idx_i + PADDING_SHAPE_1] + paddings_[idx_i][0] + paddings_[idx_i][1];
    if ((tmp_shape % block_size_) != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', padded shape must be divisible by block_size, , but got padded shape: " << tmp_shape
                        << ", block_size: " << block_size_;
    }
    if ((tmp_shape / block_size_) == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', padded shape cannot be less than block_size"
                        << ", but got padded shape: " << tmp_shape << ", block_size: " << block_size_;
    }
  }
  in_ = input_shape_[DIM_0];
  ic_ = input_shape_[DIM_1];
  ih_ = input_shape_[DIM_2];
  iw_ = input_shape_[DIM_3];
  on_ = in_ * block_size_ * block_size_;
  oc_ = ic_;
  oh_ = (ih_ + LongToSize(paddings_[0][0] + paddings_[0][1])) / block_size_;
  ow_ = (iw_ + LongToSize(paddings_[1][0] + paddings_[1][1])) / block_size_;
  return KRET_OK;
}

template <typename T>
bool SpaceToBatchGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);

  size_t size = in_ * ih_ * iw_ * ic_;

  CalSpaceToBatch<T>(size, input, in_, ih_, iw_, ic_, on_, oh_, ow_, oc_, LongToSize(paddings_[0][0]),
                     LongToSize(paddings_[0][1]), LongToSize(paddings_[1][0]), LongToSize(paddings_[1][1]), block_size_,
                     output, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}
#define DTYPE_REGISTER_ATTR(INPUT, OUTPUT, T) \
  { KernelAttr().AddInputAttr(INPUT).AddOutputAttr(OUTPUT), &SpaceToBatchGpuKernelMod::LaunchKernel<T> }

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SpaceToBatchGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    DTYPE_REGISTER_ATTR(kNumberTypeFloat32, kNumberTypeFloat32, float),
    DTYPE_REGISTER_ATTR(kNumberTypeFloat16, kNumberTypeFloat16, half),
    DTYPE_REGISTER_ATTR(kNumberTypeInt32, kNumberTypeInt32, int),
    DTYPE_REGISTER_ATTR(kNumberTypeUInt32, kNumberTypeUInt32, uint32_t),
    DTYPE_REGISTER_ATTR(kNumberTypeInt64, kNumberTypeInt64, int64_t),
    DTYPE_REGISTER_ATTR(kNumberTypeUInt64, kNumberTypeUInt64, uint64_t),
    DTYPE_REGISTER_ATTR(kNumberTypeInt16, kNumberTypeInt16, int16_t),
    DTYPE_REGISTER_ATTR(kNumberTypeUInt16, kNumberTypeUInt16, uint16_t),
    DTYPE_REGISTER_ATTR(kNumberTypeInt8, kNumberTypeInt8, int8_t),
    DTYPE_REGISTER_ATTR(kNumberTypeUInt8, kNumberTypeUInt8, uint8_t)};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SpaceToBatch, SpaceToBatchGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
