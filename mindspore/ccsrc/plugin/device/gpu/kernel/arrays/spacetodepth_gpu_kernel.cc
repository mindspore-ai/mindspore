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

#include "plugin/device/gpu/kernel/arrays/spacetodepth_gpu_kernel.h"
#include <map>
#include <utility>
namespace mindspore {
namespace kernel {
using KernelRunFunc = SpaceToDepthGpuKernelMod::KernelRunFunc;
const int64_t MIN_BLOCK_SIZE = 2;
const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

bool SpaceToDepthGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  auto attr_pointer = std::dynamic_pointer_cast<ops::SpaceToDepth>(base_operator);
  block_size_ = attr_pointer->get_block_size();
  if (block_size_ < MIN_BLOCK_SIZE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'block_size' cannot be less than 2, but got "
                      << block_size_;
  }
  return true;
}
int SpaceToDepthGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  // check input num and output num
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
  auto input_shape = inputs[0]->GetShapeVector();
  if (input_shape.size() != SPACETODEPTH_BUFFER_DIMENSION) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to "
                      << SPACETODEPTH_BUFFER_DIMENSION << ", but got " << shape_size_;
  }
  in_ = static_cast<size_t>(input_shape[DIM_0]);
  ic_ = static_cast<size_t>(input_shape[DIM_1]);
  ih_ = static_cast<size_t>(input_shape[DIM_2]);
  iw_ = static_cast<size_t>(input_shape[DIM_3]);

  on_ = in_;
  oc_ = ic_ * block_size_ * block_size_;
  oh_ = ih_ / block_size_;
  ow_ = iw_ / block_size_;
  return KRET_OK;
}

template <typename T>
bool SpaceToDepthGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  // get device buffer ptr
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  // get input size
  size_t size = in_ * ic_ * ih_ * iw_;
  // call cuda kernel
  CalSpaceToDepth(size, input, in_, ic_, ih_, iw_, on_, oc_, oh_, ow_, block_size_, output,
                  reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}
#define DTYPE_REGISTER_ATTR(INPUT, OUTPUT, T) \
  { KernelAttr().AddInputAttr(INPUT).AddOutputAttr(OUTPUT), &SpaceToDepthGpuKernelMod::LaunchKernel<T> }

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SpaceToDepthGpuKernelMod::GetFuncList() const {
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
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SpaceToDepth, SpaceToDepthGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
