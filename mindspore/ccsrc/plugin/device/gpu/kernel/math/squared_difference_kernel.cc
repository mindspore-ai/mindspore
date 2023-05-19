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

#include "plugin/device/gpu/kernel/math/squared_difference_kernel.h"
#include <map>
#include <utility>
namespace mindspore {
namespace kernel {
using KernelRunFunc = SquaredDifferenceOpGpuKernelMod::KernelRunFunc;
bool SquaredDifferenceOpGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}
int SquaredDifferenceOpGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape1 = Convert2SizeTClipNeg(inputs[0]->GetShapeVector());
  auto input_shape2 = Convert2SizeTClipNeg(inputs[1]->GetShapeVector());
  auto output_shape = Convert2SizeTClipNeg(outputs[0]->GetShapeVector());
  need_broadcast_ = false;
  if (input_shape1.size() != input_shape2.size()) {
    need_broadcast_ = true;
  } else {
    for (size_t i = 0; i < input_shape1.size(); i++) {
      if (input_shape1[i] != input_shape2[i]) {
        need_broadcast_ = true;
      }
    }
  }

  if (need_broadcast_ && output_shape.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be greater than " << MAX_DIMS
                      << ", but got " << output_shape.size();
  }

  lhs_shape_.resize(MAX_DIMS, 1);
  rhs_shape_.resize(MAX_DIMS, 1);
  output_shape_.resize(MAX_DIMS, 1);
  output_num_ = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    if (need_broadcast_) {
      output_shape_[i] = output_shape[i];
    }
    output_num_ *= static_cast<size_t>(output_shape[i]);
  }
  size_t lhs_offset = output_shape.size() - input_shape1.size();
  for (size_t j = 0; j < input_shape1.size(); j++) {
    if (need_broadcast_) {
      if ((j + lhs_offset) < MAX_DIMS) {
        lhs_shape_[j + lhs_offset] = input_shape1[j];
      } else {
        auto index = j + lhs_offset;
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of input cannot be " << index << ", but got "
                          << index;
      }
    }
  }
  size_t rhs_offset = output_shape.size() - input_shape2.size();
  for (size_t k = 0; k < input_shape2.size(); k++) {
    if (need_broadcast_) {
      if ((k + rhs_offset) < MAX_DIMS) {
        rhs_shape_[k + rhs_offset] = input_shape2[k];
      } else {
        auto index = k + rhs_offset;
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of input cannot be " << index << ", but got "
                          << index;
      }
    }
  }
  return KRET_OK;
}

template <typename T>
bool SquaredDifferenceOpGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  T *lhs = GetDeviceAddress<T>(inputs, 0);
  T *rhs = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  if (need_broadcast_) {
    BroadcastArith(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output,
                   reinterpret_cast<cudaStream_t>(stream_ptr_));
  } else {
    ElewiseArith(output_num_, op_type_, lhs, rhs, output, reinterpret_cast<cudaStream_t>(stream_ptr_));
  }
  return true;
}

template <typename T>
bool SquaredDifferenceOpGpuKernelMod::LaunchComplexKernel(const std::vector<AddressPtr> &inputs,
                                                          const std::vector<AddressPtr> &workspace,
                                                          const std::vector<AddressPtr> &outputs) {
  T *lhs = GetDeviceAddress<T>(inputs, 0);
  T *rhs = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  if (need_broadcast_) {
    BroadcastComplexArith(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output,
                          reinterpret_cast<cudaStream_t>(stream_ptr_));
  } else {
    ElewiseComplexArith(output_num_, op_type_, lhs, rhs, output, reinterpret_cast<cudaStream_t>(stream_ptr_));
  }
  return true;
}

#define DTYPE_REGISTER_ATTR(INPUT1, INPUT2, OUTPUT, T)                            \
  {                                                                               \
    KernelAttr().AddInputAttr(INPUT1).AddInputAttr(INPUT2).AddOutputAttr(OUTPUT), \
      &SquaredDifferenceOpGpuKernelMod::LaunchKernel<T>                           \
  }

#define COMPLEX_REGISTER_ATTR(INPUT1, INPUT2, OUTPUT, T)                          \
  {                                                                               \
    KernelAttr().AddInputAttr(INPUT1).AddInputAttr(INPUT2).AddOutputAttr(OUTPUT), \
      &SquaredDifferenceOpGpuKernelMod::LaunchComplexKernel<T>                    \
  }

template <typename T>
using Complex = mindspore::utils::Complex<T>;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SquaredDifferenceOpGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    DTYPE_REGISTER_ATTR(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, float),
    DTYPE_REGISTER_ATTR(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeFloat64, double),
    COMPLEX_REGISTER_ATTR(kNumberTypeComplex64, kNumberTypeComplex64, kNumberTypeComplex64, Complex<float>),
    COMPLEX_REGISTER_ATTR(kNumberTypeComplex128, kNumberTypeComplex128, kNumberTypeComplex128, Complex<double>),
    DTYPE_REGISTER_ATTR(kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, half),
    DTYPE_REGISTER_ATTR(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t),
    DTYPE_REGISTER_ATTR(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int)};

  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SquaredDifference, SquaredDifferenceOpGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
