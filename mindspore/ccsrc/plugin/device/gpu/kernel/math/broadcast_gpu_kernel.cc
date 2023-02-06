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

#include "plugin/device/gpu/kernel/math/broadcast_gpu_kernel.h"

#include <iostream>

namespace mindspore {
namespace kernel {
#define MS_REG_BROADCAST_COMPLEX_GPU_KERNEL1(T0_MS_DTYPE, T1_MS_DTYPE, T0_DTYPE, T1_DTYPE)     \
  KernelAttr().AddInputAttr(T0_MS_DTYPE).AddInputAttr(T0_MS_DTYPE).AddOutputAttr(T0_MS_DTYPE), \
    &BroadcastOpGpuKernelMod::LaunchComplexKernel<T0_DTYPE, T0_DTYPE, T0_DTYPE>

#define MS_REG_BROADCAST_COMPLEX_GPU_KERNEL2(T0_MS_DTYPE, T1_MS_DTYPE, T0_DTYPE, T1_DTYPE)     \
  KernelAttr().AddInputAttr(T0_MS_DTYPE).AddInputAttr(T1_MS_DTYPE).AddOutputAttr(T0_MS_DTYPE), \
    &BroadcastOpGpuKernelMod::LaunchComplexKernel<T0_DTYPE, T1_DTYPE, T0_DTYPE>

#define MS_REG_BROADCAST_COMPLEX_GPU_KERNEL3(T0_MS_DTYPE, T1_MS_DTYPE, T0_DTYPE, T1_DTYPE)     \
  KernelAttr().AddInputAttr(T1_MS_DTYPE).AddInputAttr(T0_MS_DTYPE).AddOutputAttr(T0_MS_DTYPE), \
    &BroadcastOpGpuKernelMod::LaunchComplexKernel<T1_DTYPE, T0_DTYPE, T0_DTYPE>

bool BroadcastOpGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  support_complex_ = false;
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetOpType()) {
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

bool BroadcastOpGpuKernelMod::GetOpType() {
  auto iter = kBroadcastComplexAndRealTypeMap.find(kernel_name_);
  if (iter != kBroadcastComplexAndRealTypeMap.end()) {
    op_type_ = iter->second;
    support_complex_ = true;
  }

  iter = kBroadcastComplexOnlyTypeMap.find(kernel_name_);
  if (iter != kBroadcastComplexOnlyTypeMap.end()) {
    op_type_ = iter->second;
    support_complex_ = true;
    support_real_ = false;
    return true;
  }

  iter = kBroadcastCmpTypeMap.find(kernel_name_);
  if (iter != kBroadcastCmpTypeMap.end()) {
    op_type_ = iter->second;
    is_compare_op_ = true;
    return true;
  }

  iter = kBroadcastArithmetricTypeMap.find(kernel_name_);
  if (iter != kBroadcastArithmetricTypeMap.end()) {
    op_type_ = iter->second;
    is_compare_op_ = false;
    return true;
  }

  MS_LOG(ERROR) << "For 'BroadcastGpuOp', it only support these types: " << GetValidKernelTypes()
                << " currently, but got " << kernel_name_;
  return false;
}

std::string BroadcastOpGpuKernelMod::GetValidKernelTypes() {
  std::ostringstream valid_types;
  valid_types << "Valid Compare Types: ";
  std::for_each(kBroadcastCmpTypeMap.cbegin(), kBroadcastCmpTypeMap.cend(),
                [&valid_types](const std::map<std::string, BinaryOpType>::value_type &p) {
                  valid_types << p.first << std::string(", ");
                });
  valid_types << "; Valid Arithmetric Types: ";
  std::for_each(kBroadcastArithmetricTypeMap.cbegin(), kBroadcastArithmetricTypeMap.cend(),
                [&valid_types](const std::map<std::string, BinaryOpType>::value_type &p) {
                  valid_types << p.first << std::string(", ");
                });
  valid_types << "; Valid Complex Types: ";
  std::for_each(kBroadcastComplexOnlyTypeMap.cbegin(), kBroadcastComplexOnlyTypeMap.cend(),
                [&valid_types](const std::map<std::string, BinaryOpType>::value_type &p) {
                  valid_types << p.first << std::string(", ");
                });
  return valid_types.str();
}

int BroadcastOpGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  lhs_shape_ = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  rhs_shape_ = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());
  output_shape_ = LongVecToSizeVec(outputs.at(kIndex0)->GetShapeVector());
  output_num_ = std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = CHECK_SHAPE_NULL(lhs_shape_, kernel_name_, "input_0") ||
                   CHECK_SHAPE_NULL(rhs_shape_, kernel_name_, "input_1") ||
                   CHECK_SHAPE_NULL(output_shape_, kernel_name_, "output_0");
  if (is_null_input_) {
    return KRET_OK;
  }
  need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(lhs_shape_, rhs_shape_);
  if (!broadcast_utils::AlignedBroadCastShape(MAX_DIMS, &output_shape_, &lhs_shape_, &rhs_shape_)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", and output dimension can't less than input; but got x_shape dimension:" << lhs_shape_.size()
                      << " ,y_shape dimension:" << rhs_shape_.size()
                      << " ,out_shape dimension:" << output_shape_.size();
  }
  return KRET_OK;
}

template <typename T>
bool BroadcastOpGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  auto lhs = GetDeviceAddress<T>(inputs, kIndex0);
  auto rhs = GetDeviceAddress<T>(inputs, kIndex1);

  if (is_compare_op_) {
    bool *output = GetDeviceAddress<bool>(outputs, kIndex0);
    if (need_broadcast_) {
      BroadcastCmp(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output, cuda_stream_);
    } else {
      ElewiseCmp(output_num_, op_type_, lhs, rhs, output, cuda_stream_);
    }
  } else {
    T *output = GetDeviceAddress<T>(outputs, 0);
    if (need_broadcast_) {
      BroadcastArith(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output, cuda_stream_);
    } else {
      ElewiseArith(output_num_, op_type_, lhs, rhs, output, cuda_stream_);
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda calculate error for " << kernel_name_ << ", error desc:" << cudaGetErrorString(err);
    return false;
  }
  return true;
}

template <typename T, typename S, typename G>
bool BroadcastOpGpuKernelMod::LaunchComplexKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &outputs) {
  T *lhs = GetDeviceAddress<T>(inputs, kIndex0);
  S *rhs = GetDeviceAddress<S>(inputs, kIndex1);

  G *output = GetDeviceAddress<G>(outputs, kIndex0);
  if (need_broadcast_) {
    BroadcastComplexArith(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output, cuda_stream_);
  } else {
    ElewiseComplexArith(output_num_, op_type_, lhs, rhs, output, cuda_stream_);
  }
  return true;
}

std::vector<KernelAttr> BroadcastOpGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  func_list_.clear();
  if (support_complex_) {
    (void)std::transform(complex_list_.begin(), complex_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, BroadCastFunc> &pair) { return pair.first; });
    (void)std::transform(complex_list_.begin(), complex_list_.end(), std::back_inserter(func_list_),
                         [](const std::pair<KernelAttr, BroadCastFunc> &pair) { return pair; });
  }
  if (support_real_) {
    (void)std::transform(real_list_.begin(), real_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, BroadCastFunc> &pair) { return pair.first; });
    (void)std::transform(real_list_.begin(), real_list_.end(), std::back_inserter(func_list_),
                         [](const std::pair<KernelAttr, BroadCastFunc> &pair) { return pair; });
  }
  return support_list;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, BroadcastOpGpuKernelMod::BroadCastFunc>> BroadcastOpGpuKernelMod::complex_list_ = {
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL1(kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float)},
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL2(kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float)},
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL3(kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float)},
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL1(kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double)},
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL2(kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double)},
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL3(kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double)},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
   &BroadcastOpGpuKernelMod::LaunchComplexKernel<float, float, Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
   &BroadcastOpGpuKernelMod::LaunchComplexKernel<double, double, Complex<double>>},
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL1(kNumberTypeComplex64, kNumberTypeComplex64, Complex<float>, Complex<float>)},
  {MS_REG_BROADCAST_COMPLEX_GPU_KERNEL1(kNumberTypeComplex128, kNumberTypeComplex128, Complex<double>,
                                        Complex<double>)},
};
std::vector<std::pair<KernelAttr, BroadcastOpGpuKernelMod::BroadCastFunc>> BroadcastOpGpuKernelMod::real_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &BroadcastOpGpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &BroadcastOpGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &BroadcastOpGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &BroadcastOpGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &BroadcastOpGpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &BroadcastOpGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &BroadcastOpGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &BroadcastOpGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &BroadcastOpGpuKernelMod::LaunchKernel<double>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Add, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Atan2, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AbsGrad, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BitwiseAnd, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BitwiseOr, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BitwiseXor, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Div, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DivNoNan, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Equal, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FloorMod, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FloorDiv, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Greater, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GreaterEqual, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Less, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LessEqual, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogicalOr, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogicalAnd, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Mul, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MulNoNan, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Mod, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Minimum, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Maximum, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NotEqual, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Pow, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, RealDiv, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Sub, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TruncateDiv, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TruncateMod, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Complex, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Xdivy, BroadcastOpGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Xlogy, BroadcastOpGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
