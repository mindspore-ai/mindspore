/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/math/elementwise_ops_gpu_kernel.h"
#include <memory>
#include <type_traits>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

#define ADD_UNARY_SAME_TYPE(Op, NUMBER_TYPE, TYPE)                   \
  KernelAttr().AddInputAttr(NUMBER_TYPE).AddOutputAttr(NUMBER_TYPE), \
    &ElementwiseOpsGpuKernel::UnaryLaunchKernel<Op, TYPE, TYPE>

#define ADD_UNARY_DIFF_TYPE(Op, INP_NUM_TYPE, OUT_NUM_TYPE, INP_TYPE, OUT_TYPE) \
  {                                                                             \
    KernelAttr().AddInputAttr(INP_NUM_TYPE).AddOutputAttr(OUT_NUM_TYPE),        \
      &ElementwiseOpsGpuKernel::UnaryLaunchKernel<Op, INP_TYPE, OUT_TYPE>       \
  }

#define REGISTER_UNARY_FLOAT_TYPE(Op)                                                                          \
  {ADD_UNARY_SAME_TYPE(Op, kNumberTypeFloat16, half)}, {ADD_UNARY_SAME_TYPE(Op, kNumberTypeFloat32, float)}, { \
    ADD_UNARY_SAME_TYPE(Op, kNumberTypeFloat64, double)                                                        \
  }

#define REGISTER_UNARY_ALL_INT_TYPE(Op)                                                                           \
  {ADD_UNARY_SAME_TYPE(Op, kNumberTypeInt8, int8_t)}, {ADD_UNARY_SAME_TYPE(Op, kNumberTypeInt16, int16_t)},       \
    {ADD_UNARY_SAME_TYPE(Op, kNumberTypeInt32, int32_t)}, {ADD_UNARY_SAME_TYPE(Op, kNumberTypeInt64, int64_t)},   \
    {ADD_UNARY_SAME_TYPE(Op, kNumberTypeUInt8, uint8_t)}, {ADD_UNARY_SAME_TYPE(Op, kNumberTypeUInt16, uint16_t)}, \
    {ADD_UNARY_SAME_TYPE(Op, kNumberTypeUInt32, uint32_t)}, {                                                     \
    ADD_UNARY_SAME_TYPE(Op, kNumberTypeUInt64, uint64_t)                                                          \
  }

#define REGISTER_UNARY_BOOL_TYPE(Op) \
  { ADD_UNARY_SAME_TYPE(Op, kNumberTypeBool, bool) }

#define REGISTER_UNARY_COMPLEX_TYPE(Op)                              \
  {ADD_UNARY_SAME_TYPE(Op, kNumberTypeComplex64, Complex<float>)}, { \
    ADD_UNARY_SAME_TYPE(Op, kNumberTypeComplex128, Complex<double>)  \
  }

#define ADD_BINARY_SAME_TYPE(Op, NUMBER_TYPE, TYPE)                                            \
  KernelAttr().AddInputAttr(NUMBER_TYPE).AddInputAttr(NUMBER_TYPE).AddOutputAttr(NUMBER_TYPE), \
    &ElementwiseOpsGpuKernel::BinaryLaunchKernel<Op, TYPE, TYPE, TYPE>

#define REGISTER_BINARY_FLOAT_TYPE(Op)                                                                           \
  {ADD_BINARY_SAME_TYPE(Op, kNumberTypeFloat16, half)}, {ADD_BINARY_SAME_TYPE(Op, kNumberTypeFloat32, float)}, { \
    ADD_BINARY_SAME_TYPE(Op, kNumberTypeFloat64, double)                                                         \
  }

#define REGISTER_BINARY_COMPLEX_TYPE(Op)                              \
  {ADD_BINARY_SAME_TYPE(Op, kNumberTypeComplex64, Complex<float>)}, { \
    ADD_BINARY_SAME_TYPE(Op, kNumberTypeComplex128, Complex<double>)  \
  }

std::map<std::string, std::vector<std::pair<KernelAttr, ElementwiseOpsGpuKernel::OpsFunc>>>
  ElementwiseOpsGpuKernel::kernel_attr_map_ = {
    {"Sin", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSin), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSin)}},
    {"Cos", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kCos), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kCos)}},
    {"Tan", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kTan), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kTan)}},
    {"Sinh", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSinh), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSinh)}},
    {"Cosh", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kCosh), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kCosh)}},
    {"Tanh", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kTanh), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kTanh)}},
    {"Asin", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kAsin), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kAsin)}},
    {"ACos", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kAcos), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kAcos)}},
    {"Atan", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kAtan), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kAtan)}},
    {"Asinh", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kAsinh), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kAsinh)}},
    {"Acosh", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kAcosh), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kAcosh)}},
    {"Atanh", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kAtanh), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kAtanh)}},
    {"SiLU", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSiLU), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSiLU)}},
    {"Erfinv", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kErfinv)}},
    {"Erf", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kErf)}},
    {"Erfc", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kErfc)}},
    {"Abs",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kAbs), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kAbs),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kAbs), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kAbs)}},
    {"Sqrt",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kSqrt), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSqrt),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSqrt), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kSqrt),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeInt32, kNumberTypeFloat32, int, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeInt16, kNumberTypeFloat32, int16_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeInt8, kNumberTypeFloat32, int8_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeUInt8, kNumberTypeFloat32, uint8_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeUInt16, kNumberTypeFloat32, uint16_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeUInt32, kNumberTypeFloat32, uint32_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeUInt64, kNumberTypeFloat32, uint64_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kSqrt, kNumberTypeBool, kNumberTypeFloat32, bool, float)}},
    {"Invert", {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kInvert), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kInvert)}},
    {"Rsqrt", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kRsqrt), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kRsqrt)}},
    {"Sign",
     {{ADD_UNARY_SAME_TYPE(ElwiseOpType::kSign, kNumberTypeInt32, int32_t)},
      {ADD_UNARY_SAME_TYPE(ElwiseOpType::kSign, kNumberTypeInt64, int64_t)},
      REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSign),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSign)}},
    {"Square",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kSquare), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSquare),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSquare), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kSquare)}},
    {"Exp",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kExp), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kExp),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kExp), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kExp),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeInt32, kNumberTypeFloat32, int, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeInt16, kNumberTypeFloat32, int16_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeInt8, kNumberTypeFloat32, int8_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeUInt8, kNumberTypeFloat32, uint8_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeUInt16, kNumberTypeFloat32, uint16_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeUInt32, kNumberTypeFloat32, uint32_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeUInt64, kNumberTypeFloat32, uint64_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kExp, kNumberTypeBool, kNumberTypeFloat32, bool, float)}},
    {"Sigmoid",
     {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSigmoid), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSigmoid)}},
    {"ReLU",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kReLU), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kReLU),
      REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kReLU)}},
    {"Log", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kLog), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kLog)}},
    {"Log1p", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kLog1p), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kLog1p)}},
    {"Neg",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kNeg), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kNeg),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kNeg)}},
    {"Reciprocal",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kReciprocal), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kReciprocal),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kReciprocal), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kReciprocal),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeInt32, kNumberTypeFloat32, int, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeInt16, kNumberTypeFloat32, int16_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeInt8, kNumberTypeFloat32, int8_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeUInt8, kNumberTypeFloat32, uint8_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeUInt16, kNumberTypeFloat32, uint16_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeUInt32, kNumberTypeFloat32, uint32_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeUInt64, kNumberTypeFloat32, uint64_t, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReciprocal, kNumberTypeBool, kNumberTypeFloat32, bool, float)}},
    {"Inv",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kReciprocal), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kReciprocal),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kReciprocal), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kReciprocal)}},
    {"Expm1", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kExpm1), REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kExpm1)}},
    {"Mish", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kMish)}},
    {"Softsign", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSoftsign)}},
    {"Trunc",
     {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kTrunc), REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kTrunc),
      REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kTrunc)}},
    {"Floor", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kFloor)}},
    {"Ceil", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kCeil)}},
    {"OnesLike",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kOnesLike), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kOnesLike),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kOnesLike), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kOnesLike)}},
    {"Rint", {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kRint)}},
    {"LogicalNot",
     {ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeInt64, kNumberTypeBool, int64_t, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeInt32, kNumberTypeBool, int, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeInt16, kNumberTypeBool, int16_t, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeInt8, kNumberTypeBool, int8_t, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeUInt8, kNumberTypeBool, uint8_t, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeUInt16, kNumberTypeBool, uint16_t, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeUInt32, kNumberTypeBool, uint32_t, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeUInt64, kNumberTypeBool, uint64_t, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeBool, kNumberTypeBool, bool, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeFloat16, kNumberTypeBool, half, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeFloat32, kNumberTypeBool, float, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeFloat64, kNumberTypeBool, double, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeComplex64, kNumberTypeBool, Complex<float>, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeComplex128, kNumberTypeBool, Complex<double>, bool),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kLogicalNot, kNumberTypeBool, kNumberTypeBool, bool, bool)}},
    {"Conj",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kConj), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kConj),
      REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kConj), REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kConj)}},
    {"Imag",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kImag), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kImag),
      REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kImag),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kImag, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kImag, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double)}},
    {"Real",
     {REGISTER_UNARY_ALL_INT_TYPE(ElwiseOpType::kReal), REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kReal),
      REGISTER_UNARY_BOOL_TYPE(ElwiseOpType::kReal),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReal, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kReal, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>, double)}},
    {"ComplexAbs",
     {ADD_UNARY_DIFF_TYPE(ElwiseOpType::kComplexAbs, kNumberTypeComplex64, kNumberTypeFloat32, Complex<float>, float),
      ADD_UNARY_DIFF_TYPE(ElwiseOpType::kComplexAbs, kNumberTypeComplex128, kNumberTypeFloat64, Complex<double>,
                          double)}},
    {"AsinGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kAsinGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kAsinGrad)}},
    {"ACosGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kACosGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kACosGrad)}},
    {"AtanGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kAtanGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kAtanGrad)}},
    {"AsinhGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kAsinhGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kAsinhGrad)}},
    {"AcoshGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kAcoshGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kAcoshGrad)}},
    {"TanhGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kTanhGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kTanhGrad)}},
    {"SqrtGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kSqrtGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kSqrtGrad)}},
    {"RsqrtGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kRsqrtGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kRsqrtGrad)}},
    {"ReciprocalGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kReciprocalGrad),
      REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kReciprocalGrad)}},
    {"InvGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kReciprocalGrad),
      REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kReciprocalGrad)}},  // same as kReciprocalGrad
    {"Zeta",
     {{ADD_BINARY_SAME_TYPE(ElwiseOpType::kZeta, kNumberTypeFloat32, float)},
      {ADD_BINARY_SAME_TYPE(ElwiseOpType::kZeta, kNumberTypeFloat64, double)}}},
    {"SigmoidGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kSigmoidGrad),
      REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kSigmoidGrad)}},
    {"SiLUGrad",
     {REGISTER_BINARY_FLOAT_TYPE(ElwiseOpType::kSiLUGrad), REGISTER_BINARY_COMPLEX_TYPE(ElwiseOpType::kSiLUGrad)}},
};
bool ElementwiseOpsGpuKernel::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'elementwise op', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, ElementwiseOpsGpuKernel::OpsFunc>>>(
                       kernel_attr_map_)
                  << ", but got " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = iter->second[index].second;
  return true;
}

int ElementwiseOpsGpuKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ele_num_ = SizeOf(inputs.at(kIndex0)->GetShapeVector());
  is_null_input_ = (ele_num_ == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  return KRET_OK;
}

std::vector<KernelAttr> ElementwiseOpsGpuKernel::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'elementwise op', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, ElementwiseOpsGpuKernel::OpsFunc>>>(
                       kernel_attr_map_)
                  << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, OpsFunc> &item) { return item.first; });
  return support_list;
}
template <ElwiseOpType Op, typename Inp_t, typename Out_t>
bool ElementwiseOpsGpuKernel::UnaryLaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                const std::vector<kernel::KernelTensor *> &outputs) {
  auto input_ptr = reinterpret_cast<Inp_t *>(inputs.at(kIndex0)->device_ptr());
  auto output_ptr = reinterpret_cast<Out_t *>(outputs.at(kIndex0)->device_ptr());
  MS_EXCEPTION_IF_NULL(input_ptr);
  MS_EXCEPTION_IF_NULL(output_ptr);
  auto ret =
    UnaryOpsCudaFunc<Op, Inp_t, Out_t>(ele_num_, input_ptr, output_ptr, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(ret, kernel_name_);
  return true;
}
template <ElwiseOpType Op, typename In0_t, typename In1_t, typename Out_t>
bool ElementwiseOpsGpuKernel::BinaryLaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                 const std::vector<kernel::KernelTensor *> &outputs) {
  auto in0_ptr = reinterpret_cast<In0_t *>(inputs.at(kIndex0)->device_ptr());
  auto in1_ptr = reinterpret_cast<In1_t *>(inputs.at(kIndex1)->device_ptr());
  auto out_ptr = reinterpret_cast<Out_t *>(outputs.at(kIndex0)->device_ptr());
  MS_EXCEPTION_IF_NULL(in0_ptr);
  MS_EXCEPTION_IF_NULL(in1_ptr);
  MS_EXCEPTION_IF_NULL(out_ptr);
  auto ret = BinaryOpsCudaFunc<Op, In0_t, In1_t, Out_t>(ele_num_, in0_ptr, in1_ptr, out_ptr,
                                                        reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(ret, kernel_name_);
  return true;
}
#define MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(kernel)       \
  MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, kernel, \
                                   []() { return std::make_shared<ElementwiseOpsGpuKernel>(#kernel); })

MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Sin);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Cos);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Tan);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Sinh);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Cosh);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Tanh);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Asin);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(ACos);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Atan);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Asinh);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Acosh);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Atanh);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Erfinv);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Erf);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Erfc);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Abs);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Sqrt);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Inv);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Invert);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Rsqrt);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Sign);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Square);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Exp);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Sigmoid);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(ReLU);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Log);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Log1p);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Neg);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Reciprocal);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Expm1);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Mish);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Softsign);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Trunc);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Floor);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Ceil);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(OnesLike);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Rint);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(LogicalNot);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Conj);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(SiLU);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Imag);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Real);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(ComplexAbs);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(AsinGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(ACosGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(AtanGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(AsinhGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(AcoshGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(TanhGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(SqrtGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(RsqrtGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(ReciprocalGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(InvGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Zeta);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(SigmoidGrad);
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(SiLUGrad);
}  // namespace kernel
}  // namespace mindspore
