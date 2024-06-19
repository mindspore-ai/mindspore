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

#include "plugin/device/gpu/kernel/math/binary_ops_gpu_kernel.h"
#include <memory>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "plugin/device/gpu/kernel/math/broadcast_public.h"

namespace mindspore {
namespace kernel {
bool BroadcastOptGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto iter = kBroadcastOpMap.find(kernel_name_);
  if (iter != kBroadcastOpMap.end()) {
    op_type_ = iter->second;
  } else {
    MS_LOG(ERROR) << "For BroadcastOptGpuKernelMod, it does not support this op: " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = supported_type_map_.find(kernel_name_)->second[index].second;
  return true;
}

int BroadcastOptGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto in0_shape = inputs[kIndex0]->GetShapeVector();
  auto in1_shape = inputs[kIndex1]->GetShapeVector();
  auto out_shape = outputs[kIndex0]->GetShapeVector();
  if (in0_shape.size() == 0) {
    in0_shape.emplace_back(1);
  }
  if (in1_shape.size() == 0) {
    in1_shape.emplace_back(1);
  }
  if (out_shape.size() == 0) {
    out_shape.emplace_back(1);
  }
  is_null_input_ = CHECK_SHAPE_NULL(in0_shape, kernel_name_, "input_0") ||
                   CHECK_SHAPE_NULL(in1_shape, kernel_name_, "input_1") ||
                   CHECK_SHAPE_NULL(out_shape, kernel_name_, "output_0");
  if (is_null_input_) {
    return KRET_OK;
  }
  SimplifyBinaryBroadcastShape(in0_shape, in1_shape, out_shape, &simplified_in0_shape_, &simplified_in1_shape_,
                               &simplified_out_shape_);
  auto input0_num = SizeOf(simplified_in0_shape_);
  auto input1_num = SizeOf(simplified_in1_shape_);
  if (input0_num > 1 && input1_num > 1 && IsBinaryBroadcast(simplified_in0_shape_, simplified_in1_shape_)) {
    is_broadcast_ = true;
  } else {
    is_broadcast_ = false;
  }
  return KRET_OK;
}

template <BinaryOpType op, typename In0_t, typename In1_t, typename Out_t>
bool BroadcastOptGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  auto lhs = GetDeviceAddress<In0_t>(inputs, kIndex0);
  auto rhs = GetDeviceAddress<In1_t>(inputs, kIndex1);
  auto out = GetDeviceAddress<Out_t>(outputs, kIndex0);
  auto status = BinaryOpWithBroadcastCudaFunc<op, In0_t, In1_t, Out_t>(is_broadcast_, simplified_in0_shape_,
                                                                       simplified_in1_shape_, simplified_out_shape_,
                                                                       lhs, rhs, out, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<KernelAttr> BroadcastOptGpuKernelMod::GetOpSupport() {
  auto iter = supported_type_map_.find(kernel_name_);
  std::vector<KernelAttr> support_list;
  if (iter != supported_type_map_.end()) {
    (void)std::transform(
      iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
      [](const std::pair<KernelAttr, BroadcastOptGpuKernelMod::BroadCastFunc> &item) { return item.first; });
  }
  return support_list;
}
#define MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, NUM_TYPE, TYPE)                          \
  {                                                                                     \
    KernelAttr().AddInputAttr(NUM_TYPE).AddInputAttr(NUM_TYPE).AddOutputAttr(NUM_TYPE), \
      &BroadcastOptGpuKernelMod::LaunchKernel<OP_TYPE, TYPE, TYPE, TYPE>                \
  }

#define MS_REG_BROADCAST_OP_DIFF_TYPE(OP_TYPE, In0_t_NUM_TYPE, In1_t_NUM_TYPE, OUT_NUM_TYPE, In0_t_TYPE, In1_t_TYPE, \
                                      OUT_TYPE)                                                                      \
  {                                                                                                                  \
    KernelAttr().AddInputAttr(In0_t_NUM_TYPE).AddInputAttr(In1_t_NUM_TYPE).AddOutputAttr(OUT_NUM_TYPE),              \
      &BroadcastOptGpuKernelMod::LaunchKernel<OP_TYPE, In0_t_TYPE, In1_t_TYPE, OUT_TYPE>                             \
  }

#define MS_REG_BROADCAST_OP_BOOL_TYPE(OP_TYPE) MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeBool, bool)

#define MS_REG_BROADCAST_OP_INT_TYPE(OP_TYPE)                            \
  MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeUInt8, uint8_t),     \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeUInt16, uint16_t), \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeUInt32, uint32_t), \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeUInt64, uint64_t), \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeInt8, int8_t),     \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeInt16, int16_t),   \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeInt32, int32_t),   \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeInt64, int64_t)

#define MS_REG_BROADCAST_OP_FLOAT_TYPE(OP_TYPE)                        \
  MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeFloat16, half),    \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeFloat32, float), \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeFloat64, double)

#define MS_REG_BROADCAST_OP_MIX_COMPLEX_TYPE(OP_TYPE, FLOAT_NUM_TYPE, COMPLEX_NUM_TYPE, FLOAT_TYPE, COMPLEX_TYPE)      \
  MS_REG_BROADCAST_OP_DIFF_TYPE(OP_TYPE, FLOAT_NUM_TYPE, COMPLEX_NUM_TYPE, COMPLEX_NUM_TYPE, FLOAT_TYPE, COMPLEX_TYPE, \
                                COMPLEX_TYPE),                                                                         \
    MS_REG_BROADCAST_OP_DIFF_TYPE(OP_TYPE, COMPLEX_NUM_TYPE, FLOAT_NUM_TYPE, COMPLEX_NUM_TYPE, COMPLEX_TYPE,           \
                                  FLOAT_TYPE, COMPLEX_TYPE)

#define MS_REG_BROADCAST_OP_COMPLEX_TYPE(OP_TYPE)                                                                   \
  MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeComplex64, Complex<float>),                                     \
    MS_REG_BROADCAST_OP_SAME_TYPE(OP_TYPE, kNumberTypeComplex128, Complex<double>),                                 \
    MS_REG_BROADCAST_OP_MIX_COMPLEX_TYPE(OP_TYPE, kNumberTypeFloat32, kNumberTypeComplex64, float, Complex<float>), \
    MS_REG_BROADCAST_OP_MIX_COMPLEX_TYPE(OP_TYPE, kNumberTypeFloat64, kNumberTypeComplex128, double, Complex<double>)

#define MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, NUM_TYPE, TYPE) \
  MS_REG_BROADCAST_OP_DIFF_TYPE(OP_TYPE, NUM_TYPE, NUM_TYPE, kNumberTypeFloat32, TYPE, TYPE, float)

#define MS_REG_BROADCAST_DIV_INT_TO_FLOAT_TYPE(OP_TYPE)                  \
  MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeUInt8, uint8_t),     \
    MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeUInt16, uint16_t), \
    MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeUInt32, uint32_t), \
    MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeUInt64, uint64_t), \
    MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeInt8, int8_t),     \
    MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeInt16, int16_t),   \
    MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeInt32, int32_t),   \
    MS_REG_BROADCAST_DIV_INT_TYPE(OP_TYPE, kNumberTypeInt64, int64_t)

#define MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, NUM_TYPE, TYPE) \
  MS_REG_BROADCAST_OP_DIFF_TYPE(OP_TYPE, NUM_TYPE, NUM_TYPE, kNumberTypeBool, TYPE, TYPE, bool)

#define MS_REG_BROADCAST_COMP_OP_INT_TYPE(OP_TYPE)                       \
  MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeBool, bool),         \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeUInt8, uint8_t),   \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeUInt16, uint16_t), \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeUInt32, uint32_t), \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeUInt64, uint64_t), \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeInt8, int8_t),     \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeInt16, int16_t),   \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeInt32, int32_t),   \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeInt64, int64_t)

#define MS_REG_BROADCAST_COMP_OP_FLOAT_TYPE(OP_TYPE)                   \
  MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeFloat16, half),    \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeFloat32, float), \
    MS_REG_BROADCAST_COMP_OP_TYPE(OP_TYPE, kNumberTypeFloat64, double)

#define MS_REG_BROADCAST_COMPARE_OP_TYPE(OP_TYPE) \
  MS_REG_BROADCAST_COMP_OP_INT_TYPE(OP_TYPE), MS_REG_BROADCAST_COMP_OP_FLOAT_TYPE(OP_TYPE)

// now only for op named Complex
#define MS_REG_BROADCAST_OP_COMPLEX(OP_TYPE)                                                                         \
  MS_REG_BROADCAST_OP_DIFF_TYPE(OP_TYPE, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeComplex64, float, float, \
                                Complex<float>),                                                                     \
    MS_REG_BROADCAST_OP_DIFF_TYPE(OP_TYPE, kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeComplex128, double,    \
                                  double, Complex<double>)

std::map<std::string, std::vector<std::pair<KernelAttr, BroadcastOptGpuKernelMod::BroadCastFunc>>>
  BroadcastOptGpuKernelMod::supported_type_map_ = {
    {"Add",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kAdd), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kAdd),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kAdd)}},
    {"Sub",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kSub), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kSub),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kSub)}},
    {"Mul",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kMul), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kMul),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kMul)}},
    {"Div",
     {MS_REG_BROADCAST_DIV_INT_TO_FLOAT_TYPE(BinaryOpType::kDiv), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kDiv),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kDiv)}},
    {"Pow",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kPow), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kPow),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kPow)}},
    {"Xdivy",
     {MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kXdivy), MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kXdivy)}},
    {"Xlogy",
     {MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kXlogy), MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kXlogy)}},
    {"RealDiv",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kRealDiv), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kRealDiv),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kRealDiv)}},
    {"MulNoNan",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kMulNoNan), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kMulNoNan),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kMulNoNan)}},
    {"Atan2",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kAtan2), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kAtan2)}},
    {"AbsGrad",
     {MS_REG_BROADCAST_OP_BOOL_TYPE(BinaryOpType::kAbsGrad), MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kAbsGrad),
      MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kAbsGrad)}},
    {"BitwiseAnd",
     {MS_REG_BROADCAST_OP_BOOL_TYPE(BinaryOpType::kBitwiseAnd),
      MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kBitwiseAnd)}},
    {"BitwiseOr",
     {MS_REG_BROADCAST_OP_BOOL_TYPE(BinaryOpType::kBitwiseOr), MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kBitwiseOr)}},
    {"BitwiseXor",
     {MS_REG_BROADCAST_OP_BOOL_TYPE(BinaryOpType::kBitwiseXor),
      MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kBitwiseXor)}},
    {"DivNoNan",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kDivNoNan), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kDivNoNan),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kDivNoNan)}},
    {"FloorMod",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kFloorMod), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kFloorMod)}},
    {"FloorDiv",
     {MS_REG_BROADCAST_OP_BOOL_TYPE(BinaryOpType::kFloorDiv), MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kFloorDiv),
      MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kFloorDiv)}},
    {"Mod", {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kMod), MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kMod)}},
    {"Minimum",
     {MS_REG_BROADCAST_OP_BOOL_TYPE(BinaryOpType::kMinimum), MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kMinimum),
      MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kMinimum)}},
    {"Maximum",
     {MS_REG_BROADCAST_OP_BOOL_TYPE(BinaryOpType::kMaximum), MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kMaximum),
      MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kMaximum)}},
    {"SquaredDifference",
     {MS_REG_BROADCAST_OP_SAME_TYPE(BinaryOpType::kSquaredDifference, kNumberTypeInt32, int32_t),
      MS_REG_BROADCAST_OP_SAME_TYPE(BinaryOpType::kSquaredDifference, kNumberTypeInt64, int64_t),
      MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kSquaredDifference),
      MS_REG_BROADCAST_OP_COMPLEX_TYPE(BinaryOpType::kSquaredDifference)}},
    {"TruncateDiv",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kTruncateDiv),
      MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kTruncateDiv)}},
    {"TruncateMod",
     {MS_REG_BROADCAST_OP_INT_TYPE(BinaryOpType::kTruncateMod),
      MS_REG_BROADCAST_OP_FLOAT_TYPE(BinaryOpType::kTruncateMod)}},
    {"Complex", {MS_REG_BROADCAST_OP_COMPLEX(BinaryOpType::kComplex)}},
    {"Greater", {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kGreater)}},
    {"Less", {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kLess)}},
    {"Equal", {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kEqual)}},
    {"GreaterEqual", {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kGreaterEqual)}},
    {"LessEqual", {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kLessEqual)}},
    {"NotEqual", {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kNotEqual)}},
    {"LogicalAnd",
     {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kLogicalAnd),
      MS_REG_BROADCAST_COMP_OP_TYPE(BinaryOpType::kLogicalAnd, kNumberTypeComplex64, Complex<float>),
      MS_REG_BROADCAST_COMP_OP_TYPE(BinaryOpType::kLogicalAnd, kNumberTypeComplex128, Complex<double>)}},
    {"LogicalOr",
     {MS_REG_BROADCAST_COMPARE_OP_TYPE(BinaryOpType::kLogicalOr),
      MS_REG_BROADCAST_COMP_OP_TYPE(BinaryOpType::kLogicalOr, kNumberTypeComplex64, Complex<float>),
      MS_REG_BROADCAST_COMP_OP_TYPE(BinaryOpType::kLogicalOr, kNumberTypeComplex128, Complex<double>)}},
};

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Add,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Add"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Div,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Div"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Mul,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Mul"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sub,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Sub"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Atan2,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Atan2"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AbsGrad,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("AbsGrad"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BitwiseAnd,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("BitwiseAnd"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BitwiseOr,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("BitwiseOr"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, BitwiseXor,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("BitwiseXor"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, DivNoNan,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("DivNoNan"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, FloorMod,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("FloorMod"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, FloorDiv,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("FloorDiv"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MulNoNan,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("MulNoNan"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Mod,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Mod"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Minimum,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Minimum"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Maximum,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Maximum"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Pow,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Pow"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, RealDiv,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("RealDiv"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, TruncateDiv,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("TruncateDiv"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, TruncateMod,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("TruncateMod"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Complex,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Complex"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Xdivy,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Xdivy"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Xlogy,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Xlogy"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Greater,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Greater"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Less,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Less"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Equal,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("Equal"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, GreaterEqual,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("GreaterEqual"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, LessEqual,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("LessEqual"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, NotEqual,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("NotEqual"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, LogicalAnd,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("LogicalAnd"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, LogicalOr,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("LogicalOr"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SquaredDifference,
                                 []() { return std::make_shared<BroadcastOptGpuKernelMod>("SquaredDifference"); });
}  // namespace kernel
}  // namespace mindspore
