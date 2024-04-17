/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/binary_ext_ops_gpu_kernel.h"
#include <memory>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "plugin/device/gpu/kernel/math/broadcast_public.h"

namespace mindspore {
namespace kernel {
bool BroadcastExtOptGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto iter = kBroadcastOpMap.find(kernel_name_);
  if (iter != kBroadcastOpMap.end()) {
    op_type_ = iter->second;
  } else {
    MS_LOG(ERROR) << "For BroadcastExtOptGpuKernelMod, it does not support this op: " << kernel_name_;
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

int BroadcastExtOptGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
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

template <BinaryOpType op, typename In0_t, typename In1_t, typename In2_t, typename Out_t>
bool BroadcastExtOptGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  auto lhs = GetDeviceAddress<In0_t>(inputs, kIndex0);
  auto rhs = GetDeviceAddress<In1_t>(inputs, kIndex1);
  auto alpha = GetDeviceAddress<In2_t>(inputs, kIndex2);
  auto out = GetDeviceAddress<Out_t>(outputs, kIndex0);
  auto status = BinaryExtOpWithBroadcastCudaFunc<op, In0_t, In1_t, In2_t, Out_t>(
    is_broadcast_, simplified_in0_shape_, simplified_in1_shape_, simplified_out_shape_, lhs, rhs, alpha, out,
    device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<KernelAttr> BroadcastExtOptGpuKernelMod::GetOpSupport() {
  auto iter = supported_type_map_.find(kernel_name_);
  std::vector<KernelAttr> support_list;
  if (iter != supported_type_map_.end()) {
    (void)std::transform(
      iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
      [](const std::pair<KernelAttr, BroadcastExtOptGpuKernelMod::BroadCastFunc> &item) { return item.first; });
  }
  return support_list;
}
#define MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, NUM_TYPE, TYPE, ALPHA_TYPE, ALPHA_NUM_TYPE)                       \
  {                                                                                                                  \
    KernelAttr().AddInputAttr(NUM_TYPE).AddInputAttr(NUM_TYPE).AddInputAttr(ALPHA_NUM_TYPE).AddOutputAttr(NUM_TYPE), \
      &BroadcastExtOptGpuKernelMod::LaunchKernel<OP_TYPE, TYPE, TYPE, ALPHA_TYPE, TYPE>                              \
  }

#define MS_REG_BROADCAST_OP_ALL_TYPE_EXT(OP_TYPE)                                                                  \
  MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeBool, bool, int64_t, kNumberTypeInt64),                    \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeUInt8, uint8_t, int64_t, kNumberTypeInt64),              \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeUInt16, uint16_t, int64_t, kNumberTypeInt64),            \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeUInt32, uint32_t, int64_t, kNumberTypeInt64),            \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeUInt64, uint64_t, int64_t, kNumberTypeInt64),            \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeInt8, int8_t, int64_t, kNumberTypeInt64),                \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeInt16, int16_t, int64_t, kNumberTypeInt64),              \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeInt32, int32_t, int64_t, kNumberTypeInt64),              \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeInt64, int64_t, int64_t, kNumberTypeInt64),              \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeFloat16, half, float, kNumberTypeFloat32),               \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeFloat32, float, float, kNumberTypeFloat32),              \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeFloat64, double, float, kNumberTypeFloat32),             \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeFloat16, half, int64_t, kNumberTypeInt64),               \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeFloat32, float, int64_t, kNumberTypeInt64),              \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeFloat64, double, int64_t, kNumberTypeInt64),             \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeComplex64, Complex<float>, int64_t, kNumberTypeInt64),   \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeComplex128, Complex<double>, int64_t, kNumberTypeInt64), \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeComplex64, Complex<float>, float, kNumberTypeFloat32),   \
    MS_REG_BROADCAST_OP_SAME_TYPE_EXT(OP_TYPE, kNumberTypeComplex128, Complex<double>, float, kNumberTypeFloat32)

std::map<std::string, std::vector<std::pair<KernelAttr, BroadcastExtOptGpuKernelMod::BroadCastFunc>>>
  BroadcastExtOptGpuKernelMod::supported_type_map_ = {
    {"AddExt", {MS_REG_BROADCAST_OP_ALL_TYPE_EXT(BinaryOpType::kAddExt)}},
    {"SubExt", {MS_REG_BROADCAST_OP_ALL_TYPE_EXT(BinaryOpType::kSubExt)}},
};

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AddExt,
                                 []() { return std::make_shared<BroadcastExtOptGpuKernelMod>("AddExt"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SubExt,
                                 []() { return std::make_shared<BroadcastExtOptGpuKernelMod>("SubExt"); });
}  // namespace kernel
}  // namespace mindspore
