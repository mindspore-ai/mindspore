/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dequant_swiglu_quant.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> DequantSwigluQuantAscendCustomize(
  const TensorPtr &x, const std::optional<TensorPtr> &weight_scale, const std::optional<TensorPtr> &activation_scale,
  const std::optional<TensorPtr> &bias, const std::optional<TensorPtr> &quant_scale,
  const std::optional<TensorPtr> &quant_offset, const std::optional<TensorPtr> &group_index,
  const BoolImmPtr &activate_left, const Int64ImmPtr &quant_mode) {
  MS_LOG(DEBUG) << "Call DequantSwigluQuant start";
  auto outputs = InferOutput<2>(prim::kPrimDequantSwigluQuant, x, weight_scale, activation_scale, bias, quant_scale,
                                quant_offset, group_index, activate_left, quant_mode);
  auto quant_mode_str =
    device::ascend::DequantSwigluQuantInputQuantMode::ConvertEnumToString(GetValue<int64_t>(quant_mode));
  auto activate_left_bool = GetValue<bool>(activate_left);
  TensorToDevice(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index);
  TensorMalloc(outputs);
  const auto &y = std::get<0>(outputs);
  const auto &scale = std::get<1>(outputs);
  DISPATCH_ACLNN(aclnnDequantSwigluQuant, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                 group_index, activate_left_bool, quant_mode_str, y, scale);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
