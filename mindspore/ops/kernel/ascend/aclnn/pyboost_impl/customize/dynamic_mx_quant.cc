/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dynamic_mx_quant.h"
#include <mutex>
#include <string>
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_tensor_wrapper.h"
#include "kernel/ascend/aclnn/pyboost_impl/customize/dynamic_mx_quant_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"

namespace mindspore::kernel::pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> DynamicMxQuantAscendCustomize(
  const tensor::TensorPtr &input, const Int64ImmPtr &dst_type, const Int64ImmPtr &axis, const StringImmPtr &round_mode,
  const Int64ImmPtr &block_size, const Int64ImmPtr &scale_alg, const FP32ImmPtr &dst_type_max,
  const FP32ImmPtr &max_low_bound) {
  static const auto prim = std::make_shared<Primitive>("DynamicMxQuant");
  auto outputs =
    InferOutput<2>(prim, input, dst_type, axis, round_mode, block_size, scale_alg, dst_type_max, max_low_bound);

  const auto axis_value = GetValue<int64_t>(axis);
  const auto round_mode_value = GetValue<std::string>(round_mode);
  const auto acl_dst_type = dynamic_mx_quant_utils::ConvertDstType(GetValue<int64_t>(dst_type));
  const auto acl_dst_type_value = static_cast<int64_t>(acl_dst_type);
  const auto block_size_value = GetValue<int64_t>(block_size);
  const auto scale_alg_value = GetValue<int64_t>(scale_alg);
  const auto dst_type_max_value = static_cast<double>(GetValue<float>(dst_type_max));
  const auto max_low_bound_value = static_cast<double>(GetValue<float>(max_low_bound));

  TensorToDevice(input);
  TensorMalloc(outputs);
  const device::ascend::AclTensorWrapper y = {std::get<0>(outputs), acl_dst_type};
  const device::ascend::AclTensorWrapper mxscale = {std::get<1>(outputs), ACL_FLOAT8_E8M0};

  static const bool has_v3 = dynamic_mx_quant_utils::IsAclnnApiAvailable("aclnnDynamicMxQuantV3");
  static const bool has_v2 = dynamic_mx_quant_utils::IsAclnnApiAvailable("aclnnDynamicMxQuantV2");
  if (has_v3) {
    DISPATCH_ACLNN(aclnnDynamicMxQuantV3, input, axis_value, round_mode_value, acl_dst_type_value, block_size_value,
                   scale_alg_value, dst_type_max_value, max_low_bound_value, y, mxscale);
  } else if (has_v2) {
    static std::once_flag v2_warning_once;
    std::call_once(v2_warning_once, []() {
      MS_LOG(WARNING) << "aclnnDynamicMxQuantV2 does not support 'max_low_bound'; the value is ignored. "
                         "Update CANN to use this parameter.";
    });
    DISPATCH_ACLNN(aclnnDynamicMxQuantV2, input, axis_value, round_mode_value, acl_dst_type_value, block_size_value,
                   scale_alg_value, dst_type_max_value, y, mxscale);
  } else {
    static std::once_flag base_warning_once;
    std::call_once(base_warning_once, []() {
      MS_LOG(WARNING) << "aclnnDynamicMxQuant does not support 'dst_type_max' or 'max_low_bound'; "
                         "the values are ignored. Update CANN to use these parameters.";
    });
    DISPATCH_ACLNN(aclnnDynamicMxQuant, input, axis_value, round_mode_value, acl_dst_type_value, block_size_value,
                   scale_alg_value, y, mxscale);
  }
  return outputs;
}
}  // namespace mindspore::kernel::pyboost
