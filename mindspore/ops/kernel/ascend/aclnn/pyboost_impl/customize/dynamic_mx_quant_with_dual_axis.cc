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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dynamic_mx_quant_with_dual_axis.h"
#include <mutex>
#include <string>
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_tensor_wrapper.h"
#include "kernel/ascend/aclnn/pyboost_impl/customize/dynamic_mx_quant_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"

namespace mindspore::kernel::pyboost {
DynamicMxQuantWithDualAxisOutputs DynamicMxQuantWithDualAxisAscendCustomize(const tensor::TensorPtr &input,
                                                                            const Int64ImmPtr &dst_type,
                                                                            const StringImmPtr &round_mode,
                                                                            const Int64ImmPtr &scale_alg,
                                                                            const FP32ImmPtr &dst_type_max) {
  static const auto prim = std::make_shared<Primitive>("DynamicMxQuantWithDualAxis");
  auto outputs = InferOutput<4>(prim, input, dst_type, round_mode, scale_alg, dst_type_max);

  const auto round_mode_value = GetValue<std::string>(round_mode);
  const auto acl_dst_type = dynamic_mx_quant_utils::ConvertDstType(GetValue<int64_t>(dst_type));
  const auto acl_dst_type_value = static_cast<int64_t>(acl_dst_type);
  const auto scale_alg_value = GetValue<int64_t>(scale_alg);
  const auto dst_type_max_value = static_cast<double>(GetValue<float>(dst_type_max));

  TensorToDevice(input);
  TensorMalloc(outputs);
  const device::ascend::AclTensorWrapper y1 = {std::get<0>(outputs), acl_dst_type};
  const device::ascend::AclTensorWrapper mxscale1 = {std::get<1>(outputs), ACL_FLOAT8_E8M0};
  const device::ascend::AclTensorWrapper y2 = {std::get<2>(outputs), acl_dst_type};
  const device::ascend::AclTensorWrapper mxscale2 = {std::get<3>(outputs), ACL_FLOAT8_E8M0};

  static const bool has_v2 = dynamic_mx_quant_utils::IsAclnnApiAvailable("aclnnDynamicMxQuantWithDualAxisV2");
  if (has_v2) {
    DISPATCH_ACLNN(aclnnDynamicMxQuantWithDualAxisV2, input, round_mode_value, acl_dst_type_value, scale_alg_value,
                   dst_type_max_value, y1, mxscale1, y2, mxscale2);
  } else {
    static std::once_flag warning_once;
    std::call_once(warning_once, []() {
      MS_LOG(WARNING) << "aclnnDynamicMxQuantWithDualAxis does not support 'dst_type_max'; the value is ignored. "
                         "Update CANN to use this parameter.";
    });
    DISPATCH_ACLNN(aclnnDynamicMxQuantWithDualAxis, input, round_mode_value, acl_dst_type_value, scale_alg_value, y1,
                   mxscale1, y2, mxscale2);
  }
  return outputs;
}
}  // namespace mindspore::kernel::pyboost
