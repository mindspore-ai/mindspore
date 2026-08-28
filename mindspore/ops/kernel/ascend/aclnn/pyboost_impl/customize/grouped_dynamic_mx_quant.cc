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

#include "kernel/ascend/aclnn/pyboost_impl/customize/grouped_dynamic_mx_quant.h"

#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_tensor_wrapper.h"
#include "kernel/ascend/aclnn/pyboost_impl/customize/dynamic_mx_quant_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"

namespace mindspore::kernel::pyboost {
namespace {
constexpr int64_t kDefaultScaleAlg = 0;
constexpr double kDefaultDstTypeMax = 0.0;
}  // namespace

std::tuple<tensor::TensorPtr, tensor::TensorPtr> GroupedDynamicMxQuantAscendCustomize(
  const tensor::TensorPtr &x, const tensor::TensorPtr &group_index, const StringImmPtr &round_mode,
  const Int64ImmPtr &dst_type, const Int64ImmPtr &blocksize, const Int64ImmPtr &scale_alg,
  const FP32ImmPtr &dst_type_max) {
  static const auto prim = std::make_shared<Primitive>("GroupedDynamicMxQuant");
  auto outputs = InferOutput<2>(prim, x, group_index, round_mode, dst_type, blocksize, scale_alg, dst_type_max);

  const auto round_mode_value = GetValue<std::string>(round_mode);
  const auto acl_dst_type = dynamic_mx_quant_utils::ConvertDstType(GetValue<int64_t>(dst_type));
  const auto acl_dst_type_value = static_cast<int64_t>(acl_dst_type);
  const auto blocksize_value = GetValue<int64_t>(blocksize);
  const auto scale_alg_value = GetValue<int64_t>(scale_alg);
  const auto dst_type_max_value = static_cast<double>(GetValue<float>(dst_type_max));

  TensorToDevice(x, group_index);
  TensorMalloc(outputs);
  const device::ascend::AclTensorWrapper y = {std::get<0>(outputs), acl_dst_type};
  const device::ascend::AclTensorWrapper mxscale = {std::get<1>(outputs), ACL_FLOAT8_E8M0};

  static const bool has_v2 = dynamic_mx_quant_utils::IsAclnnApiAvailable("aclnnGroupedDynamicMxQuantV2");
  static const bool has_base = dynamic_mx_quant_utils::IsAclnnApiAvailable("aclnnGroupedDynamicMxQuant");
  if (has_v2) {
    DISPATCH_ACLNN(aclnnGroupedDynamicMxQuantV2, x, group_index, round_mode_value, acl_dst_type_value, blocksize_value,
                   scale_alg_value, dst_type_max_value, y, mxscale);
  } else if (has_base) {
    if (scale_alg_value != kDefaultScaleAlg || dst_type_max_value != kDefaultDstTypeMax) {
      MS_EXCEPTION(ValueError)
        << "For 'GroupedDynamicMxQuant', the installed CANN does not provide aclnnGroupedDynamicMxQuantV2, so "
           "'scale_alg' and 'dst_type_max' must retain their default values 0 and 0.0.";
    }
    DISPATCH_ACLNN(aclnnGroupedDynamicMxQuant, x, group_index, round_mode_value, acl_dst_type_value, blocksize_value, y,
                   mxscale);
  } else {
    MS_LOG(EXCEPTION) << "For 'GroupedDynamicMxQuant', neither aclnnGroupedDynamicMxQuantV2 nor "
                         "aclnnGroupedDynamicMxQuant is available in the installed CANN.";
  }
  return outputs;
}
}  // namespace mindspore::kernel::pyboost
