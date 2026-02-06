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

#include "kernel/ascend/aclnn/pyboost_impl/customize/kl_div.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "primitive/auto_generate/gen_ops_primitive_k.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr KLDivAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &target_tensor,
                                       const Int64ImmPtr &reduction, const BoolImmPtr &log_target) {
  auto output = InferOutput<1>(prim::kPrimKLDiv, input_tensor, target_tensor, reduction, log_target);

  auto log_target_imm = GetValue<bool>(log_target);
  auto reduction_enum = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_imm = ops::ConvertReductionForAclnn(reduction_enum);

  TensorToDevice(input_tensor, target_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnKlDiv, input_tensor, target_tensor, reduction_imm, log_target_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
