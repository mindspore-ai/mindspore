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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nllloss_2d_grad.h"
#include <unordered_map>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NLLLoss2dGradAscendCustomize(const TensorPtr &loss_grad_tensor, const TensorPtr &logits_tensor,
                                               const TensorPtr &labels_tensor, const TensorPtr &weight_tensor,
                                               const Int64ImmPtr &reduction, const Int64ImmPtr &ignore_index,
                                               const TensorPtr &total_weight_tensor) {
  MS_LOG(DEBUG) << "NLLLoss2dGrad call start";
  auto output = InferOutput<1>(prim::kPrimNLLLoss2dGrad, loss_grad_tensor, logits_tensor, labels_tensor, weight_tensor,
                               reduction, ignore_index, total_weight_tensor);

  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  auto reduction_value = device::ascend::AclHelper::ConvertMsReductionToGe(reduction_imm);

  auto ignore_index_imm = GetValue<int64_t>(ignore_index);

  // No need to convert input
  TensorToDevice(loss_grad_tensor, logits_tensor, labels_tensor, weight_tensor, total_weight_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnNLLLoss2dBackward, loss_grad_tensor, logits_tensor, labels_tensor, weight_tensor, reduction_value,
                 ignore_index_imm, total_weight_tensor, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
