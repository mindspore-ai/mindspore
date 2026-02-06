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

#include "kernel/ascend/aclnn/pyboost_impl/customize/soft_margin_loss.h"
#include <unordered_map>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SoftMarginLossAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &target_tensor,
                                                const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "SoftMarginLoss call start";
  auto output = InferOutput<1>(prim::kPrimSoftMarginLoss, input_tensor, target_tensor, reduction);
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  auto reduction_value = ops::ConvertReductionForAclnn(reduction_imm);

  // No need to convert input
  TensorToDevice(input_tensor, target_tensor);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnSoftMarginLoss, input_tensor, target_tensor, reduction_value, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
