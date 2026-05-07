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

#include "kernel/ascend/aclnn/pyboost_impl/customize/smooth_l1_loss_grad.h"
#include <unordered_map>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SmoothL1LossGradAscendCustomize(const TensorPtr &prediction_tensor, const TensorPtr &target_tensor,
                                                  const TensorPtr &dout_tensor, const FP32ImmPtr &beta,
                                                  const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "SmoothL1LossGrad call start";
  auto output =
    InferOutput<1>(prim::kPrimSmoothL1LossGrad, prediction_tensor, target_tensor, dout_tensor, beta, reduction);
  // Convert ValuePtr to c++ scalar
  auto beta_imm = GetValue<float>(beta);
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_value = ops::ConvertReductionForAclnn(reduction_imm);

  // No need to convert input
  TensorToDevice(prediction_tensor, target_tensor, dout_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnSmoothL1LossBackward, dout_tensor, prediction_tensor, target_tensor, reduction_value, beta_imm,
                 output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
