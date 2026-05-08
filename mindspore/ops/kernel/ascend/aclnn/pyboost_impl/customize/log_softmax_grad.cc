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

#include "kernel/ascend/aclnn/pyboost_impl/customize/log_softmax_grad.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr LogSoftmaxGradAscendCustomize(const TensorPtr &logits_tensor, const TensorPtr &grad_tensor,
                                                const Int64ImmPtr &axis) {
  MS_LOG(DEBUG) << "LogSoftmaxGrad call start";
  auto output = InferOutput<1>(prim::kPrimLogSoftmaxGrad, logits_tensor, grad_tensor, axis);
  auto axis_imm = GetValue<int64_t>(axis);

  TensorToDevice(grad_tensor, logits_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnLogSoftmaxBackward, grad_tensor, logits_tensor, axis_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
