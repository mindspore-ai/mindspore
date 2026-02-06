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

#include "kernel/ascend/aclnn/pyboost_impl/customize/embedding_dense_backward.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr EmbeddingDenseBackwardAscendCustomize(const TensorPtr &grad_tensor, const TensorPtr &indices_tensor,
                                                        const Int64ImmPtr &num_weights,
                                                        const std::optional<Int64ImmPtr> &padding_idx,
                                                        const BoolImmPtr &scale_grad_by_freq) {
  auto output = InferOutput<1>(prim::kPrimEmbeddingDenseBackward, grad_tensor, indices_tensor, num_weights, padding_idx,
                               scale_grad_by_freq);

  TensorToDevice(grad_tensor, indices_tensor);
  TensorAllocate(output);

  MS_EXCEPTION_IF_NULL(num_weights);
  auto num_weights_imm = num_weights->value();
  // the type of padding_idx is uint64_t in aclnnEmbeddingDenseBackward api,
  // but the type of padding_idx is int64_t in the operator belowing aclnn api where -1 indicates None value
  // this maybe a risk.
  int64_t padding_idx_imm = 0xFFFFFFFF;

  if (padding_idx.has_value()) {
    MS_EXCEPTION_IF_NULL(padding_idx.value());
    padding_idx_imm = padding_idx.value()->value();
    padding_idx_imm = padding_idx_imm < 0 ? padding_idx_imm + num_weights_imm : padding_idx_imm;
  }

  MS_EXCEPTION_IF_NULL(scale_grad_by_freq);
  auto scale_grad_by_freq_imm = scale_grad_by_freq->value();

  // Async
  DISPATCH_ACLNN(aclnnEmbeddingDenseBackward, grad_tensor, indices_tensor, static_cast<uint64_t>(num_weights_imm),
                 static_cast<uint64_t>(padding_idx_imm), scale_grad_by_freq_imm, output);

  MS_LOG(DEBUG) << "Run device task EmbeddingDenseBackward end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
