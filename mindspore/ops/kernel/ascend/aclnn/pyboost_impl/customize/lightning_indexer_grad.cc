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

#include "kernel/ascend/aclnn/pyboost_impl/customize/lightning_indexer_grad.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "include/op_enum.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> LightningIndexerGradAscendCustomize(
  const tensor::TensorPtr &query, const tensor::TensorPtr &key, const tensor::TensorPtr &dy,
  const tensor::TensorPtr &sparse_indices, const tensor::TensorPtr &weights,
  const std::optional<tensor::TensorPtr> &actual_seq_lengths_query,
  const std::optional<tensor::TensorPtr> &actual_seq_lengths_key, const Int64ImmPtr &layout,
  const Int64ImmPtr &sparse_mode, const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens) {
  static auto prim = std::make_shared<Primitive>("LightningIndexerGrad");
  auto outputs = InferOutput<3>(prim, query, key, dy, sparse_indices, weights, actual_seq_lengths_query,
                                actual_seq_lengths_key, layout, sparse_mode, pre_tokens, next_tokens);

  auto layout_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout->value());
  auto sparse_mode_v = sparse_mode->value();
  auto pre_tokens_v = pre_tokens->value();
  auto next_tokens_v = next_tokens->value();
  int64_t head_num = 0;
  bool deterministic = true;

  TensorToDevice(query, key, dy, sparse_indices, weights, actual_seq_lengths_query, actual_seq_lengths_key);
  TensorMalloc(outputs);

  const auto &dq = std::get<0>(outputs);
  const auto &dk = std::get<1>(outputs);
  const auto &dw = std::get<2>(outputs);
  DISPATCH_ACLNN(aclnnLightningIndexerGrad, query, key, dy, sparse_indices, weights, actual_seq_lengths_query,
                 actual_seq_lengths_key, head_num, layout_str, sparse_mode_v, pre_tokens_v, next_tokens_v,
                 deterministic, dq, dk, dw);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
