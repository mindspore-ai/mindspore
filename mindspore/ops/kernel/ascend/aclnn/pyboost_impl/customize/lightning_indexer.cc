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

#include "kernel/ascend/aclnn/pyboost_impl/customize/lightning_indexer.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> LightningIndexerAscendCustomize(
  const TensorPtr &query, const TensorPtr &key, const TensorPtr &weights,
  const std::optional<TensorPtr> &actual_seq_lengths_query, const std::optional<TensorPtr> &actual_seq_lengths_key,
  const std::optional<TensorPtr> &block_table, const Int64ImmPtr &layout_query, const Int64ImmPtr &layout_key,
  const Int64ImmPtr &sparse_count, const Int64ImmPtr &sparse_mode, const Int64ImmPtr &pre_tokens,
  const Int64ImmPtr &next_tokens, const BoolImmPtr &return_value) {
  auto outputs = InferOutput<2>(prim::kPrimLightningIndexer, query, key, weights, actual_seq_lengths_query,
                                actual_seq_lengths_key, block_table, layout_query, layout_key, sparse_count,
                                sparse_mode, pre_tokens, next_tokens, return_value);

  auto layout_query_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_query->value());
  auto layout_key_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_key->value());
  auto sparse_count_v = sparse_count->value();
  auto sparse_mode_v = sparse_mode->value();
  auto pre_tokens_v = pre_tokens->value();
  auto next_tokens_v = next_tokens->value();
  auto return_value_v = return_value->value();

  TensorToDevice(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table);
  TensorAllocate(outputs);

  MS_LOG(DEBUG) << "LightningIndexer Call start";
  DISPATCH_ACLNN(aclnnLightningIndexer, query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                 block_table, layout_query_str, layout_key_str, sparse_count_v, sparse_mode_v, pre_tokens_v,
                 next_tokens_v, return_value_v, std::get<0>(outputs), std::get<1>(outputs));
  MS_LOG(DEBUG) << "LightningIndexer Launch end";
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
