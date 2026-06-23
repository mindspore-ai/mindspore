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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nsa_select_attention.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> NsaSelectAttentionAscendCustomize(
  const TensorPtr &query, const TensorPtr &key, const TensorPtr &value, const TensorPtr &topk_indices,
  const FP32ImmPtr &scale_value, const Int64ImmPtr &head_num, const Int64ImmPtr &select_block_size,
  const Int64ImmPtr &select_block_count, const std::optional<TensorPtr> &atten_mask,
  const std::optional<ValueTuplePtr> &actual_seq_qlen, const std::optional<ValueTuplePtr> &actual_seq_kvlen) {
  constexpr size_t kOutputAttentionOut = 0;
  constexpr size_t kOutputSoftmaxMax = 1;
  constexpr size_t kOutputSoftmaxSum = 2;
  constexpr int64_t kNsaSelectSparseMode = 2;
  MS_LOG(DEBUG) << "NsaSelectAttention call start";
  auto outputs = InferOutput<3>(prim::kPrimNsaSelectAttention, query, key, value, topk_indices, scale_value, head_num,
                                select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen);

  auto scale_value_imm = static_cast<double>(scale_value->value());
  auto head_num_imm = head_num->value();
  auto select_block_size_imm = select_block_size->value();
  auto select_block_count_imm = select_block_count->value();
  auto actual_seq_qlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_qlen);
  auto actual_seq_kvlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_kvlen);
  auto sparse_mode = kNsaSelectSparseMode;
  std::string layout_str = "TND";

  TensorToDevice(query, key, value, topk_indices, atten_mask);
  TensorMalloc(outputs);
  DISPATCH_ACLNN(aclnnNsaSelectedAttention, query, key, value, topk_indices, atten_mask, actual_seq_qlen_array,
                 actual_seq_kvlen_array, scale_value_imm, head_num_imm, layout_str, sparse_mode, select_block_size_imm,
                 select_block_count_imm, std::get<kOutputSoftmaxMax>(outputs), std::get<kOutputSoftmaxSum>(outputs),
                 std::get<kOutputAttentionOut>(outputs));
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
