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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nsa_compress_attention.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> NsaCompressAttentionAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const FP32ImmPtr &scale_value, const Int64ImmPtr &head_num, const Int64ImmPtr &compress_block_size,
  const Int64ImmPtr &compress_stride, const Int64ImmPtr &select_block_size, const Int64ImmPtr &select_block_count,
  const std::optional<TensorPtr> &topk_mask, const std::optional<TensorPtr> &atten_mask,
  const std::optional<ValueTuplePtr> &actual_seq_qlen, const std::optional<ValueTuplePtr> &actual_cmp_seq_kvlen,
  const std::optional<ValueTuplePtr> &actual_sel_seq_kvlen) {
  OpRunner::InferOpOutput(op, query, key, value, scale_value, head_num, compress_block_size, compress_stride,
                          select_block_size, select_block_count, topk_mask, atten_mask, actual_seq_qlen,
                          actual_cmp_seq_kvlen, actual_sel_seq_kvlen);

  auto scale_value_val = static_cast<double>(GetValue<float>(scale_value));
  auto head_num_val = GetValue<int64_t>(head_num);
  auto compress_block_size_val = GetValue<int64_t>(compress_block_size);
  auto compress_stride_val = GetValue<int64_t>(compress_stride);
  auto select_block_size_val = GetValue<int64_t>(select_block_size);
  auto select_block_count_val = GetValue<int64_t>(select_block_count);

  // Convert tuple to vector
  std::vector<int64_t> actual_seq_qlen_vec;
  std::vector<int64_t> actual_cmp_seq_kvlen_vec;
  std::vector<int64_t> actual_sel_seq_kvlen_vec;

  if (actual_seq_qlen.has_value()) {
    actual_seq_qlen_vec = ConvertValueTupleToVector<int64_t>(actual_seq_qlen.value());
  }
  if (actual_cmp_seq_kvlen.has_value()) {
    actual_cmp_seq_kvlen_vec = ConvertValueTupleToVector<int64_t>(actual_cmp_seq_kvlen.value());
  }
  if (actual_sel_seq_kvlen.has_value()) {
    actual_sel_seq_kvlen_vec = ConvertValueTupleToVector<int64_t>(actual_sel_seq_kvlen.value());
  }

  const auto actual_seq_qlen_pair = std::make_pair(actual_seq_qlen_vec, true);
  const auto actual_cmp_seq_kvlen_pair = std::make_pair(actual_cmp_seq_kvlen_vec, true);
  const auto actual_sel_seq_kvlen_pair = std::make_pair(actual_sel_seq_kvlen_vec, true);

  const std::string input_layout = "TND";
  const int64_t sparse_mode = 1;

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, value, atten_mask, topk_mask);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, value, scale_value_val, head_num_val, compress_block_size_val, compress_stride_val,
     select_block_size_val, select_block_count_val, atten_mask, topk_mask, actual_seq_qlen_pair,
     actual_cmp_seq_kvlen_pair, actual_sel_seq_kvlen_pair, input_layout, sparse_mode]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      PyBoostUtils::MallocOpInputs(device_context, query, key, value, atten_mask, topk_mask);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << "Run aclnnNsaCompressAttention in pyboost";
      LAUNCH_ACLNN(aclnnNsaCompressAttention, device_context, op->stream_id(), query, key, value, atten_mask, topk_mask,
                   actual_seq_qlen_pair, actual_cmp_seq_kvlen_pair, actual_sel_seq_kvlen_pair, scale_value_val,
                   head_num_val, input_layout, sparse_mode, compress_block_size_val, compress_stride_val,
                   select_block_size_val, select_block_count_val, outputs[kIndex2], outputs[kIndex3], outputs[kIndex0],
                   outputs[kIndex1]);
    }));
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
