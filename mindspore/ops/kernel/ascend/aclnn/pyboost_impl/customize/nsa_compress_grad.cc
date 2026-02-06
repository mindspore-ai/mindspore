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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nsa_compress_grad.h"
#include <string>
#include <vector>
#include <utility>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> NsaCompressGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &grad, const TensorPtr &input, const TensorPtr &weight,
  const Int64ImmPtr &compress_block_size, const Int64ImmPtr &compress_stride, const ValueTuplePtr &actual_seq_len) {
  OpRunner::InferOpOutput(op, grad, input, weight, compress_block_size, compress_stride, actual_seq_len);

  auto compress_block_size_value = GetValue<int64_t>(compress_block_size);
  auto compress_stride_value = GetValue<int64_t>(compress_stride);
  std::vector<int64_t> actual_seq_len_array;
  actual_seq_len_array = ConvertValueTupleToVector<int64_t>(actual_seq_len);
  const std::string layout_string = "TND";
  const int64_t actual_seq_len_type_value = 0;
  const auto actual_seq_len_pair = std::make_pair(actual_seq_len_array, true);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad, input, weight);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad, input, weight, compress_block_size_value, compress_stride_value, actual_seq_len_pair, layout_string,
     actual_seq_len_type_value]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      PyBoostUtils::MallocOpInputs(device_context, grad, input, weight);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << "Run aclnnNsaCompressGrad in pyboost";
      LAUNCH_ACLNN(aclnnNsaCompressGrad, device_context, op->stream_id(), grad, input, weight, actual_seq_len_pair,
                   compress_block_size_value, compress_stride_value, actual_seq_len_type_value, layout_string,
                   outputs[0], outputs[1]);
    }));
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
