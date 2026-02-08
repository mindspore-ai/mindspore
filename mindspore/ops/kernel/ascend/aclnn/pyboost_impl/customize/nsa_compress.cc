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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nsa_compress.h"
#include <string>
#include <vector>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NsaCompressAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                             const TensorPtr &weight, const Int64ImmPtr &compress_block_size,
                                             const Int64ImmPtr &compress_stride,
                                             const std::optional<ValueTuplePtr> &actual_seq_len) {
  OpRunner::InferOpOutput(op, input, weight, compress_block_size, compress_stride, actual_seq_len);

  auto compress_block_size_value = GetValue<int64_t>(compress_block_size);
  auto compress_stride_value = GetValue<int64_t>(compress_stride);
  std::vector<int64_t> actual_seq_len_array;
  if (actual_seq_len.has_value()) {
    actual_seq_len_array = ConvertValueTupleToVector<int64_t>(actual_seq_len.value());
  }
  const std::string layout_string = "TND";
  const int64_t actual_seq_len_type_value = 0;
  constexpr size_t kTndInputRank = 3;

  // Runtime checks for common illegal values to provide early diagnostics
  // 1) D must be multiple of 16
  {
    const auto &in_shape = input->shape();
    if (in_shape.size() == kTndInputRank) {
      const int64_t D = static_cast<int64_t>(in_shape[2]);
      if (D % 16 != 0) {
        MS_LOG(EXCEPTION) << "For '" << op->primitive()->name()
                          << "', the last dimension D must be a multiple of 16, but got D=" << D << ".";
      }
    }
  }
  // 2) actual_seq_len last value must equal T
  if (!actual_seq_len_array.empty()) {
    const auto &in_shape = input->shape();
    if (in_shape.size() >= 1) {
      const int64_t T = static_cast<int64_t>(in_shape[0]);
      const int64_t last = actual_seq_len_array.back();
      if (last != T) {
        MS_LOG(EXCEPTION) << "For '" << op->primitive()->name()
                          << "', the last element of actual_seq_len must equal T. got last=" << last << ", T=" << T
                          << ".";
      }
    }
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, weight);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, weight, compress_block_size_value,
                                                                          compress_stride_value, actual_seq_len_array,
                                                                          layout_string, actual_seq_len_type_value]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    PyBoostUtils::MallocOpInputs(device_context, input, weight);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    MS_LOG(DEBUG) << "Run aclnnNsaCompress in pyboost";
    LAUNCH_ACLNN(aclnnNsaCompress, device_context, op->stream_id(), input, weight, actual_seq_len_array, layout_string,
                 compress_block_size_value, compress_stride_value, actual_seq_len_type_value, outputs[0]);
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
