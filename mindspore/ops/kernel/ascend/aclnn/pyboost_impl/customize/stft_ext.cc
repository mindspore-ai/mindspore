/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/stft_ext.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void STFTExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                            const TensorPtr &window_tensor, const Int64ImmPtr &n_fft,
                            const Int64ImmPtr &hop_length, const Int64ImmPtr &win_length,
                            const BoolImmPtr &normalized, const BoolImmPtr &onesided,
                            const BoolImmPtr &return_complex) {
  MS_LOG(DEBUG) << "Call aclStft start";
  OpRunner::InferOpOutput(op, input_tensor, window_tensor, n_fft, hop_length, win_length, normalized, onesided,
                          return_complex);

  auto n_fft_imm = GetValue<int64_t>(n_fft);
  auto hop_length_imm = GetValue<int64_t>(hop_length);
  auto win_length_imm = GetValue<int64_t>(win_length);
  auto normalized_imm = GetValue<bool>(normalized);
  auto onesided_imm = GetValue<bool>(onesided);
  auto return_complex_imm = GetValue<bool>(return_complex);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, window_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_tensor, window_tensor, n_fft_imm, hop_length_imm, win_length_imm, normalized_imm, onesided_imm,
     return_complex_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, window_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclStft, device_context, op->stream_id(), input_tensor, window_tensor, outputs[kIndex0], n_fft_imm,
                   hop_length_imm, win_length_imm, normalized_imm, onesided_imm, return_complex_imm);
      MS_LOG(DEBUG) << "Launch aclStft end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
