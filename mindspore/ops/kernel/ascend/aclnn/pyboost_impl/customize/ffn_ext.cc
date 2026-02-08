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

#include "kernel/ascend/aclnn/pyboost_impl/customize/ffn_ext.h"
#include <string>
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
namespace mindspore {
using mindspore::device::ascend::FFNActivationMode;
namespace kernel {
namespace pyboost {
tensor::TensorPtr FFNExtAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor, const TensorPtr &weight1_tensor,
  const TensorPtr &weight2_tensor, const std::optional<ValueTuplePtr> &expertTokens,
  const std::optional<TensorPtr> &bias1_tensor, const std::optional<TensorPtr> &bias2_tensor,
  const std::optional<TensorPtr> &scale_tensor, const std::optional<TensorPtr> &offset_tensor,
  const std::optional<TensorPtr> &deqScale1_tensor, const std::optional<TensorPtr> &deqScale2_tensor,
  const std::optional<TensorPtr> &antiquant_scale1, const std::optional<TensorPtr> &antiquant_scale2,
  const std::optional<TensorPtr> &antiquant_offset1, const std::optional<TensorPtr> &antiquant_offset2,
  const Int64ImmPtr &activation, const Int64ImmPtr &inner_precise) {
  OpRunner::InferOpOutput(op, x_tensor, expertTokens, weight1_tensor, bias1_tensor, weight2_tensor, bias2_tensor,
                          scale_tensor, offset_tensor, deqScale1_tensor, deqScale2_tensor, antiquant_scale1,
                          antiquant_scale2, antiquant_offset1, antiquant_offset2);
  std::string activation_string = "fastgelu";
  auto activation_imm = GetValue<int64_t>(activation);
  activation_string = FFNActivationMode::ConvertEnumToString(activation_imm);
  std::vector<int64_t> expertTokens_array;
  if (expertTokens.has_value()) {
    expertTokens_array = ConvertValueTupleToVector<int64_t>(expertTokens.value());
  }
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, weight1_tensor, bias1_tensor,
                                weight2_tensor, bias2_tensor, scale_tensor, offset_tensor, deqScale1_tensor,
                                deqScale2_tensor, antiquant_scale1, antiquant_scale2, antiquant_offset1,
                                antiquant_offset2);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor, weight1_tensor, weight2_tensor, expertTokens_array, bias1_tensor, bias2_tensor, scale_tensor,
     offset_tensor, deqScale1_tensor, deqScale2_tensor, antiquant_scale1, antiquant_scale2, antiquant_offset1,
     antiquant_offset2, activation_string, inner_precise]() {
      MS_LOG(DEBUG) << "Run device task FFN start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor, weight1_tensor, weight2_tensor, bias1_tensor, bias2_tensor,
                                   scale_tensor, offset_tensor, deqScale1_tensor, deqScale2_tensor, antiquant_scale1,
                                   antiquant_scale2, antiquant_offset1, antiquant_offset2);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      auto innerPrecise_ = GetValue<int64_t>(inner_precise);
      LAUNCH_ACLNN(aclnnFFN, device_context, op->stream_id(), x_tensor, weight1_tensor, weight2_tensor,
                   expertTokens_array, bias1_tensor, bias2_tensor, scale_tensor, offset_tensor, deqScale1_tensor,
                   deqScale2_tensor, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2,
                   activation_string, innerPrecise_, outputs[0]);
      MS_LOG(DEBUG) << "Run device task FFN end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
