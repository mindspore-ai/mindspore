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

#include "kernel/ascend/aclnn/pyboost_impl/customize/multinomial_ext.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MultinomialExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                const Int64ImmPtr &num_samples, const BoolImmPtr &replacement,
                                                const TensorPtr &seed, const TensorPtr &offset) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";
  OpRunner::InferOpOutput(op, input_tensor, num_samples, replacement, seed, offset);

  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, num_samples, replacement, seed_imm, offset_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      auto num_samples_value = static_cast<int64_t>(num_samples->value());
      auto replacement_value = static_cast<bool>(replacement->value());

      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnMultinomial, device_context, op->stream_id(), input_tensor, num_samples_value,
                   replacement_value, seed_imm, offset_imm, outputs[0]);
      MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
