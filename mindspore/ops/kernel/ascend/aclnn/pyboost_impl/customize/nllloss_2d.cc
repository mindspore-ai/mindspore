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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nllloss_2d.h"
#include <memory>
#include <unordered_map>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> NLLLoss2dAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &logits_tensor, const TensorPtr &label_tensor,
  const TensorPtr &weight_tensor, const Int64ImmPtr &reduction, const Int64ImmPtr &ignore_index) {
  MS_LOG(DEBUG) << "NLLLoss2d call start";
  OpRunner::InferOpOutput(op, logits_tensor, label_tensor, weight_tensor, reduction, ignore_index);

  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  auto reduction_value = device::ascend::AclHelper::ConvertMsReductionToGe(reduction_imm);

  auto ignore_index_imm = GetValue<int64_t>(ignore_index);

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), logits_tensor, label_tensor, weight_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, logits_tensor, label_tensor, weight_tensor, reduction_value, ignore_index_imm]() {
      MS_LOG(DEBUG) << "Run device task NLLLoss2d start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, logits_tensor, label_tensor, weight_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      ScalarPtr alpha = std::make_shared<Int64Imm>(1);
      LAUNCH_ACLNN(aclnnNLLLoss2d, device_context, op->stream_id(), logits_tensor, label_tensor, weight_tensor,
                   reduction_value, ignore_index_imm, outputs[0], outputs[1]);
      MS_LOG(DEBUG) << "Run device task NLLLoss2d end";
    }));
  return std::make_tuple(op->output(0), op->output(1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
