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

#include "kernel/ascend/aclnn/pyboost_impl/customize/sort_ext.h"
#include <tuple>
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> SortExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                                        const TensorPtr &input_tensor,
                                                                        const Int64ImmPtr &dim,
                                                                        const BoolImmPtr &descending,
                                                                        const BoolImmPtr &stable) {
  OpRunner::InferOpOutput(op, input_tensor, dim, descending, stable);

  auto dim_imm = GetValue<int64_t>(dim);
  auto descending_imm = GetValue<bool>(descending);
  auto stable_imm = GetValue<bool>(stable);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, stable_imm, dim_imm, descending_imm]() {
      MS_LOG(DEBUG) << "Run device task Sort start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnSort, device_context, op->stream_id(), input_tensor, stable_imm, dim_imm, descending_imm,
                   outputs[0], outputs[1]);
      MS_LOG(DEBUG) << "Run device task Sort end";
    }));
  return std::make_tuple(op->outputs()[0], op->outputs()[1]);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
