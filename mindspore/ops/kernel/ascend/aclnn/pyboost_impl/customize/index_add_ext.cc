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

#include "kernel/ascend/aclnn/pyboost_impl/customize/index_add_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/auto_generate/inplace_copy.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr IndexAddExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                             const Int64ImmPtr &dim, const TensorPtr &index_tensor,
                                             const TensorPtr &source_tensor, const ScalarPtr &alpha) {
  OpRunner::InferOpOutput(op, input_tensor, dim, index_tensor, source_tensor, alpha);

  auto dim_imm = GetValue<int64_t>(dim);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, index_tensor, source_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto copy_op = CREATE_PYBOOST_OP(InplaceCopy, device::DeviceType::kAscend);
  (void)copy_op->Call(op->output(0), input_tensor, std::make_shared<BoolImm>(false));

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, index_tensor, source_tensor, dim_imm, alpha]() {
      auto device_context = op->device_context();

      PyBoostUtils::MallocOpInputs(device_context, index_tensor, source_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnIndexAdd, device_context, op->stream_id(), op->output(0), dim_imm, index_tensor, source_tensor,
                   alpha, op->output(0));
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
