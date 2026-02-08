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
#include "kernel/ascend/aclnn/pyboost_impl/customize/matmul_allreduce_add_rmsnorm.h"
#include <string>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "kernel/ascend/acl_ir/op_api_util.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void MatmulAllReduceAddRmsNormAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x1_tensor,
                                              const TensorPtr &x2_tensor, const TensorPtr &bias_tensor,
                                              const TensorPtr &residual_tensor, const TensorPtr &gamma_tensor,
                                              const FP32ImmPtr &epsilon, const StringImmPtr &group,
                                              const Int64ImmPtr &reduction, const Int64ImmPtr &comm_turn,
                                              const Int64ImmPtr &stream_mode) {
  MS_LOG(DEBUG) << "MatmulAllReduceAddRmsNormAscendCustomize call start.";
  OpRunner::InferOpOutput(op, x1_tensor, x2_tensor, bias_tensor, residual_tensor, gamma_tensor, epsilon, group,
                          reduction, comm_turn, stream_mode);

  // Convert ValuePtr to c++ scalar
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));
  auto group_str = GetValue<std::string>(group);
  std::string group_imm = device::ascend::OpApiUtil::GetCommName(group_str);
  auto comm_turn_imm = GetValue<int64_t>(comm_turn);
  auto stream_mode_imm = GetValue<int64_t>(stream_mode);

  // transform reduction enum value to corresponding value
  auto reduction_value = device::ascend::GEReduction::ConvertEnumToString(GetValue<int64_t>(reduction));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x1_tensor, x2_tensor, bias_tensor,
                                residual_tensor, gamma_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x1_tensor, x2_tensor, bias_tensor, residual_tensor, gamma_tensor, epsilon_imm, group_imm, reduction_value,
     comm_turn_imm, stream_mode_imm]() {
      MS_LOG(DEBUG) << "Run device task MatmulAllReduceAddRmsNorm start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), x1_tensor, x2_tensor, bias_tensor, residual_tensor,
                                   gamma_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
      // for empty bias input
      if (bias_tensor->ElementsNum() == 0) {
        LAUNCH_ACLNN(aclnnMatmulAllReduceAddRmsNorm, device_context, op->stream_id(), x1_tensor, x2_tensor, nullptr,
                     residual_tensor, gamma_tensor, epsilon_imm, group_imm, reduction_value, comm_turn_imm,
                     stream_mode_imm, outputs[kIndex0], outputs[kIndex1]);
      } else {
        LAUNCH_ACLNN(aclnnMatmulAllReduceAddRmsNorm, device_context, op->stream_id(), x1_tensor, x2_tensor, bias_tensor,
                     residual_tensor, gamma_tensor, epsilon_imm, group_imm, reduction_value, comm_turn_imm,
                     stream_mode_imm, outputs[kIndex0], outputs[kIndex1]);
      }
      MS_LOG(DEBUG) << "Run device task MatmulAllReduceAddRmsNorm end";
    }));
  MS_LOG(DEBUG) << "MatmulAllReduceAddRmsNormAscendCustomize call end.";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
