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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inner_unique.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> InnerUniqueAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                                            const TensorPtr &input_tensor,
                                                                            const BoolImmPtr &sorted,
                                                                            const BoolImmPtr &return_inverse) {
  MS_LOG(DEBUG) << "Call InnerUnique start";
  OpRunner::InferOpOutput(op, input_tensor, sorted, return_inverse);

  auto sorted_imm = GetValue<bool>(sorted);
  auto return_inverse_imm = GetValue<bool>(return_inverse);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, sorted_imm,
                                                                          return_inverse_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    const auto &all_acl_tensor = LAUNCH_ACLNN_SYNC(aclnnUnique, device_context, op->stream_id(), input_tensor,
                                                   sorted_imm, return_inverse_imm, outputs[kIndex0], outputs[kIndex1]);

    auto value_out_real_shape = all_acl_tensor[kIndex3];
    auto inverse_out_real_shape = all_acl_tensor[kIndex4];
    auto simple_infer_ptr = op->output_value_simple_info();
    simple_infer_ptr->shape_vector_ = ShapeArray{value_out_real_shape, inverse_out_real_shape};

    op->UpdateOutputShape(op->output(kIndex0), value_out_real_shape);
    op->UpdateOutputShape(op->output(kIndex1), inverse_out_real_shape);
  }));
  runtime::Pipeline::Get().backend_stage()->Wait();
  MS_LOG(DEBUG) << "Call InnerUnique end";
  return std::make_tuple(op->output(kIndex0), op->output(kIndex1));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
