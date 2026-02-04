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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dropout_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DropoutExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                            const FP32ImmPtr &p, const TensorPtr &seed, const TensorPtr &offset) {
  OpRunner::InferOpOutput(op, input, p, seed, offset);
  // Create device address for input/output tensors.
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  auto [seed_value, offset_value] = GetGeneratorState(seed, offset);
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, p, seed_value, offset_value]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(op->device_context(), input);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
    auto p_value = static_cast<double>(p->value());

    std::vector<int64_t> input_shape = input->shape();
    auto tensor_type = op->input_abs()[kIndex0]->GetType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto dtype = tensor_type->element()->type_id();
    MS_LOG(DEBUG) << "DropoutExt input dtype = " << TypeIdToString(dtype);

    LAUNCH_ACLNN(aclnnDropoutGenMaskV2, device_context, op->stream_id(), input_shape, p_value, seed_value, offset_value,
                 dtype, outputs[kIndex1]);
    LAUNCH_ACLNN(aclnnDropoutDoMask, device_context, op->stream_id(), input, outputs[kIndex1], p_value,
                 outputs[kIndex0]);
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
