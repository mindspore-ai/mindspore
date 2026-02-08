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

#include "kernel/ascend/aclnn/pyboost_impl/customize/repeat_interleave_grad.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/auto_generate/copy.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RepeatInterleaveGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                      const TensorPtr &input_tensor, const TensorPtr &repeats,
                                                      const Int64ImmPtr &dim) {
  OpRunner::InferOpOutput(op, input_tensor, repeats, dim);
  const ShapeVector &output_shape = op->output_value_simple_info()->shape_vector_[0];
  auto repeats_tensor = repeats;
  auto repeats_shape = repeats_tensor->shape();
  if (repeats_shape.empty()) {
    repeats_tensor = std::make_shared<Tensor>(*repeats);
    repeats_tensor->set_shape(ShapeVector{1});
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, repeats_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  int64_t dim_imm = GetValue<int64_t>(dim);
  auto rank = SizeToLong(output_shape.size());
  dim_imm = (dim_imm < 0) ? (dim_imm + rank) : dim_imm;
  MS_LOG(DEBUG) << op->primitive()->name() << " Call start";

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, repeats_tensor, dim_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, repeats_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnRepeatInterleaveGrad, device_context, op->stream_id(), input_tensor, repeats_tensor, dim_imm,
                 outputs[0]);
  }));

  MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  return op->output(0);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
