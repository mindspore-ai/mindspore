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

#include "plugin/device/ascend/kernel/pyboost/customize/repeat_interleave_int.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/copy.h"
#include "kernel/pyboost/auto_generate/view.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr RepeatInterleaveIntAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                         const BaseTensorPtr &input_tensor, const Int64ImmPtr &repeats,
                                                         const std::optional<Int64ImmPtr> &dim,
                                                         const std::optional<Int64ImmPtr> &output_size) {
  OpRunner::InferOpOutput(op, input_tensor, repeats, dim, output_size);
  const ShapeVector &output_shape = op->output_value_simple_info()->shape_vector_[0];
  int64_t repeats_imm = GetValue<int64_t>(repeats);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  MS_LOG(DEBUG) << op->primitive()->name() << " Call start";

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, repeats_imm, dim, output_size, output_shape]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      if (dim.has_value()) {
        int64_t dim_imm = GetValue<int64_t>(*dim);
        auto rank = SizeToLong(output_shape.size());
        dim_imm = (dim_imm < 0) ? (dim_imm + rank) : dim_imm;
        if (output_size.has_value()) {
          int64_t output_size_imm = GetValue<int64_t>(*output_size);
          if (output_size_imm != output_shape[dim_imm]) {
            MS_EXCEPTION(RuntimeError) << "Allocated size does not match required size.";
          }
        }

        LAUNCH_ACLNN(aclnnRepeatInterleaveIntWithDim, device_context, op->stream_id(), input_tensor, repeats_imm,
                     dim_imm, output_shape[dim_imm], outputs[0]);
      } else {
        if (output_size.has_value()) {
          int64_t output_size_imm = GetValue<int64_t>(*output_size);
          if (output_size_imm != output_shape[0]) {
            MS_EXCEPTION(RuntimeError) << "Allocated size does not match required size.";
          }
        }

        LAUNCH_ACLNN(aclnnRepeatInterleaveInt, device_context, op->stream_id(), input_tensor, repeats_imm,
                     output_shape[0], outputs[0]);
      }
    }));

  MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  return op->output(0);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
