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

#include "kernel/ascend/aclnn/pyboost_impl/customize/unique_consecutive.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> UniqueConsecutiveAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const BoolImmPtr &return_inverse,
  const BoolImmPtr &return_counts, const std::optional<Int64ImmPtr> &dim) {
  MS_LOG(DEBUG) << "Run device task unique_consecutive start";
  OpRunner::InferOpOutput(op, input_tensor, return_inverse, return_counts, dim);

  auto return_inverse_imm = GetValue<bool>(return_inverse);
  auto return_counts_imm = GetValue<bool>(return_counts);

  constexpr int64_t NoneN = 1000;
  auto dim_imm = dim.has_value() ? GetValue<int64_t>(dim.value()) : NoneN;

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, return_inverse_imm, return_counts_imm, dim_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // Run sync
      const auto &all_acl_tensor =
        LAUNCH_ACLNN_SYNC(aclnnUniqueConsecutive, device_context, op->stream_id(), input_tensor, return_inverse_imm,
                          return_counts_imm, dim_imm, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);

      // update shape
      auto output_real_shape0 = all_acl_tensor[kIndex4];
      auto output_real_shape1 = all_acl_tensor[kIndex5];
      auto output_real_shape2 = all_acl_tensor[kIndex6];

      // For scalar tensor input, shape of return_inverse should be '{}', otherwise ACLNN returns a weird empty shape.
      const auto &input_tensor_shape = input_tensor->shape();
      if (input_tensor_shape.empty()) {
        output_real_shape1.clear();
      }

      auto simple_infer_ptr = op->output_value_simple_info();
      simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape0, output_real_shape1, output_real_shape2};
      op->UpdateOutputShape(op->output(kIndex0), output_real_shape0);
      op->UpdateOutputShape(op->output(kIndex1), output_real_shape1);
      op->UpdateOutputShape(op->output(kIndex2), output_real_shape2);
    }));
  runtime::Pipeline::Get().backend_stage()->Wait();
  MS_LOG(DEBUG) << "Run device task unique_consecutive end";
  return std::make_tuple(op->output(kIndex0), op->output(kIndex1), op->output(kIndex2));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
