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

#include "kernel/ascend/aclnn/pyboost_impl/customize/concat.h"
#include <vector>
#include <memory>
#include "ir/scalar.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

// Infer output dtype by promoting all input tensor dtypes
static TypeId InferConcatOutputType(const std::vector<tensor::TensorPtr> &tensors_vector) {
  TypeId out_type = tensors_vector[0]->data_type();
  for (const auto &t : tensors_vector) {
    out_type = ops::PromoteType(TypeIdToType(out_type), TypeIdToType(t->data_type()), "Concat")->type_id();
  }
  return out_type;
}

// Infer output shape by matching all non-concat dimensions and summing along concat axis
static ShapeVector InferConcatOutputShape(const std::vector<tensor::TensorPtr> &tensors_vector, size_t axis_norm) {
  const auto &first_shape = tensors_vector[0]->shape();
  ShapeVector out_shape(first_shape.begin(), first_shape.end());
  const auto first_rank = first_shape.size();
  for (size_t i = 1; i < tensors_vector.size(); ++i) {
    const auto &sh = tensors_vector[i]->shape();
    if (sh.size() != first_rank) {
      MS_EXCEPTION(ValueError) << "For 'Concat', all inputs must have same rank, but got " << sh.size() << " and "
                               << first_rank << ". The complete shapes of mismatch inputs are " << sh << " and "
                               << first_shape;
    }
    for (size_t d = 0; d < sh.size(); ++d) {
      if (d == axis_norm) {
        continue;
      }
      if (sh[d] != first_shape[d]) {
        MS_EXCEPTION(ValueError) << "For 'Concat', shapes must match on non-concat dims: " << d << ", but got " << sh[d]
                                 << " and " << first_shape[d] << ". The complete shapes of mismatch inputs are " << sh
                                 << " and " << first_shape;
      }
    }
    out_shape[axis_norm] += sh[axis_norm];
  }
  return out_shape;
}

tensor::TensorPtr ConcatAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &tensors,
                                        const Int64ImmPtr &axis) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";
  // Manual InferType & InferShape (fast path): assume inputs are a value tuple of dense tensors; no dynamic rank.
  std::vector<tensor::TensorPtr> tensors_vector = ConvertValueTupleToVector<tensor::TensorPtr>(tensors);
  if (tensors_vector.empty()) {
    MS_EXCEPTION(ValueError) << "For 'Concat', input tensors should not be empty.";
  }
  // Normalize axis
  auto axis_imm = GetValue<int64_t>(axis);
  const auto &first_shape = tensors_vector[0]->shape();
  const auto rank = SizeToLong(first_shape.size());
  if (rank == 0) {
    MS_EXCEPTION(ValueError) << "For 'Concat', zero rank tensor is not supported.";
  }
  if (!(-rank <= axis_imm && axis_imm < rank)) {
    MS_EXCEPTION(ValueError) << "For 'Concat', axis must be in range of [" << -rank << ", " << rank << "), but got "
                             << axis_imm;
  }
  size_t axis_norm = LongToSize(axis_imm >= 0 ? axis_imm : (axis_imm + rank));

  // Infer type: promote element types like InferType does
  TypeId out_type = InferConcatOutputType(tensors_vector);

  // Infer shape: match all dims except concat axis; sum on axis
  ShapeVector out_shape = InferConcatOutputShape(tensors_vector, axis_norm);

  // Create output tensor
  std::vector<tensor::TensorPtr> outputs_vec;
  PyBoostUtils::CreateOutputTensor(out_type, out_shape, &outputs_vec);
  op->set_outputs(outputs_vec);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), tensors_vector);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensors_vector, axis_norm]() {
    MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " start.";
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostDeviceTask,
                                       op->primitive()->name(), false);
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, tensors_vector);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    // Launch aclnn kernel
    LAUNCH_ACLNN(aclnnCat, device_context, op->stream_id(), tensors_vector, axis_norm, outputs[0]);
    MS_LOG(DEBUG) << "Run device task " << op->primitive()->name() << " end.";
  }));
  return op->outputs()[0];
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
