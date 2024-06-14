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

#include "plugin/device/ascend/kernel/pyboost/customize/weight_quant_batch_matmul.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/kernel/pyboost/auto_generate/transpose.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void WeightQuantBatchMatmulV2AscendCall(const std::shared_ptr<OpRunner> &op,
                                        const device::DeviceContext *device_context, const BaseTensorPtr &x_tensor,
                                        const BaseTensorPtr &weight_tensor, const BaseTensorPtr &antiquant_scale_tensor,
                                        const std::optional<BaseTensorPtr> &antiquant_offset_tensor,
                                        const std::optional<BaseTensorPtr> &quant_scale_tensor,
                                        const std::optional<BaseTensorPtr> &quant_offset_tensor,
                                        const std::optional<BaseTensorPtr> &bias_tensor, int64_t antiquant_group_size,
                                        const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  LAUNCH_ACLNN(aclnnWeightQuantBatchMatmulV2, device_context, op->stream_id(), x_tensor, weight_tensor,
               antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor, quant_offset_tensor, bias_tensor,
               antiquant_group_size, outputs[0]);
  MS_LOG(DEBUG) << "Launch end";
}
ValueTuplePtr GetTransposePerm(const BaseTensorPtr &weight_tensor) {
  const auto &shape = weight_tensor->shape();
  int64_t size = shape.size();
  std::vector<ValuePtr> perm(size);
  if (size < 2) {
    auto zero = std::make_shared<Int64Imm>(0);
    perm[0] = MakeValue(zero);
    return std::make_shared<ValueTuple>(perm);
  }
  perm[size - 1] = MakeValue(size - 2);
  perm[size - 2] = MakeValue(size - 1);
  for (int64_t i = 0; i < size - 2; ++i) {
    perm[i] = MakeValue(i);
  }
  return std::make_shared<ValueTuple>(perm);
}
}  // namespace
tensor::BaseTensorPtr WeightQuantBatchMatmulV2AscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor, const BaseTensorPtr &weight_tensor,
  const BaseTensorPtr &antiquant_scale_tensor, const std::optional<BaseTensorPtr> &antiquant_offset_tensor,
  const std::optional<BaseTensorPtr> &quant_scale_tensor, const std::optional<BaseTensorPtr> &quant_offset_tensor,
  const std::optional<BaseTensorPtr> &bias_tensor, const BoolImmPtr &transpose_x, const BoolImmPtr &transpose_weight,
  const Int64ImmPtr &antiquant_group_size) {
  OpRunner::InferOpOutput(op, x_tensor, weight_tensor, antiquant_scale_tensor, antiquant_offset_tensor,
                          quant_scale_tensor, quant_offset_tensor, bias_tensor, transpose_x, transpose_weight,
                          antiquant_group_size);
  auto transpose_x_imm = GetValue<bool>(transpose_x);
  auto transpose_weight_imm = GetValue<bool>(transpose_weight);
  auto antiquant_group_size_imm = GetValue<int64_t>(antiquant_group_size);

  BaseTensorPtr new_weight_tensor = weight_tensor;
  auto tensor_type = op->input_abs()[kIndex1]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  if (tensor_type->element()->type_id() == kNumberTypeInt4) {
    ShapeVector weight_shape = weight_tensor->shape();
    int kInt4ShapeMul = 2;
    weight_shape.back() *= kInt4ShapeMul;
    const ShapeVector &new_weight_shape = weight_shape;
    new_weight_tensor =
      std::make_shared<tensor::Tensor>(weight_tensor->data_type(), new_weight_shape, weight_tensor->data_ptr());
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, new_weight_tensor,
                                antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor,
                                quant_offset_tensor, bias_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  BaseTensorPtr x_tensor_trans = x_tensor;
  if (transpose_x_imm) {
    const auto &device_name = device_context->device_context_key_.device_name_;
    auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
    x_tensor_trans = transpose_op->Call(x_tensor_trans, GetTransposePerm(x_tensor_trans));
  }
  BaseTensorPtr weight_tensor_trans = new_weight_tensor;
  if (transpose_weight_imm) {
    const auto &device_name = device_context->device_context_key_.device_name_;
    auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
    weight_tensor_trans = transpose_op->Call(weight_tensor_trans, GetTransposePerm(weight_tensor_trans));
  }
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor_trans, weight_tensor_trans, antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor,
     quant_offset_tensor, bias_tensor, antiquant_group_size_imm]() {
      MS_LOG(DEBUG) << "Run device task weight quant batchMatmul v2 start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor_trans, weight_tensor_trans, antiquant_scale_tensor,
                                   antiquant_offset_tensor, quant_scale_tensor, quant_offset_tensor, bias_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      WeightQuantBatchMatmulV2AscendCall(op, device_context, x_tensor_trans, weight_tensor_trans,
                                         antiquant_scale_tensor, antiquant_offset_tensor, quant_scale_tensor,
                                         quant_offset_tensor, bias_tensor, antiquant_group_size_imm, outputs);
      MS_LOG(DEBUG) << "Run device task weight quant batchMatmul v2 end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
