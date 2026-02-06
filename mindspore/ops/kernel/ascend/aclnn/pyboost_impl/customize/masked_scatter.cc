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

#include "kernel/ascend/aclnn/pyboost_impl/customize/masked_scatter.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/contiguous.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

namespace {

tensor::TensorPtr MaskedScatterAscendCall(const std::shared_ptr<OpRunner> &op,
                                          const device::DeviceContext *device_context, const TensorPtr &x_tensor_bd,
                                          const TensorPtr &mask_tensor_bd, const TensorPtr &updates_tensor,
                                          const TensorPtr &output_tensor) {
  LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), output_tensor, x_tensor_bd);
  LAUNCH_ACLNN(aclnnInplaceMaskedScatter, device_context, op->stream_id(), output_tensor, mask_tensor_bd,
               updates_tensor);
  return output_tensor;
}
}  // namespace

tensor::TensorPtr MaskedScatterAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor,
                                               const TensorPtr &mask_tensor, const TensorPtr &updates_tensor) {
  const std::vector<int64_t> &input_shape = x_tensor->shape();
  const std::vector<int64_t> &target_shape = mask_tensor->shape();
  std::vector<int64_t> expand_shape = ops::CalBroadCastShapeV3(input_shape, target_shape);

  const ValueTuplePtr &expand_shape_ptr = ops::ConvertShapeVectorToValueTuple(expand_shape);
  MS_EXCEPTION_IF_NULL(expand_shape_ptr);

  TensorPtr x_tensor_bd = x_tensor;
  TensorPtr mask_tensor_bd = mask_tensor;

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  if (input_shape != expand_shape) {
    x_tensor_bd = broadcast_to_view(x_tensor, expand_shape);
  }
  if (target_shape != expand_shape) {
    mask_tensor_bd = broadcast_to_view(mask_tensor, expand_shape);
  }

  OpRunner::InferOpOutput(op, x_tensor_bd, mask_tensor_bd, updates_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor_bd, mask_tensor_bd, updates_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor_bd, mask_tensor_bd, updates_tensor]() {
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), x_tensor_bd, mask_tensor_bd, updates_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
      MaskedScatterAscendCall(op, device_context, x_tensor_bd, mask_tensor_bd, updates_tensor, op->output(0));
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
