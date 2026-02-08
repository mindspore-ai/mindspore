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

#include "kernel/ascend/aclnn/pyboost_impl/customize/avg_pool3d_grad_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AvgPool3DGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &grad,
                                                  const TensorPtr &input, const ValueTuplePtr &kernel_size,
                                                  const std::optional<mindspore::ValueTuplePtr> &stride,
                                                  const ValueTuplePtr &padding, const BoolImmPtr &ceil_mode,
                                                  const BoolImmPtr &count_include_pad,
                                                  const std::optional<Int64ImmPtr> &divisor_override) {
  MS_LOG(INFO) << "AvgPool3DGradExtAscendCustomize start";
  OpRunner::InferOpOutput(op, grad, input, kernel_size, stride, padding, ceil_mode, count_include_pad,
                          divisor_override);

  auto kernel_size_val = ConvertValueTupleToVector<int64_t>(kernel_size);
  auto stride_val = stride.has_value() ? ConvertValueTupleToVector<int64_t>(stride.value()) : kernel_size_val;
  auto padding_val = ConvertValueTupleToVector<int64_t>(padding);
  auto ceil_mode_val = GetValue<bool>(ceil_mode);
  auto count_include_pad_val = GetValue<bool>(count_include_pad);
  auto divisor_override_val = divisor_override.has_value() ? GetValue<int64_t>(divisor_override.value()) : 0;

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad, input);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, grad, input, kernel_size_val, stride_val, padding_val,
                                                  ceil_mode_val, count_include_pad_val, divisor_override_val]() {
      MS_LOG(DEBUG) << "Run device task AvgPool3DGradExtAscend start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, grad, input);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // Call aclnnAvgPool2dBackward
      LAUNCH_ACLNN(aclnnAvgPool3dBackward, device_context, op->stream_id(), grad, input, kernel_size_val, stride_val,
                   padding_val, ceil_mode_val, count_include_pad_val, divisor_override_val, outputs[0]);
      MS_LOG(DEBUG) << "Run device task AvgPool3DGradExtAscend end";
    }));

  MS_LOG(INFO) << "AvgPool3DGradExtAscendCustomize end";

  return op->output(0);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
