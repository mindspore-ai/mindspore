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

#include "plugin/device/ascend/kernel/pyboost/customize/convolution.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ConvolutionAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                                 const BaseTensorPtr &weight_tensor,
                                                 const std::optional<BaseTensorPtr> &bias_tensor,
                                                 const ValueTuplePtr &stride, const ValueTuplePtr &pad,
                                                 const ValueTuplePtr &dilation, const BoolImmPtr &transposed,
                                                 const ValueTuplePtr &output_padding, const Int64ImmPtr &group) {
  OpRunner::InferOpOutput(op, input_tensor, weight_tensor, bias_tensor, stride, pad, dilation, transposed,
                          output_padding, group);
  // Convert ValueTuple to std::vector
  std::vector<int64_t> pad_vector = ConvertValueTupleToVector<int64_t>(pad);
  std::vector<int64_t> stride_vector = ConvertValueTupleToVector<int64_t>(stride);
  std::vector<int64_t> dilation_vector = ConvertValueTupleToVector<int64_t>(dilation);
  std::vector<int64_t> output_padding_vector = ConvertValueTupleToVector<int64_t>(output_padding);
  // Convert ValuePtr to c++ scalar
  auto transposed_imm = GetValue<bool>(transposed);
  auto group_imm = GetValue<int64_t>(group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, weight_tensor, bias_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_tensor, weight_tensor, bias_tensor, pad_vector, stride_vector, dilation_vector, transposed_imm,
     output_padding_vector, group_imm]() {
      MS_LOG(DEBUG) << "Run device task Convolution end";

      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor, weight_tensor, bias_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
      LAUNCH_ACLNN(aclnnConvolution, device_context, op->stream_id(), input_tensor, weight_tensor, bias_tensor,
                   stride_vector, pad_vector, dilation_vector, transposed_imm, output_padding_vector, group_imm,
                   outputs[0], GetCubeMathType());
      MS_LOG(DEBUG) << "Run device task Convolution end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
