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

#include "plugin/device/ascend/kernel/pyboost/customize/convolution_grad.h"
#include <memory>
#include <algorithm>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::BaseTensorPtr, tensor::BaseTensorPtr, tensor::BaseTensorPtr> ConvolutionGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &dout_tensor, const BaseTensorPtr &input_tensor,
  const BaseTensorPtr &weight_tensor, const std::optional<BaseTensorPtr> &bias_tensor, const ValueTuplePtr &stride,
  const ValueTuplePtr &pad, const ValueTuplePtr &dilation, const BoolImmPtr &transposed,
  const ValueTuplePtr &output_padding, const Int64ImmPtr &group, const ValueTuplePtr &output_mask) {
  OpRunner::InferOpOutput(op, dout_tensor, input_tensor, weight_tensor, bias_tensor, stride, pad, dilation, transposed,
                          output_padding, group, output_mask);
  // Convert ValueTuple to std::vector
  std::vector<int64_t> pad_vector = ConvertValueTupleToVector<int64_t>(pad);
  std::vector<int64_t> stride_vector = ConvertValueTupleToVector<int64_t>(stride);
  std::vector<int64_t> dilation_vector = ConvertValueTupleToVector<int64_t>(dilation);
  std::vector<int64_t> output_padding_vector = ConvertValueTupleToVector<int64_t>(output_padding);
  std::vector<int64_t> output_mask_vector = ConvertValueTupleToVector<int64_t>(output_mask);
  std::vector<uint8_t> output_mask_u8_vec;
  std::transform(output_mask_vector.begin(), output_mask_vector.end(), std::back_inserter(output_mask_u8_vec),
                 [](const int64_t &value) { return static_cast<uint8_t>(value); });
  // Convert ValuePtr to c++ scalar
  auto transposed_imm = GetValue<bool>(transposed);
  auto group_imm = GetValue<int64_t>(group);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dout_tensor, input_tensor, weight_tensor,
                                bias_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, dout_tensor, input_tensor, weight_tensor, bias_tensor, pad_vector, stride_vector, dilation_vector,
     transposed_imm, output_padding_vector, group_imm, output_mask_u8_vec]() {
      MS_LOG(DEBUG) << "Run device task ConvolutionGrad end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), dout_tensor, input_tensor, weight_tensor, bias_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());

      const auto &dout_shape = dout_tensor->shape();
      const size_t dout_dims = 4;
      if (dout_shape.size() != dout_dims) {
        MS_LOG(EXCEPTION) << "dout_shape must be 4, but got:" << dout_shape;
      }

      const size_t c_axis = 1;
      // Only `NCHW` support in Ascend, so the index of `C` is 1.
      std::vector<int64_t> bias_size = {dout_shape[c_axis]};
      LAUNCH_ACLNN(aclnnConvolutionBackward, device_context, op->stream_id(), dout_tensor, input_tensor, weight_tensor,
                   bias_size, stride_vector, pad_vector, dilation_vector, transposed_imm, output_padding_vector,
                   group_imm, output_mask_u8_vec, GetCubeMathType(), outputs[0], outputs[1], outputs[kIndex2]);
      MS_LOG(DEBUG) << "Run device task ConvolutionGrad end";
    }));
  return std::make_tuple(op->output(0), op->output(1), op->output(kIndex2));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
