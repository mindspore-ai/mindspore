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

#include "plugin/device/ascend/kernel/pyboost/customize/chunk.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/split_tensor.h"
#include "kernel/pyboost/auto_generate/split_with_size.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> ChunkAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                        const BaseTensorPtr &input_tensor, const Int64ImmPtr &chunks,
                                                        const std::optional<Int64ImmPtr> &dim) {
  auto device_context = op->device_context();
  const auto &device_name = device_context->device_context_key_.device_name_;
  // infer
  OpRunner::InferOpOutput(op, input_tensor, chunks, dim);
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  int64_t chunks_imm = GetValue<int64_t>(chunks);
  int64_t dim_imm = 0;
  if (dim.has_value()) {
    dim_imm = GetValue<int64_t>(dim.value());
  }
  auto dim_ptr = std::make_shared<Int64Imm>(dim_imm);
  const auto &input_shape = input_tensor->shape();
  if (dim_imm < 0) {
    dim_imm += SizeToLong(input_shape.size());
  }
  int64_t dim_size = input_shape[dim_imm];
  int64_t split_size = (dim_size + chunks_imm - 1) / chunks_imm;
  MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
  std::vector<tensor::BaseTensorPtr> output{};
  if (split_size == 0 && dim_size == 0) {
    auto split_with_size_op = CREATE_PYBOOST_OP(SplitWithSize, device_name);
    auto split_sizes =
      std::make_shared<ValueTuple>(std::vector<ValuePtr>(chunks_imm, std::make_shared<Int64Imm>(split_size)));
    output = split_with_size_op->Call(input_tensor, split_sizes, dim_ptr);
    op->set_outputs(split_with_size_op->outputs());
    return output;
  }
  auto split_tensor_op = CREATE_PYBOOST_OP(SplitTensor, device_name);
  auto split_size_ptr = std::make_shared<Int64Imm>(split_size);
  output = split_tensor_op->Call(input_tensor, split_size_ptr, dim_ptr);
  op->set_outputs(split_tensor_op->outputs());
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
