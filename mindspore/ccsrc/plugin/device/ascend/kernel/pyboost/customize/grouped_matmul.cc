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

#include "plugin/device/ascend/kernel/pyboost/customize/grouped_matmul.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<BaseTensorPtr> ConvertOptiaonlValueTupleToVector(const std::optional<ValueTuplePtr> &tensor_list_opt) {
  if (tensor_list_opt.has_value()) {
    return ConvertValueTupleToVector<BaseTensorPtr>(tensor_list_opt.value());
  }
  return {};
}
}  // namespace
void GroupedMatmulAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list,
                                  const ValueTuplePtr &weight_tensor_list,
                                  const std::optional<ValueTuplePtr> &bias_tensor_list,
                                  const std::optional<ValueTuplePtr> &scale_tensor_list,
                                  const std::optional<ValueTuplePtr> &offset_tensor_list,
                                  const std::optional<ValueTuplePtr> &antiquant_scale_tensor_list,
                                  const std::optional<ValueTuplePtr> &antiquant_offset_tensor_list,
                                  const std::optional<BaseTensorPtr> &group_list_tensor, const Int64ImmPtr &split_item,
                                  const Int64ImmPtr &group_type) {
  MS_LOG(DEBUG) << "Call GroupedMatmul start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, x_tensor_list, weight_tensor_list, bias_tensor_list, scale_tensor_list,
                          offset_tensor_list, antiquant_scale_tensor_list, antiquant_offset_tensor_list,
                          group_list_tensor, split_item, group_type);

  std::vector<BaseTensorPtr> x_tensor_list_vector = ConvertValueTupleToVector<BaseTensorPtr>(x_tensor_list);
  std::vector<BaseTensorPtr> weight_tensor_list_vector = ConvertValueTupleToVector<BaseTensorPtr>(weight_tensor_list);
  std::vector<BaseTensorPtr> bias_tensor_list_vector = ConvertOptiaonlValueTupleToVector(bias_tensor_list);
  std::vector<BaseTensorPtr> scale_tensor_list_vector = ConvertOptiaonlValueTupleToVector(scale_tensor_list);
  std::vector<BaseTensorPtr> offset_tensor_list_vector = ConvertOptiaonlValueTupleToVector(offset_tensor_list);
  std::vector<BaseTensorPtr> antiquant_scale_tensor_list_vector =
    ConvertOptiaonlValueTupleToVector(antiquant_scale_tensor_list);
  std::vector<BaseTensorPtr> antiquant_offset_tensor_list_vector =
    ConvertOptiaonlValueTupleToVector(antiquant_offset_tensor_list);

  auto split_item_imm = GetValue<int64_t>(split_item);
  auto group_type_imm = GetValue<int64_t>(group_type);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor_list_vector, weight_tensor_list_vector,
                                bias_tensor_list_vector, scale_tensor_list_vector, offset_tensor_list_vector,
                                antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector,
                                group_list_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor_list_vector, weight_tensor_list_vector, bias_tensor_list_vector, scale_tensor_list_vector,
     offset_tensor_list_vector, antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector,
     group_list_tensor, split_item_imm, group_type_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor_list_vector, weight_tensor_list_vector,
                                   bias_tensor_list_vector, scale_tensor_list_vector, offset_tensor_list_vector,
                                   antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector,
                                   group_list_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnGroupedMatmulV3, device_context, op->stream_id(), x_tensor_list_vector,
                   weight_tensor_list_vector, bias_tensor_list_vector, scale_tensor_list_vector,
                   offset_tensor_list_vector, antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector,
                   group_list_tensor, split_item_imm, group_type_imm, outputs);
      MS_LOG(DEBUG) << "Launch GroupedMatmul end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
