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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inner_inplace_index_put.h"
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/op_register.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr size_t kInplaceIndexPutEmptyShape = 9;
// Remove empty tensors from the end of the indices list.
std::vector<TensorPtr> RemoveTrailingEmptyTensor(const std::vector<TensorPtr> &indices) {
  std::vector<TensorPtr> new_indices = indices;
  while (!new_indices.empty()) {
    auto back_shape = new_indices.back()->shape();
    if (back_shape.size() == kInplaceIndexPutEmptyShape &&
        std::all_of(back_shape.begin(), back_shape.end(), [](int i) { return i == 0; })) {
      new_indices.pop_back();
    } else {
      break;
    }
  }
  return new_indices;
}
}  // namespace

tensor::TensorPtr InnerInplaceIndexPutAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                      const TensorPtr &input_tensor,
                                                      const ValueTuplePtr &indices_tensor_list,
                                                      const TensorPtr &values_tensor, const BoolImmPtr &accumulate) {
  // Inplace op does not require infer, but for check broadcasts of indexes, inputs and value.
  OpRunner::InferOpOutput(op, input_tensor, indices_tensor_list, values_tensor, accumulate);
  op->set_outputs({input_tensor});

  const auto &input_shape = input_tensor->shape();
  const auto &values_shape = values_tensor->shape();
  std::vector<TensorPtr> indices_tensor_vector = ConvertValueTupleToVector<TensorPtr>(indices_tensor_list);
  indices_tensor_vector = RemoveTrailingEmptyTensor(indices_tensor_vector);
  auto input_numel =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto values_numel =
    std::accumulate(values_shape.begin(), values_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (input_numel == 0 || values_numel == 0 || indices_tensor_vector.size() == 0) {
    return op->output(0);
  }

  auto accumulate_imm = GetValue<bool>(accumulate);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, indices_tensor_vector,
                                values_tensor);

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_tensor, indices_tensor_vector, values_tensor, accumulate_imm]() {
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, indices_tensor_vector, values_tensor);

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnIndexPutImpl, device_context, op->stream_id(), input_tensor, indices_tensor_vector,
                   values_tensor, accumulate_imm, false);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
