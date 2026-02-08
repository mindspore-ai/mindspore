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

#include "kernel/ascend/aclnn/pyboost_impl/customize/nansum.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NansumAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                        const std::optional<ValueTuplePtr> &dim, const BoolImmPtr &keep_dim,
                                        const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, input_tensor, dim, keep_dim, dtype);

  std::vector<int64_t> dim_vector{};
  if (dim.has_value()) {
    dim_vector = ConvertValueTupleToVector<int64_t>(dim.value());
  }
  const auto keep_dim_imm = GetValue<bool>(keep_dim);
  // Infer function has confirmed the actual dtype of output
  TypeId out_dtype = op->output_value_simple_info()->dtype_vector_[kIndex0]->type_id();

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_vector, keep_dim_imm, out_dtype]() {
      auto device_context = op->device_context();

      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, op->outputs());

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnReduceNansum, device_context, op->stream_id(), input_tensor, dim_vector, keep_dim_imm,
                   out_dtype, op->output(0));
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
