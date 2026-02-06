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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_scatter_value_reduce.h"
#include <memory>
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "mindspore/ops/ops_utils/memory_overlap.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceScatterValueReduceAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                           const TensorPtr &input_tensor, const Int64ImmPtr &dim,
                                                           const TensorPtr &index_tensor, const ScalarPtr &value,
                                                           const Int64ImmPtr &reduce) {
  MS_LOG(DEBUG) << "Call InplaceScatterValueReduce start";
  // No need to call infer
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, index_tensor);
  op->set_outputs({input_tensor});

  auto dim_imm = GetValue<int64_t>(dim);
  auto reduce_imm = GetValue<int64_t>(reduce);
  // 0 means 'none' (replace) in aclnn, but should use scatter_ without reduce instead of using 'none'
  if ((reduce_imm != Reduce::ADD) && (reduce_imm != Reduce::MULTIPLY)) {
    MS_EXCEPTION(ValueError) << "For InplaceScatterValueReduce, reduce must be either 'add' or 'multiply', but got: '"
                             << mindspore::device::ascend::ScatterReduceMode::ConvertEnumToString(reduce_imm) << "'.";
  }

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_imm, index_tensor, value, reduce_imm]() {
      MS_LOG(DEBUG) << "Run device task InplaceScatterValueReduce start";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, index_tensor);
      // Check Memory Partial Overlap
      CheckMemory({input_tensor, index_tensor}, {input_tensor});
      LAUNCH_ACLNN(aclnnInplaceScatterValue, device_context, op->stream_id(), input_tensor, dim_imm, index_tensor,
                   value, reduce_imm);
      MS_LOG(DEBUG) << "Run device task InplaceScatterValueReduce end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
