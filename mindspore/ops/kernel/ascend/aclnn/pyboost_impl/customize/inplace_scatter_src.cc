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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_scatter_src.h"
#include <memory>
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "mindspore/ops/ops_utils/memory_overlap.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceScatterSrcAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                   const Int64ImmPtr &dim, const TensorPtr &index_tensor,
                                                   const TensorPtr &src_tensor) {
  MS_LOG(DEBUG) << "Call InplaceScatterSrc start";
  // No need to call infer
  // forbid complex support: grad backward use aclnnGather which not supports complex
  if (!input_tensor->used_in_bprop_graph()) {
    TypeId input_type = input_tensor->Dtype()->type_id();
    if ((input_type == kNumberTypeComplex) || (input_type == kNumberTypeComplex64) ||
        (input_type == kNumberTypeComplex128)) {
      MS_EXCEPTION(RuntimeError)
        << "Complex Type is forbidden because the grad operation 'Gather' does not supports complex type";
    }
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, index_tensor, src_tensor);
  op->set_outputs({input_tensor});

  auto dim_imm = GetValue<int64_t>(dim);

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_imm, index_tensor, src_tensor]() {
      MS_LOG(DEBUG) << "Run device task InplaceScatterSrc start";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, index_tensor, src_tensor);
      // Check Memory Partial Overlap
      CheckMemory({input_tensor, index_tensor, src_tensor}, {input_tensor});
      const int64_t reduce = 0;  // reduce none
      LAUNCH_ACLNN(aclnnInplaceScatter, device_context, op->stream_id(), input_tensor, dim_imm, index_tensor,
                   src_tensor, reduce);
      MS_LOG(DEBUG) << "Run device task InplaceScatterSrc end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
