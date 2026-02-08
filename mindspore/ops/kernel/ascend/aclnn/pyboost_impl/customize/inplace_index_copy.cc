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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_index_copy.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceIndexCopyAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                  const Int64ImmPtr &dim, const TensorPtr &index,
                                                  const TensorPtr &tensor) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, index, tensor);
  op->set_outputs({input});
  auto dim_imm = dim->value();
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, dim_imm, index, tensor]() {
    auto device_context = op->device_context();

    PyBoostUtils::MallocOpInputs(device_context, input, index, tensor);

    MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
    LAUNCH_ACLNN(aclnnInplaceIndexCopy, device_context, op->stream_id(), input, dim_imm, index, tensor);
    MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
