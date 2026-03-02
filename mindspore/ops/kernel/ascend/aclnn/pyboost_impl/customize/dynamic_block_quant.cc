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

#include "kernel/ascend/aclnn/pyboost_impl/customize/dynamic_block_quant.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include <string>

#include "include/utils/utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void DynamicBlockQuantAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x,
                                      const FP32ImmPtr &min_scale, const StringImmPtr &round_mode,
                                      const Int64ImmPtr &dst_type, const Int64ImmPtr &row_block_size,
                                      const Int64ImmPtr &col_block_size) {
  OpRunner::InferOpOutput(op, x, min_scale, round_mode, dst_type, row_block_size, col_block_size);
  float min_scale_imm = GetValue<float>(min_scale);
  std::string round_mode_imm = GetValue<std::string>(round_mode);
  int64_t ms_dst_type_imm = GetValue<int64_t>(dst_type);
  auto acl_dst_type_imm = mindspore::device::ascend::AclConverter::ConvertType(static_cast<TypeId>(ms_dst_type_imm));
  int64_t row_block_size_imm = GetValue<int64_t>(row_block_size);
  int64_t col_block_size_imm = GetValue<int64_t>(col_block_size);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x, min_scale_imm, round_mode_imm, acl_dst_type_imm, row_block_size_imm, col_block_size_imm]() {
      MS_LOG(DEBUG) << "Run device task DynamicBlockQuant start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnDynamicBlockQuant, device_context, op->stream_id(), x, min_scale_imm, round_mode_imm,
                   acl_dst_type_imm, row_block_size_imm, col_block_size_imm, outputs[kIndex0], outputs[kIndex1]);
      MS_LOG(DEBUG) << "Run device task DynamicBlockQuant end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
