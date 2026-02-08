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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/quant_batch_matmul_all_reduce_aclnn_kernel.h"
#include <cstdint>
#include <string>
#include "include/utils/utils.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_util.h"

namespace mindspore {
namespace kernel {
namespace quant_batch_matmul_all_reduce {
void QuantBatchMatmulAllReduceAscend::InitializeCommonAttributes() {
  trans_a_ = GetRequiredAttr<bool>(kAttrIsTransA);
  trans_b_ = GetRequiredAttr<bool>(kAttrIsTransB);
  group_ = GetRequiredAttr<std::string>(kAttrGroup);
  hccl_inner_comm_name_ = mindspore::device::ascend::OpApiUtil::GetCommName(group_);
  reduce_op_ = GetRequiredAttr<std::string>(kAttrOp);
}

void QuantBatchMatmulAllReduceAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);

  InitializeCommonAttributes();
  input_a_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], trans_a_);
  input_b_ = std::pair<KernelTensor *, bool>(inputs[kIndex1], trans_b_);
  GetWorkspaceForResize(input_a_, input_b_, inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], inputs[kIndex5],
                        hccl_inner_comm_name_, reduce_op_, comm_turn_, stream_mode_, outputs[kIndex0]);
}

bool QuantBatchMatmulAllReduceAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (mindspore::device::ascend::OpApiUtil::NeedRebuildWorkspaceSize(group_, hccl_inner_comm_name_)) {
    MS_LOG(WARNING) << "Hccl inner name had changed, need rebuild workspace size";
    GetWorkSpaceInfo(inputs, outputs);
  }

  input_a_.first = inputs[kIndex0];
  input_b_.first = inputs[kIndex1];
  RunOp(stream_ptr, workspace, input_a_, input_b_, inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], inputs[kIndex5],
        hccl_inner_comm_name_, reduce_op_, comm_turn_, stream_mode_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(QuantBatchMatmulAllReduce, QuantBatchMatmulAllReduceAscend);
}  // namespace quant_batch_matmul_all_reduce
}  // namespace kernel
}  // namespace mindspore
