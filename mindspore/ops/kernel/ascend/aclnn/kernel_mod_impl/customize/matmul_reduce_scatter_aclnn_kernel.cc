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

#include <cstdint>
#include <string>
#include <unordered_map>
#include "include/utils/utils.h"
#include "ir/tensor.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/matmul_reduce_scatter_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "mindspore/ops/infer/ops_func_impl/matmul_reduce_scatter.h"

namespace mindspore {
namespace kernel {
namespace matmul_reduce_scatter {
void MatmulReduceScatterAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);

  trans_input_ = inputs[mindspore::ops::kMatmulReduceScatterInputTransInputIndex]->GetValueWithCheck<bool>();
  trans_x2_ = inputs[mindspore::ops::kMatmulReduceScatterInputTransX2Index]->GetValueWithCheck<bool>();
  input_ = std::pair<KernelTensor *, bool>(inputs[mindspore::ops::kMatmulReduceScatterInputInputIndex], trans_input_);
  x2_ = std::pair<KernelTensor *, bool>(inputs[mindspore::ops::kMatmulReduceScatterInputX2Index], trans_x2_);
  group_ = inputs[mindspore::ops::kMatmulReduceScatterInputGroupIndex]->GetValueWithCheck<std::string>();
  hccl_inner_comm_name_ = mindspore::device::ascend::OpApiUtil::GetCommName(group_);
  world_size_ = inputs[mindspore::ops::kMatmulReduceScatterInputWorldSizeIndex]->GetValueWithCheck<int64_t>();
  auto reduction = static_cast<Reduction>(
    inputs[mindspore::ops::kMatmulReduceScatterInputReduceOpIndex]->GetValueWithCheck<int64_t>());

  mindspore::device::ascend::OpApiUtil::CheckWorldSize(group_, world_size_, primitive_->name());
  std::unordered_map<Reduction, std::string> reduction_map = {{Reduction::REDUCTION_SUM, "sum"}};
  auto iter = reduction_map.find(reduction);
  if (iter == reduction_map.end()) {
    MS_LOG(EXCEPTION) << primitive_->name() << ": the value of reduce_op is invalid.";
  }
  reduce_op_ = iter->second;
  comm_turn_ = inputs[mindspore::ops::kMatmulReduceScatterInputCommTurnIndex]->GetValueWithCheck<int64_t>();

  GetWorkspaceForResize(input_, x2_, nullptr, hccl_inner_comm_name_, reduce_op_, comm_turn_, stream_mode_,
                        outputs[mindspore::ops::kMatmulReduceScatterOutputYIndex]);
}

bool MatmulReduceScatterAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (mindspore::device::ascend::OpApiUtil::NeedRebuildWorkspaceSize(group_, hccl_inner_comm_name_)) {
    MS_LOG(WARNING) << "Hccl inner name had changed, need rebuild workspace size";
    GetWorkSpaceInfo(inputs, outputs);
  }
  // The following two lines are nessisary; deleting them will cause an error: "Sync default stream failed."
  input_.first = inputs[mindspore::ops::kMatmulReduceScatterInputInputIndex];
  x2_.first = inputs[mindspore::ops::kMatmulReduceScatterInputX2Index];
  RunOp(stream_ptr, workspace, input_, x2_, nullptr, hccl_inner_comm_name_, reduce_op_, comm_turn_, stream_mode_,
        outputs[mindspore::ops::kMatmulReduceScatterOutputYIndex]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MatmulReduceScatter, MatmulReduceScatterAscend);
}  // namespace matmul_reduce_scatter
}  // namespace kernel
}  // namespace mindspore
