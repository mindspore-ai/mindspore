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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/all_to_all_all_gather_batchmatmul_aclnn_kernel.h"
#include <cstdint>
#include <string>
#include "include/utils/utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
namespace mindspore {
namespace kernel {
namespace all_to_all_all_gather_batchmatmul {
void AlltoAllAllGatherBatchMatMulAscend::InitializeCommunicationAttributes() {
  group_ep_ = GetRequiredAttr<std::string>(kAttrGroupEp);
  group_tp_ = GetRequiredAttr<std::string>(kAttrGroupTp);
  hccl_inner_comm_ep_name_ = mindspore::device::ascend::OpApiUtil::GetCommName(group_ep_);
  hccl_inner_comm_tp_name_ = mindspore::device::ascend::OpApiUtil::GetCommName(group_tp_);
  ep_world_size_ = GetRequiredAttr<int64_t>(kAttrEpWorldSize);
  tp_world_size_ = GetRequiredAttr<int64_t>(kAttrTpWorldSize);
  x_shard_type_ = GetRequiredAttr<int64_t>(kAttrXShardType);
  act_type_ = GetRequiredAttr<int64_t>(kAttrActType);
  transpose_weight_ = GetRequiredAttr<bool>(kAttrTransposeWeight);
  output_y2_flag_ = GetRequiredAttr<bool>(kAttrOutputY2Flag);
  output_y3_flag_ = GetRequiredAttr<bool>(kAttrOutputY3Flag);
}

void AlltoAllAllGatherBatchMatMulAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                          const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);

  InitializeCommunicationAttributes();

  // support weight transposition
  input_x_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], false);
  input_weight_ = std::pair<KernelTensor *, bool>(inputs[kIndex1], transpose_weight_);
  output_y1_ = outputs[kIndex0];

  if (inputs.size() == kIndex2) {
    if (outputs.size() == kIndex3) {
      output_y2_ = outputs[kIndex1];
      output_y3_ = outputs[kIndex2];
      GetWorkspaceForResize(input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
                            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, output_y2_,
                            output_y3_);
    } else if (outputs.size() == kIndex1) {
      GetWorkspaceForResize(input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
                            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, nullptr, nullptr);
    } else if (output_y2_flag_) {
      output_y2_ = outputs[kIndex1];
      GetWorkspaceForResize(input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
                            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, output_y2_, nullptr);
    } else if (output_y3_flag_) {
      output_y3_ = outputs[kIndex1];
      GetWorkspaceForResize(input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
                            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, nullptr, output_y3_);
    } else {
      MS_LOG(ERROR) << "AlltoAllAllGatherBatchMatMul: The size of outputs must be 1, 2 or 3, but got "
                    << outputs.size();
    }
  } else if (inputs.size() == kIndex3) {
    if (outputs.size() == kIndex3) {
      output_y2_ = outputs[kIndex1];
      output_y3_ = outputs[kIndex2];
      GetWorkspaceForResize(input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
                            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_,
                            output_y1_, output_y2_, output_y3_);
    } else if (outputs.size() == kIndex1) {
      GetWorkspaceForResize(input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
                            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_,
                            output_y1_, nullptr, nullptr);
    } else if (output_y2_flag_) {
      output_y2_ = outputs[kIndex1];
      GetWorkspaceForResize(input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
                            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_,
                            output_y1_, output_y2_, nullptr);
    } else if (output_y3_flag_) {
      output_y3_ = outputs[kIndex1];
      GetWorkspaceForResize(input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
                            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_,
                            output_y1_, nullptr, output_y3_);
    } else {
      MS_LOG(ERROR) << "AlltoAllAllGatherBatchMatMul: The size of outputs must be 1, 2 or 3, but got "
                    << outputs.size();
    }
  } else {
    MS_LOG(ERROR) << "AlltoAllAllGatherBatchMatMul: The size of inputs must be 2 or 3, but got " << inputs.size();
  }
}

bool AlltoAllAllGatherBatchMatMulAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (mindspore::device::ascend::OpApiUtil::NeedRebuildWorkspaceSize(group_ep_, hccl_inner_comm_ep_name_) ||
      mindspore::device::ascend::OpApiUtil::NeedRebuildWorkspaceSize(group_tp_, hccl_inner_comm_tp_name_)) {
    MS_LOG(WARNING) << "Hccl inner name had changed, need rebuild workspace size";
    GetWorkSpaceInfo(inputs, outputs);
  }

  input_x_.first = inputs[kIndex0];
  input_weight_.first = inputs[kIndex1];
  output_y1_ = outputs[kIndex0];

  if (inputs.size() == kIndex2) {
    if (outputs.size() == kIndex3) {
      output_y2_ = outputs[kIndex1];
      output_y3_ = outputs[kIndex2];
      RunOp(stream_ptr, workspace, input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, output_y2_, output_y3_);
    } else if (outputs.size() == kIndex1) {
      RunOp(stream_ptr, workspace, input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, nullptr, nullptr);
    } else if (output_y2_flag_) {
      output_y2_ = outputs[kIndex1];
      RunOp(stream_ptr, workspace, input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, output_y2_, nullptr);
    } else if (output_y3_flag_) {
      output_y3_ = outputs[kIndex1];
      RunOp(stream_ptr, workspace, input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
            ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, nullptr, output_y3_);
    } else {
      MS_LOG(ERROR) << "AlltoAllAllGatherBatchMatMul: The size of outputs must be 1, 2 or 3, but got "
                    << outputs.size();
      return false;
    }
  } else if (inputs.size() == kIndex3) {
    if (outputs.size() == kIndex3) {
      output_y2_ = outputs[kIndex1];
      output_y3_ = outputs[kIndex2];
      RunOp(stream_ptr, workspace, input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, output_y2_,
            output_y3_);
    } else if (outputs.size() == kIndex1) {
      RunOp(stream_ptr, workspace, input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, nullptr,
            nullptr);
    } else if (output_y2_flag_) {
      output_y2_ = outputs[kIndex1];
      RunOp(stream_ptr, workspace, input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, output_y2_,
            nullptr);
    } else if (output_y3_flag_) {
      output_y3_ = outputs[kIndex1];
      RunOp(stream_ptr, workspace, input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
            hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, x_shard_type_, act_type_, output_y1_, nullptr,
            output_y3_);
    } else {
      MS_LOG(ERROR) << "AlltoAllAllGatherBatchMatMul: The size of outputs must be 1, 2 or 3, but got "
                    << outputs.size();
      return false;
    }
  } else {
    MS_LOG(ERROR) << "AlltoAllAllGatherBatchMatMul: The size of inputs must be 2 or 3, but got " << inputs.size();
    return false;
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AlltoAllAllGatherBatchMatMul, AlltoAllAllGatherBatchMatMulAscend);
}  // namespace all_to_all_all_gather_batchmatmul
}  // namespace kernel
}  // namespace mindspore
