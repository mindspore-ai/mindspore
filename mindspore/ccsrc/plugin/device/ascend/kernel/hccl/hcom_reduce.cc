/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hcom_reduce.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace kernel {
bool HcomReduceKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  HcclKernel::Init(anf_node);
  if (!HcomUtil::GetHcomDestRank(anf_node, &dest_rank_)) {
    MS_LOG(ERROR) << "GetHcomDestRank fail!";
    return false;
  }
  MS_LOG(ERROR) << "Count of hcom kernel is " << hccl_count_ << ", root id is " << dest_rank_ << ", type is "
                << hccl_data_type_list_[0];
  return true;
}

bool HcomReduceKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid hccl Reduce input, output or data type size (" << inputs.size() << ", " << outputs.size()
                  << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  // Only the process with rank: dest_rank_ will receive the reduced output.
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  void *send_buf = inputs[kIndex0]->addr;
  void *recv_buf = outputs[kIndex0]->addr;

  MS_EXCEPTION_IF_NULL(send_buf);
  MS_EXCEPTION_IF_NULL(recv_buf);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduce(
    send_buf, recv_buf, hccl_count_, hccl_data_type_list_[kIndex0], op_type_, dest_rank_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclReduce failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
